import os
from typing import List, Optional, Tuple
import copy
import hashlib

import torch
import torch.distributed as dist

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0

def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def set_envs(args):
    init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    return device

def set_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    norm_weight_decay: Optional[float] = None,
    norm_classes: Optional[List[type]] = None,
    custom_keys_weight_decay: Optional[List[Tuple[str, float]]] = None,
):
    if not norm_classes:
        norm_classes = [
            torch.nn.modules.batchnorm._BatchNorm,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.modules.instancenorm._InstanceNorm,
            torch.nn.LocalResponseNorm,
        ]
    norm_classes = tuple(norm_classes)

    params = {
        "other": [],
        "norm": [],
    }
    params_weight_decay = {
        "other": weight_decay,
        "norm": norm_weight_decay,
    }
    custom_keys = []
    if custom_keys_weight_decay is not None:
        for key, weight_decay in custom_keys_weight_decay:
            params[key] = []
            params_weight_decay[key] = weight_decay
            custom_keys.append(key)

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            is_custom_key = False
            for key in custom_keys:
                target_name = f"{prefix}.{name}" if prefix != "" and "." in key else name
                if key == target_name:
                    params[key].append(p)
                    is_custom_key = True
                    break
            if not is_custom_key:
                if norm_weight_decay is not None and isinstance(module, norm_classes):
                    params["norm"].append(p)
                else:
                    params["other"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append({"params": params[key], "weight_decay": params_weight_decay[key]})
    return param_groups


def store_model_weights(model, checkpoint_path, checkpoint_key="model", strict=True):
    """
    This method can be used to prepare weights files for new models. It receives as
    input a model architecture and a checkpoint from the training script and produces
    a file with the weights ready for release.

    Examples:
        from torchvision import models as M

        # Classification
        model = M.mobilenet_v3_large(weights=None)
        print(store_model_weights(model, './class.pth'))

        # Quantized Classification
        model = M.quantization.mobilenet_v3_large(weights=None, quantize=False)
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
        _ = torch.ao.quantization.prepare_qat(model, inplace=True)
        print(store_model_weights(model, './qat.pth'))

        # Object Detection
        model = M.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
        print(store_model_weights(model, './obj.pth'))

        # Segmentation
        model = M.segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=True)
        print(store_model_weights(model, './segm.pth', strict=False))

    Args:
        model (pytorch.nn.Module): The model on which the weights will be loaded for validation purposes.
        checkpoint_path (str): The path of the checkpoint we will load.
        checkpoint_key (str, optional): The key of the checkpoint where the model weights are stored.
            Default: "model".
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``True``

    Returns:
        output_path (str): The location where the weights are saved.
    """
    # Store the new model next to the checkpoint_path
    checkpoint_path = os.path.abspath(checkpoint_path)
    output_dir = os.path.dirname(checkpoint_path)

    # Deep copy to avoid side effects on the model object.
    model = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Load the weights to the model to validate that everything works
    # and remove unnecessary weights (such as auxiliaries, etc.)
    if checkpoint_key == "model_ema":
        del checkpoint[checkpoint_key]["n_averaged"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint[checkpoint_key], "module.")
    model.load_state_dict(checkpoint[checkpoint_key], strict=strict)

    tmp_path = os.path.join(output_dir, str(model.__hash__()))
    torch.save(model.state_dict(), tmp_path)

    sha256_hash = hashlib.sha256()
    with open(tmp_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
        hh = sha256_hash.hexdigest()

    output_path = os.path.join(output_dir, "weights-" + str(hh[:8]) + ".pth")
    os.replace(tmp_path, output_path)

    return output_path

