import re 
from pathlib import Path 

from setuptools import setup 

FILE = Path(__file__).resolve()
PARENT = FILE.parent 
README = (PARENT / 'README.md').read_text(encoding='utf-8')

def get_version():
    file = PARENT / 'dl_framework/__init__.py'
    
    return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', file.read_text(encoding='utf-8'), re.M)[1]

def parse_requirements(file_path: Path):
    
    requirements = []
    for line in Path(file_path).read_text().splitlines():
        line = line.strip()
        
        if line and not line.startswith('#'):
            requirements.append(line.split("#")[0].strip())
            
    
    return requirements 

version=get_version()
packages=['dl_framework'] + [str(x) for x in Path('dl_framework').rglob('*/') if x.is_dir() and '__' not in str(x)]
install_requires=parse_requirements(PARENT / 'requirements.txt')

setup(
    version=get_version(),
    python_requires='>=3.9',
    description=('Template for Python dl_framework'),
    long_description=README,
    long_description_content_type='text/makrdown',
    packages=['dl_framework'] + [str(x) for x in Path('athena').rglob('*/') if x.is_dir() and '__' not in str(x)],
    package_data={
        '': ['*.yaml'],
    },
    include_package_data=True,
    install_requires=parse_requirements(PARENT / 'requirements.txt'),
)