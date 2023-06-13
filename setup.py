from typing import List
from setuptools import find_packages,setup

HYPEN_E_DOT ='-e.'

def get_requires(filePath:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements  = []
    with open(filePath) as file:
        requirements = file.readline()
        requirements=[req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='RizzWann',
    author_email='rizzwann1245@gmail.com',
    packages=find_packages(),
    install_requires=get_requires('requirements.txt')
)