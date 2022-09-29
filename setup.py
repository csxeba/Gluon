from setuptools import setup
from setuptools import find_packages

setup(
    name='gluon',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/csxeba/Gluon.git',
    license='MIT',
    author='Csxeba',
    author_email='csxeba@gmail.com',
    description='Experiment framework',
    install_requires=["opencv-python>=4.0",
                      "numpy",
                      "scipy",
                      "matplotlib",
                      "pytorch>=1.12",
                      "torchvision>=0.13",
                      "tqdm",
                      "git+https://github.com/csxeba/Artifactorium.git"],
)
