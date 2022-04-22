from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spharmnet",
    version="0.1.0",
    license="Apache 2.0",
    author="Ilwoo Lyu",
    author_email="ilwoolyu@unist.ac.kr",
    description="SPHARM-Net: Spherical Harmonics-based Convolutional Neural Network",
    url="https://github.com/Shape-Lab/SPHARM-Net",
    keywords=["spherical cnn", "equivariant convolution", "cortical parcellation"],
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.11.3",
        "scipy>=1.2.1",
        "joblib>=0.14.1",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
)
