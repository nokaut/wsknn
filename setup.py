import setuptools
from wsknn import __version__ as package_version


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wsknn",
    version=package_version,
    author="Szymon Moliński",
    author_email="s.molinski@digitree.pl",
    description="VSKNN model for recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.sare25.com/s.molinski/vsknn-model",
    packages=setuptools.find_packages(
        exclude=['dev']
    ),
    install_requires=[
        'numpy',
        'pyyaml'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
