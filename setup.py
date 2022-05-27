from pathlib import Path
from setuptools import setup, find_packages

REQUIREMENTS = [
    "torch>=1.10.0",
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nebulgym",
    version="0.1.0",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
