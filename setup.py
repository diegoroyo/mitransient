from setuptools import setup, find_packages

# read the contents of your README file (https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/)
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

# Read version information without loading all the library
with open('mitransient/version.py') as f:
    exec(f.read())

setup(
    name='mitransient',
    version=__version__,
    description='Transient rendering extensions for Mitsuba 3',
    url='https://github.com/diegoroyo/mitransient',
    author='Diego Royo, Miguel Crespo, Jorge García',
    author_email='droyo@unizar.es',
    license='BSD',
    packages=find_packages(),
    install_requires=[f"mitsuba>={__mi_version_min__},<={__mi_version_max__}"],
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.8"
)
