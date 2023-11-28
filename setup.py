from setuptools import setup, find_packages

# read the contents of your README file (https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/)
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

setup(
    name='mitransient',
    version='1.0.0',
    description='Transient Mitsuba 3',
    url='https://github.com/diegoroyo/mitsuba3-transient-nlos',
    author='Miguel Crespo, Diego Royo, Jorge García',
    author_email='droyo@unizar.es',
    license='BSD',
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown"
)
