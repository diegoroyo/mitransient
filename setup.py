from setuptools import setup, find_packages

# read the contents of your README file (https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/)
from pathlib import Path
this_directory = Path(__file__).parent
readme = (this_directory / "README.md").read_text()

setup(
    name='mitransient',
    version='0.1.0',
    description='Mitsuba 3 transient',
    url='https://github.com/mcrescas/mitsuba3-transient',
    author='Miguel Crespo, Diego Royo',
    author_email='miguel.crespo@epfl.ch',
    license='BSD',
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown"
)
