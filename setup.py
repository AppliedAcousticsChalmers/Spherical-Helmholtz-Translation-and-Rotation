from setuptools import setup, find_packages
import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shetar'))
from _version import hardcoded  # We cannot import the _version module, but we can import from it.

with hardcoded() as version:
    setup(
        name='shetar',
        version=version,
        description='Python implementations of fast multipole methods for the Helmholtz equation',
        long_description=open('README.rst', encoding='UTF-8').read(),
        long_description_content_type='text/x-rst',
        url='https://github.com/AppliedAcousticsChalmers/Spherical-Helmholtz-Translation-and-Rotation',
        author='Carl Andersson',
        author_email='carl.andersson@chalmers.se',
        license='MIT',
        packages=find_packages('.'),
        python_requires='>=3.6',
        install_requires=[
            'numpy',
            'scipy'],
        tests_require=['pytest', 'pytest-cov'],
        setup_requires=['pytest-runner'],
        include_package_data=True,
    )
