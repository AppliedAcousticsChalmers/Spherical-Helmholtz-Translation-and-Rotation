from setuptools import setup, find_packages, Extension
import sys
import os.path
from Cython.Build import cythonize
from Cython.Build.Dependencies import default_create_extension

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shetar'))
from _version import hardcoded  # We cannot import the _version module, but we can import from it.


def find_file_by_extension(path='.', ext='.py'):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files


# import numpy
pyx_files = find_file_by_extension('shetar', '.pyx')
# extensions = [
#     Extension(
#         'shetar.bases._legendre_cython',
#         ['shetar/bases/_legendre_cython.pyx'],
#     )
# ]
# cythonized_extensions = cythonize(extensions, annotate=True, language_level=3)
def custom_create_extension(template, kwds):
    # extra_compile_args = kwds.get('extra_compile_args', [])
    # extra_compile_args.append('-fopenmp')
    # kwds['extra_compile_args'] = extra_compile_args

    # extra_link_args = kwds.get('extra_link_args', [])
    # extra_link_args.append('-fopenmp')
    # kwds['extra_link_args'] = extra_link_args

    return default_create_extension(template, kwds)


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
        ext_modules=cythonize(pyx_files, annotate=True, create_extension=custom_create_extension, language_level=3),
        # ext_modules=cythonized_extensions,
        python_requires='>=3.6',
        install_requires=[
            'numpy',
            'scipy'],
        tests_require=['pytest', 'pytest-cov'],
        setup_requires=['pytest-runner'],
        include_package_data=True,
    )
