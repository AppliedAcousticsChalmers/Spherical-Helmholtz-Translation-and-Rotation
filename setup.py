from setuptools import setup, find_packages, Extension
import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'shetar'))
from _version import hardcoded  # We cannot import the _version module, but we can import from it.


def find_file_by_extension(path='.', ext='.py'):
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(ext):
                files.append(os.path.join(root, filename))
    return files


pyx_files = find_file_by_extension('shetar', '.pyx')
c_files = find_file_by_extension('shetar', '.c')


try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = len(pyx_files) > 0


if use_cython:
    ext_modules = cythonize(pyx_files, annotate=True, language_level=3)
else:
    ext_modules = []
    for filename in c_files:
        ext_modules.append(Extension(os.path.splitext(filename)[0].replace('/', '.'), [filename]))

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
        ext_modules=ext_modules,
        python_requires='>=3.8',
        install_requires=['numpy', 'scipy'],
        tests_require=['pytest', 'pytest-cov'],
        setup_requires=['pytest-runner'],
        include_package_data=True,
    )
