from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

with open('requirements.txt') as f:
    packages_required = f.read().splitlines()


ext_modules = [
    Pybind11Extension(
        "_migration",
        ['Migration_on_C/Migration.cpp']
        # sorted(glob('cpp_extensions/Migration_on_C/*.cpp'))
    )
]


setup(
    name='distributed-kirchhoff-migration',
    # version='0.0.1',
    # description='MPI application of kirchhoff migration implementation',
    # author='Vershinin Maxim',
    # author_email='123456vershinin@gmail.com',
    scripts=['mpi_migration.py'],
    install_requires=packages_required,
    ext_modules=ext_modules
)
