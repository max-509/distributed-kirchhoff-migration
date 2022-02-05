from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

with open('requirements.txt') as f:
    packages_required = f.read().splitlines()


ext_modules = [
    Pybind11Extension(
        "_migration",
        ['Migration_on_C/Migration.cpp']
    )
]


setup(
    name='distributed-kirchhoff-migration',
    version='0.0.1',
    description='MPI application of kirchhoff migration implementation',
    author='Vershinin Maxim, Zobnin Gleb, Dzharkinov Ruslan, Konyukhov Grigoriy',
    author_email='123456vershinin@gmail.com, 6777gleb7776@gmail.com, dzharkinovr@gmail.ru, konyukhovgm@gmail.com',
    scripts=['mpi_migration.py'],
    install_requires=packages_required,
    ext_modules=ext_modules
)
