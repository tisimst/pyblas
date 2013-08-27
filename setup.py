import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='pyblas',
    version='0.1 alpha',
    author='Abraham Lee',
    author_email='tisimst@gmail.com',
    description='Pure-python BLAS translation',
    url='https://github.com/tisimst/pyblas',
    license='BSD License',
    long_description=read('README.rst'),
    packages=[
        'pyblas', 
        'pyblas.FLOAT',
        'pyblas.COMPLEX',
        'pyblas.AUXILIARY'],
    keywords=[
        'blas',
        'linear algebra',
        'lapack',
        'numerical methods',
        'python'
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
        ]
    )
