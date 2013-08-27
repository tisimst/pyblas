============================
pyblas package documentation
============================

Introduction
============

``pyblas`` is the pure-python translation of the Fortran BLAS_ library (v3.4.2). 
All semantics are designed to allow a turn-key drop-in replacement. 

The only main difference is that there isn't a separate suite of routines for 
``SINGLE``, ``DOUBLE``, ``COMPLEX``, and ``COMPLEX*16`` type Fortran
objects, just the python equivalent ``float`` and ``complex`` types. However,
all of the BLAS function names still exist (like ``sgeev``, ``dgeev``, 
``cgeev``, ``zgeev``) for drop-in compatibility.

Contact
=======

Any questions, comments, bug reporting, etc can be forwarded to the author:
`Abraham Lee`_


.. _BLAS: http://www.netlib.org/blas/
.. _Abraham Lee: mailto:tisimst@gmail.com

