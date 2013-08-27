"""
================================================================================
pyblas: Pure-python BLAS translation
================================================================================

Translation of all the SINGLE and DOUBLE subroutines of the Fortran BLAS library
(both Sxxxx and Dxxxx). Since python doesn't distinguish between the two numeric
types, the author did not feel it necessary to define separate subroutines, 
but simply alias the SINGLE precision to the DOUBLE precision versions.

Author: Abraham Lee
Copyright: 2013

Reference
---------
BLAS 3.4.2

"""
import copy
import math
from pyblas_auxsub import *

__all__ = []

################################################################################
 
