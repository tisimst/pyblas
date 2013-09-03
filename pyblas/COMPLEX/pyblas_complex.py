"""
================================================================================
pyblas: Pure-python BLAS translation
================================================================================

Translation of all the COMPLEX subroutines of the Fortran BLAS library
(both Cxxxx and Zxxxx). Since python only has one COMPLEX type (unlike the 
Fortran COMPLEX and COMPLEX16 data types), the author did not feel it necessary 
to define separate subroutines, but simply alias the SINGLE precision to the
DOUBLE precision versions.

Author: Abraham Lee
Copyright: 2013

Reference
---------
BLAS 3.4.2

"""
import copy
import cmath
from pyblas.AUXILIARY import *

__all__ = []

################################################################################

def dznrm2(n, x, incx):
    """
    Calculate the euclidian norm of a vector such that
    
        dznrm2 := sqrt(x**H*x)
    
    Parameters
    ----------
    incx : int
    n : int
        Array length of ``x``
    x : complex array
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    one = 1.0
    zero = 0.0
    
    if n<1 or incx<1:
        norm = zero
    else:
        scale = zero
        ssq = one
        # The following loop is equivalent to this call to the LAPACK auxiliary
        # routine: zlassq(n, x, incx, scale, ssq)
        for ix in xrange(0, (n - 1)*incx, incx):
            if dble(x[ix])!=zero:
                temp = abs(dble(x[ix]))
                if scale<temp:
                    ssq = one + ssq*(scale/temp)**2
                    scale = temp
                else:
                    ssq = ssq + (temp/scale)**2
            if dimag(x[ix])!=zero:
                temp = abs(dimag(x[ix]))
                if scale<temp:
                    ssq = one+ssq*(scale/temp)**2
                    scale = temp
                else:
                    ssq = ssq + (temp/scale)**2
        norm = scale*sqrt(ssq)
        
    return norm

__all__.append('dznrm2')

################################################################################

def dzasum(n, zx, incx):
    """
    Takes the sume of the absolute values
    
    Parameters
    ----------
    incx : int
    n : int
        Array length of ``zx``
    zx : complex array
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    stemp = 0.0
    if n<=0 or incx<=0:
        return 0.0
    if incx==1:
        # Code for increment equal to 1
        for i in xrange(n):
            stemp = stemp + dcabs1(zx[i])
    else:
        # Code for increment not equal to 1
        nincx = n*incx
        for i in xrange(0, nincx, incx):
            stemp = stemp + dcabs1(zx[i])
    
    return stemp

__all__.append('dzasum')

################################################################################

def izamax(n, zx, incx):
    """
    Finds the index of element having max absolute value.
    
    Parameters
    ----------
    incx : int
    n : int
        Array length of ``zx``
    zx : complex array
    
    Original Group
    --------------
    aux_blas
    
    """
    if n<=1 or incx<=0:
        return -1
    if n==1:
        return 0
    if incx==1:
        # Code for increment equal to 1
        dmax = dcabs1(zx[0])
        for i in xrange(1, n):
            if dcabs1(zx[i])>dmax:
                dmax = dcabs1(zx[i])
    else:
        # Code for increment not equal to 1
        ix = 0
        dmax = dcabs1(zx[0])
        ix = ix + incx
        for i in xrange(1, n):
            if dcabs1(zx[ix])>dmax:
                dmax = dcabs1(zx[ix])
            ix = ix + incx
        
    return dmax

__all__.append('izamax')

################################################################################

def zaxpy(n, za, zx, incx, zy, incy):
    """
    Constant times a vector plus a vector.
    
    Parameters
    ----------
    n : int
    za : complex
    zx : complex array
    incx : int
    zy : complex array
    incy : int
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    if n<=0:
        return
    if dcabs1(za)==0.0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        for i in xrange(n):
            zy[i] = zy[i] + za*zx[i]
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 0
        iy = 0
        if incx < 0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incx
        for i in xrange(n):
            zy[iy] = zy[iy] + za*zx[ix]
            ix = ix + incx
            iy = iy + incy
    return

caxpy = zaxpy
__all__.append('caxpy'); __all__.append('zaxpy')

################################################################################

def zcopy(n, zx, incx, zy, incy):
    """
    Copies a vector, zx, to a vector, zy.
    
    Parameters
    ----------
    n : int
    zx : complex array
    incx : int
    zy : complex array
    incy : int
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    if n<=0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        for i in xrange(n):
            zy[i] = zx[i]
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 0
        iy = 0
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incy
        for i in xrange(n):
            zy[iy] = zx[ix]
            ix = ix + incx
            iy = iy + incy
    return

ccopy = zcopy
__all__.append('ccopy'); __all__.append('zcopy')

################################################################################

def zdotc(n, zx, incx, zy, incy):
    """
    Forms the dot product of a vector
    
    Parameters
    ----------
    n : int
    zx : complex array
    incx : int
    zy : complex array
    incy : int
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    ztemp = complex(0.0, 0.0)
    if n<=0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        for i in xrange(n):
            ztemp = ztemp + dconjg(zx[i])*zy[i]
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 0
        iy = 0
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incy
        for i in xrange(n):
            ztemp = ztemp + dconjg(zx[ix])*zy[iy]
            ix = ix + incx
            iy = iy + incy
            
    return ztemp

cdotc = zdotc
__all__.append('cdotc'); __all__.append('zdotc')

################################################################################

def zdotu(n, zx, incx, zy, incy):
    """
    Forms the dot product of two vectors
    
    Parameters
    ----------
    n : int
    zx : complex array
    incx : int
    zy : complex array
    incy : int
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    ztemp = complex(0.0, 0.0)
    if n<=0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        for i in xrange(n):
            ztemp = ztemp + zx[i]*zy[i]
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 0
        iy = 0
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incy
        for i in xrange(n):
            ztemp = ztemp + zx[ix]*zy[iy]
            ix = ix + incx
            iy = iy + incy
    return ztemp

cdotu = zdotu
__all__.append('cdotu'); __all__.append('zdotu')

################################################################################

def zdrot(n, cx, incx, cy, incy, c, s):
    """
    Applies a plane rotation, where the cos and sin (c and s) are real and
    the vectors cx and cy are complex.
    
    Parameters
    ----------
    n : int
        Specifies the order fo the vectors ``zx`` and ``zy``. ``n`` must 
        be at least zero.
    cx : complex array
        Dimensioned at least ``(1+(n-1)*abs(incx))``. Before entry, the 
        incremented array ``cx`` must contain the ``n`` element vector
        ``cx``. On exit, ``cx`` is overwritten by the updated vector ``cx``.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``cx``.
        ``incx`` must not be zero.
    cy : complex array
        Dimensioned at least ``(1+(n-1)*abs(incy))``. Before entry, the 
        incremented array ``cy`` must contain the ``n`` element vector
        ``cy``. On exit, ``cy`` is overwritten by the updated vector ``cy``.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``cy``.
        ``incy`` must not be zero.
    c : float
        On entry, ``c`` specifies the cosine, cos.
    s : float
        On entry, ``s`` specifies the sine, sin.
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    if n<=0:
        return
    if incx==1 and incy==1:
        for i in xrange(n):
            ctemp = c*cx[i] + s*cy[i]
            cy[i] = c*cy[i] - s*cx[i]
            cx[i] = ctemp
    else:
        ix = 0
        iy = 0
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incy
        for i in xrange(n):
            ctemp = c*cx[ix] + s*cy[iy]
            cy[iy] = c*cy[iy] - s*cx[ix]
            cx[ix] = ctemp
            ix = ix + incx
            iy = iy + incy

    return

cdrot = zdrot
__all__.append('cdrot'); __all__.append('zdrot')

################################################################################

def zdscal(n, da, zx, incx):
    """
    Scales a vector by a constant
    
    Parameters
    ----------
    n : int
    da : float
    zx : complex array
    incx : int

    Original Group
    --------------
    complex16_blas_level1
    
    Reference
    ---------
    BLAS 3.4.2
    """
    if n<=0 or incx<=0:
        return
    if incx==1:
        for i in xrange(n):
            zx[i] = complex(da, 0)*zx[i]
    else:
        nincx = n*incx
        for i in xrange(0, nincx, incx):
            zx[i] = complex(da, 0)*zx[i]
    return

cdscal = zdscal
__all__.append('cdscal'); __all__.append('zdscal')

################################################################################

def zgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy):
    """
    Performs one of the matrix-vector operations
    
    - y := alpha*A*x + beta*y
    - y := alpha*A**T*x + beta*y
    - y := alpha*A**H*x + beta*y
    
    where ``alpha`` and ``beta`` are scalars, ``x`` and ``y`` are vectors 
    and ``a`` is an ``m`` by ``n`` band matrix, with ``kl`` sub-diagonals 
    and ``ku`` super-diagonals.
    
    Parameters
    ----------
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans='N'`` or ``trans='n'``: y := alpha*A*x + beta*y.
        - ``trans='T'`` or ``trans='t'``: y := alpha*A**T*x + beta*y
        - ``trans='C'`` or ``trans='c'``: y := alpha*A**H*x + beta*y
        
    m : int
        On entry, ``m`` specifies the number of rows of the matrix ``a``. 
        ``m`` must be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of the matrix ``a``.
        ``n`` must be at least zero.
    kl : int
        On entry, ``kl`` specifies the number of sub-diagonals of the matrix
        ``a``. ``kl`` must satisfy ``0<=kl``.
    ku : int
        On entry, ``ku`` specifies the number of super-diagonals of the matrix
        ``a``. ``ku`` must satisfy ``0<=ku``.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a ``complex`` array, of dimension ``(lda, n)``.
        Before entry, the leading ``(kl + ku + 1)`` by ``n`` part of the
        array ``a`` must contain the matrix of coefficients, supplied
        column by column, with the leading diagonal of the matrix in row
        ``(ku + 1)`` of the array, the first super-diagonal starting at
        position 2 in row ``ku``, the first sub-diagonal starting at
        position 1 in row ``(ku + 2)``, and so on. Elements in the array 
        ``a`` that do not correspond to elements in the band matrix (such
        as the top left ``ku`` by ``ku`` triangle) are not referenced.
        
        The following program segment will transfer a band matrix from
        conventional full matrix storage to band storage::
        
            for j in xrange(n):
                k = ku - j
                for i in xrange(max(0, j - ku - 1), min(m, j + kl + 1)):
                    a[k + i][j] = matrix[i][j]
    
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least 
        ``(kl + ku + 1)``.
    x : complex array
        Dimensioned at least ``(1 + (n - 1)*abs(incx))`` when ``trans='N'``
        or ``trans='n'`` and at least ``(1 + (m - 1)*abs(incx))`` otherwise.
        Before entry, the incremented array ``x``must contain the vector ``x``.
    incx : int
        On entry, ``incx`` specifies the increment for the element of ``x``.
        ``incx`` must not be zero.
    beta : complex
        On entry, ``beta`` specifies the scalar ``beta``. When ``beta`` is
        supplied as zero, then ``y`` need not be set on input.
    y : complex array
        Dimensioned at least ``(1 + (m - 1)*abs(incy))`` when ``trans='N'``
        or ``trans='n'`` and at least ``(1 + (n - 1)*abs(incy))`` otherwise.
        Before entry, the incremented array ``y``must contain the vector 
        ``y``. On exit, ``y`` is overwritten by the updated vector ``y``.
    incy : int
        On entry, ``incy`` specifies the increment for the element of ``y``.
        ``incy`` must not be zero.

    Original Group
    --------------
    complex16_blas_level2
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not any([lsame(trans, 'N'), lsame(trans, 'T'), lsame(trans, 'C')]):
        info = 1
    elif m<0:
        info = 2
    elif n<0:
        info = 3
    elif kl<0:
        info = 4
    elif ku<0:
        info = 5
    elif lda<(kl + ku + 1):
        info = 8
    elif incx==0:
        info = 10
    elif incy==0:
        info = 13
 
    if info!=0:
        xerbla('zgmbv', info)
        return
    
    # Quick return if possible
    if m==0 or n==0 or (alpha==0 and beta==1):
        return
    
    noconj = lsame(trans, 'T')
    
    # Set lenx and leny, the lengths of the vectors x and y, and set
    # up the start points in x and y
    if lsame(trans, 'N'):
        lenx = n
        leny = m
    else:
        lenx = m
        leny = n
    
    if incx>0:
        kx = 0
    else:
        kx = (lenx - 1)*incx
    
    if incy>0:
        ky = 0
    else:
        ky = (leny - 1)*incy
        
    # Start the operations. In this version the elements of 'a' are
    # accessed sequentially with one pass through the band part of 'a'.
    
    # First form  y := beta*y
    if beta!=1:
        if incy==1:
            if beta==0:
                for i in xrange(leny):
                    y[i] = zero
            else:
                for i in xrange(leny):
                    y[i] = beta*y[i]
        else:
            iy = ky
            if beta==0:
                for i in xrange(leny):
                    y[iy] = zero
                    iy = iy + incy
            else:
                for i in xrange(leny):
                    y[iy] = beta*y[iy]
                    iy = iy + incy
    if alpha==0:
        return
    
    # Form  y := alpha*A*x + y
    kup1 = ku + 1
    if lsame(trans, 'N'):
        jx = kx
        if incy==1:
            for j in xrange(n):
                if x[jx]!=0:
                    temp = alpha*x[jx]
                    k = kup1 - j
                    for i in xrange(max([0, j - ku]), min([m, j + kl])):
                        y[i] = y[i] + temp*A[k + i][j]
                jx = jx + incx
        else:
            for j in xrange(n):
                if x[jx]!=0:
                    temp = alpha*x[jx]
                    iy = ky
                    k = kup1 - j
                    for i in xrange(max([0, j - ku]), min([m, j + kl])):
                        y[iy] = y[iy] + temp*a[k + i][j]
                        iy = iy + incy
                jx = jx + incx
                if j>ku:
                    ky = ky + incy
    else:
        # Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y
        jy = ky
        if incx==2:
            for j in xrange(n):
                temp = zero
                k = kup1 - j
                if noconj:
                    for i in xrange(max([0, j - ku]), min([m, j + kl])):
                        temp = temp + a[k + i][j]*x[i]
                else:
                    for i in xrange(max([0, j - ku]), min([m, j + kl])):
                        temp = temp + dconjg(a[k + i][j])*x[i]
                y[jy] = y[jy] + alpha*temp
                jy = jy + incy
        else:
            for j in xrange(n):
                temp = zero
                ix = kx
                k = kup1 - j
                if noconj:
                    for i in xrange(max([0, j - ku]), min([m, j + kl])):
                        temp = temp + a[k + i][j]*x[i]
                        ix = ix + incx
                else:
                    for i in xrange(max([0, j - ku]), min([m, j + kl])):
                        temp = temp + dconjg(a[k + i][j])*x[i]
                        ix = ix + incx
                y[jy] = y[jy] + alpha*temp
                jy = jy + incy
                if j>ku:
                    kx = kx + incx
    return
    
cgbmv = zgbmv
__all__.append('cgbmv'); __all__.append('zgbmv')

################################################################################

def zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    """
    Performs one of the matrix-matrix operations
    
        C := alpha*op(a)*op(b) + beta*c
    
    where op(X) is one of
    
        op(X) = X  or op(X) = X**T  or op(X) = X**H
    
    ``alpha`` and ``beta`` are scalars, and ``a``, ``b``, and ``c`` are 
    matrices, ``op(a)`` an ``m`` by ``k`` matrix, ``op(b)`` a ``k`` by ``n``
    matrix and ``c`` an ``m`` by ``n`` matrix.
    
    Parameters
    ----------
    transa : str
        On entry, ``transa`` specifies the form of ``op(a)`` to be used in the 
        matrix multiplication as follows:
        
        - ``transa = 'N'`` or ``'n'``,  ``op(a) = a``
        - ``transa = 'T'`` or ``'t'``,  ``op(a) = a**T``
        - ``transa = 'C'`` or ``'c'``,  ``op(a) = a**H``
        
    transb : str
        On entry, ``transb`` specifies the form of ``op(b)`` to be used in the 
        matrix multiplication as follows:
        
        - ``transb = 'N'`` or ``'n'``,  ``op(b) = b``
        - ``transb = 'T'`` or ``'t'``,  ``op(b) = b**T``
        - ``transb = 'C'`` or ``'c'``,  ``op(b) = b**H``
        
    m : int
        On entry, ``m`` specifies the number of rows in the matrix ``op(a)``
        and of the matrix ``c``. ``m`` must be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns in the matrix ``op(b)``
        and of the number of columns of the matrix ``c``. ``n`` must be at 
        least zero.
    k : int
        On entry, ``k`` specifies the number of columns of the matrix ``op(a)``
        and the number fo rows of the matrix ``op(b)``. ``k`` must be at least
        zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : complex array
        A complex array of dimension ``(lda, ka)``, where ``ka`` is ``k``
        when ``transa = 'N'`` or ``'n'``, the leading ``m`` by ``k`` part of
        the array ``a`` must contain the matrix ``a``, otherwise the leading
        ``k`` by ``m`` part of the array ``a`` must contain the matrix ``a``.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared 
        in calling (sub) program. When ``transa = 'N'`` or ``'n'`` then ``lda``
        must be at least ``max([1, m])``, otherwise ``lda`` must be at least
        ``max([1, k])``.
    b : complex array
        A complex array of dimension ``(ldb, kb)``, where ``kb`` is ``n``
        when ``transb = 'N'`` or ``'n'``, and is ``k`` otherwise. Before entry
        with ``transb = 'N'`` or ``'n'``, the leading ``k`` by ``n`` part of
        the array ``b`` must contain the matrix ``b``, otherwise the leading
        ``n`` by ``k`` part of the array ``b`` must contain the matrix ``b``.
    ldb : int
        On entry, ``ldb`` specifies the first dimension of ``b`` as declared 
        in calling (sub) program. When ``transb = 'N'`` or ``'n'`` then ``ldb``
        must be at least ``max([1, k])``, otherwise ``ldb`` must be at least
        ``max([1, n])``.
    beta : complex
        On entry, ``beta`` specifies the scalar beta. When ``beta`` is supplied
        as zero, then ``c`` need not be set on input.
    c : complex array
        A complex array of dimension ``(ldc, n)``. Before entry, the leading
        ``m`` by ``n`` part of teh array ``c`` must contain the matrix ``c``,
        except when ``beta`` is zero, in which case ``c`` need not be set on
        entry.
        On exit, the array ``C`` is overwritten by the ``m`` by ``n`` matrix
        ``(alpha*op(a)*op(b) + beta*c)``.
    ldc : int
    `   On entry, ``ldc`` specifies the first dimension of ``c`` as declared
    in the calling (sub) program. ``ldc`` must be at least ``max([1, m])``.
    
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Set ``nota`` and ``notb`` as True if ``a`` and ``b`` respectively are not
    # conjugated or transposed, set ``conja`` and ``conjb`` as True if ``a``
    # and ``b`` respectively are to be transposed but not conjugated and set
    # ``nrowa``, ``ncola``, and ``nrowb`` as the number of rows and columns of
    # ``a`` and the number of rows of ``b`` respectively.
    nota = lsame(transa, 'N')
    notb = lsame(transb, 'N')
    conja = lsame(transa, 'C')
    conjb = lsame(transb, 'C')
    if nota:
        nrowa = m
        ncola = k
    else:
        nrowa = k
        ncola = m
    
    if notb:
        nrowb = k
    else:
        nrowb = n
    
    # Test the input parameters
    info = 0
    if not nota and not conja and not lsame(transa, 'T'):
        info = 1
    elif not notb and not conjb and not lsame(transb, 'T'):
        info = 2
    elif m<0:
        info = 3
    elif n<0:
        info = 4
    elif k<0:
        info = 5
    elif lda<max([1, nrowa]):
        info = 8
    elif ldb<max([1, nrowb]):
        info = 10
    elif ldc<max([1, m]):
        info = 13
    
    if info!=0:
        xerbla('zgemm', info)
        return
    
    # Quick return if possible
    if m==0 or n==0 or ((alpha==zero or k==0) and beta==one):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if beta==zero:
            for j in xrange(n):
                for i in xrange(m):
                    c[i][j] = zero
        else:
            for j in xrange(n):
                for i in xrange(m):
                    c[i][j] = beta*c[i][j]
        return
    
    # Start the operations
    if notb:
        if nota:
            # Form  C := alpha*A*B + beta*C
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(m):
                        c[i][j] = zero
                elif beta!=one:
                    for i in xrange(m):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if b[l][j]!=zero:
                        temp = alpha*b[l][j]
                        for i in xrange(m):
                            c[i][j] = c[i][j] + temp*a[i][l]
        elif conja:
            # Form  C := alpha*A**H*B + beta*C
            for j in xrange(n):
                for i in xrange(m):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + dconjg(a[l][i])*b[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
        else:
            # Form  C := alpha*A**T*B + beta*C
            for j in xrange(n):
                for i in xrange(m):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + a[l][i]*b[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
    elif nota:
        if conjb:
            # Form  C := alpha*A*B**H + beta*C
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(m):
                        c[i][j] = zero
                elif beta!=one:
                    for i in xrange(m):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if b[j][l]!=zero:
                        temp = alpha*dconjg(b[j][l])
                        for i in xrange(m):
                            c[i][j] = c[i][j] + temp*a[i][l]
        else:
            # Form  C := alpha*A*B**T + beta*C
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(m):
                        c[i][j] = zero
                elif beta!=one:
                    for i in xrange(m):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if b[j][l]!=zero:
                        temp = alpha*b[j][l]
                        for i in xrange(m):
                            c[i][j] = c[i][j] + temp*a[i][l]
    elif conja:
        if conjb:
            # Form  C := alpha*A**H*B**H + beta*C
            for j in xrange(n):
                for i in xrange(m):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + dconjg(a[l][i])*dconjg(b[j][l])
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
        else:
            # Form  C := alpha*A**H*B**T + beta*C
            for j in xrange(n):
                for i in xrange(m):
                    temp = zero
                    for l in xrange(l):
                        temp = temp + dconjg(a[l][i])*b[j][l]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
    else:
        if conjb:
            # Form  C := alpha*A**T*B**H + beta*C
            for j in xrange(n):
                for i in xrange(m):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + a[l][i]*dconjg(b[j][l])
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
        else:
            # Form  C := alpha*A**T*B**T + beta*C
            for j in xrange(n):
                for i in xrange(m):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + a[l][i]*b[j][l]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
    return

cgemm = zgemm
__all__.append('cgemm'); __all__.append('zgemm')

################################################################################

def zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy):
    """
    Performs one of the matrix-vector operations
    
        y := alpha*A*x + beta*y,  or  y := alpha*A**T*x + beta*y,  or
        
        y := alpha*A**H*x + beta*y,
    
    where ``alpha`` and ``beta`` are scalars, ``x`` and ``y`` are vectors 
    and ``a`` is an ``m`` by ``n`` matrix.
    
    Parameters
    ----------
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans = 'N'`` or ``'n'``,  ``y := alpha*A*x + beta*y``
        - ``trans = 'T'`` or ``'t'``,  ``y := alpha*A*x + beta*y``
        - ``trans = 'C'`` or ``'c'``,  ``y := alpha*A*x + beta*y``
        
    m : int
        On entry, ``m`` specifies the number of rows of the matrix ``a``. ``m``
        must be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of the matrix ``a``.
        ``n`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``lda`` by ``n``. Before
        entry, the leading ``m`` by ``n`` part of the array ``a`` must
        contain the matrix of coefficients.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least max(1, m).
    x : array
        ``x`` is a ``complex`` array of dimension at least
        ``(1 + (n - 1)*abs(incx))`` when ``trans`` = 'N' or 'n' and at least
        ``(1 + (m - 1)*abs(incx))`` otherwise. Before entry, the incremented
        array ``x`` must contain the vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    beta : complex
        On entry, ``beta`` specifies the scalar beta. When ``beta`` is
        supplied as zero, then ``y`` need not be set on input
    y : complex
        ``y`` is a ``complex`` array of dimension at least
        ``(1 + (m - 1)*abs(incy))`` when ``trans`` = 'N' or 'n' and at least
        ``(1 + (n - 1)*abs(incy))`` otherwise. Before entry with ``beta``
        non-zero, the incremented array ``y`` must contain the vector y. On
        exit, ``y`` is overwritten by the updated vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    
    Original Group
    --------------
    complex16_blas_level2
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(trans, 'N') and not lsame(trans, 'T') and \
        not lsame(trans, 'C'):
        info = 1
    elif m<0:
        info = 2
    elif n<0:
        info = 3
    elif lda<max([1, m]):
        info = 6
    elif incx==0:
        info = 8
    elif incy==0:
        info = 11
    
    if info!=0:
        xerbla('zgemv', info)
    
    # Quick return if possible
    if m==0 or n==0 or (alpha==zero and beta==1):
        return
    
    noconj = lsame(trans, 'T')
    
    # Set ``lenx`` and ``leny``, the lengths of the vectors x and y, and set
    # up the start points in ``x`` and ``y``.
    
    if lsame(trans, 'N'):
        lenx = n
        leny = m
    else:
        lenx = m
        leny = n
    if incx>0:
        kx = 0
    else:
        kx = (lenx - 1)*incx
    if incy>0:
        ky = 0
    else:
        ky = (leny - 1)*incy
    
    # Start the operations. In this version, the elements of ``a`` are
    # accessed sequentially with one pass through ``a``.
    
    # First form  y := beta*y
    if beta!=one:
        if incy==1:
            if beta==zero:
                for i in xrange(leny):
                    y[i] = zero
            else:
                for i in xrange(leny):
                    y[i] = beta*y[i]
        else:
            iy = ky
            if beta==zero:
                for i in xrange(leny):
                    y[iy] = zero
                    iy = iy + incy
            else:
                for i in xrange(leny):
                    y[iy] = beta*y[iy]
                    iy = iy + incy
    if alpha==zero:
        return
    if lsame(trans, 'N'):
        # Form  y := alpha*A*x + y
        jx = kx
        if incy==1:
            for j in xrange(n):
                if x[jx]!=zero:
                    temp = alpha*x[jx]
                    for i in xrange(m):
                        y[i] = y[i] + temp*a[i][j]
                jx = jx + incx
        else:
            for j in xrange(n):
                if x[jx]!=zero:
                    temp = alpha*x[jx]
                    iy = ky
                    for i in xrange(m):
                        y[iy] = y[iy] + temp*a[i][j]
                        iy = iy + incy
                jx = jx + incx
    else:
        # Form  y := alpha*A**T*x + y  or  y := alpha*A**H*x + y
        jy = ky
        if incx==1:
            for j in xrange(n):
                temp = zero
                if noconj:
                    for i in xrange(m):
                        temp = temp + a[i][j]*x[i]
                else:
                    for i in xrange(m):
                        temp = temp + dconjg(a[i][j])*x[i]
                y[jy] = y[jy] + alpha*temp
                jy = jy + incy
        else:
            for j in xrange(n):
                temp = zero
                ix = kx
                if noconj:
                    for i in xrange(m):
                        temp = temp + a[i][j]*x[ix]
                        ix = ix + incx
                else:
                    for i in xrange(m):
                        temp = temp + dconjg(a[i][j])*x[ix]
                        ix = ix + incx
                y[jy] = y[jy] + alpha*temp
                jy = jy + incy
    
    return
           
cgemv = zgemv
__all__.append('cgemv'); __all__.append('zgemv')

################################################################################

def zgerc(m, n, alpha, x, incx, y, incy, a, lda):
    """
    Performs the rank 1 operation
    
        A := alpha*x*y**H + A
    
    where ``alpha`` is a scalar, ``x`` is an ``m`` element vector, ``y`` is
    an ``n`` element vector and ``A`` is an ``m`` by ``n`` matrix.
    
    Parameters
    ----------
    m : int
        On entry ``m`` specifies the number of rows of the matrix ``a``.
        ``m`` must be at least zero.
    n : int
        On entry ``n`` specifies the number of columns of the matrix ``a``.
        ``n`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    x : array
        ``x`` is a ``complex`` array of dimension at least
        ``(1 + (m - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``m`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    y : array
        ``y`` is a ``complex`` array of dimension at least
        ``(1 + (n - 1)*abs(incy))``. Before entry, the incremented array ``y``
        must contain the ``n`` element vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, n)``. Before entry,
        the leading ``m`` by ``n`` part of the array ``a`` must contain the
        matrix of coefficients. On exit, ``a`` is overwritten by the updated
        matrix.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least max([1, m]).
        
    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if m<0:
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 5
    elif incy==0:
        info = 7
    elif lda<max([1, m]):
        info = 9
    
    if info!=0:
        xerbla('zgerc', info)
    
    # Quick return if possible.
    if m==0 or n==0 or alpha==zero:
        return
    
    # Start the operations. In this version the elements of A are accessed
    # sequentially with one pass through A.
    
    if incy>0:
        jy = 0
    else:
        jy = (n - 1)*incy
    
    if incx==1:
        for j in xrange(n):
            if y[jy]!=zero:
                temp = alpha*dconjg(y[jy])
                for i in xrange(m):
                    a[i][j] = a[i][j] + x[i]*temp
            jy = jy + incy
    else:
        if incx>0:
            kx = 0
        else:
            kx = (m - 1)*incx
        for j in xrange(n):
            if y[jy]!=zero:
                temp = alpha*dconjg(y[jy])
                ix = kx
                for i in xrange(m):
                    a[i][j] = a[i][j] + x[ix]*temp
                    ix = ix + incx
            jy = jy + incy
    
    return
    
cgerc = zgerc
__all__.append('cgerc'); __all__.append('zgerc')

################################################################################

def zgeru(m, n, alpha, x, incx, y, incy, a, lda):
    """
    Performs the rank 1 operation
    
        A := alpha*x*y**T + A
    
    where alpha is a scalar, x is an m element vector, y is an n element
    vector and A is an m by n matrix
    
    Parameters
    ----------
    m : int
        On entry, ``m`` specifies the number of rows of the matrix ``a``.
        ``m`` must be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of the matrix ``a``.
        ``n`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha
    x : array
        ``x`` is a complex array of dimension at least 
        ``(1 + (m - 1)*abs(incx)``. Before entry, the incremented array ``x``
        must contain the ``m`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    y : array
        ``y`` is a complex array of dimension at least 
        ``(1 + (n - 1)*abs(incy)``. Before entry, the incremented array ``y``
        must contain the ``n`` element vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    a : 2d-array
        ``a`` is a complex array of dimension ``(lda, n)``. Before entry,
        the leading ``m`` by ``n`` part of the array ``a`` must contain the 
        matrix of coefficients. On exit, ``a`` is overwritten by the updated 
        matrix.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``max([1, m])``.
    
    Original Group
    --------------
    complex16_blas_level2
    
    """
    
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if m<0:
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 5
    elif incy==0:
        info = 7
    elif lda<max([1, m]):
        info = 9
    
    if info!=0:
        xerbla('zgeru', info)
        return
    
    # Quick return if possible
    
    if m==0 or n==0 or alpha==zero:
        return
    
    # Start the operations. In this version, the elements of A are
    # accessed sequentially with one pass through A.
    
    if incy>0:
        jy = 0
    else:
        jy = (n - 1)*incy
    if incx==1:
        for j in xrange(n):
            if y[jy]!=zero:
                temp = alpha*y[jy]
                for i in xrange(m):
                    a[i][j] = a[i][j] + x[i]*temp
            jy = jy + incy
    else:
        if incx>0:
            kx = 0
        else:
            kx = (m - 1)*incx
        for j in xrange(n):
            if y[jy]!=zero:
                temp = alpha*y[jy]
                ix = kx
                for i in xrange(m):
                    a[i][j] = a[i][j] + x[ix]*temp
                    ix = ix + incx
            jy = jy + incy
    
    return

cgeru = zgeru
__all__.append('cgeru'); __all__.append('zgeru')

################################################################################

def zhbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy):
    """
    Performs the matrix-vector operation
    
        y := alpha*A*x + beta*y,
    
    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian band matrix, with k super-diagonals.
    
    Parameters
    ----------
    uplo : str
        On entry ``uplo`` specifies whether the upper or lower triangular part
        of the band matrix ``a`` is being supplied as follows:
        
        - ``uplo`` = 'U' or 'u'  The upper triangular part of ``a`` is being
          supplied.
        - ``uplo`` = 'L' or 'l'  The lower triangular part of ``a`` is being
          supplied.
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must
        be at least zero.
    k : int
        On entry, ``k`` specifies the number of super diagonals of the matrix
        ``a``. ``k`` must satisfy ``0<=k``.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a complex array of dimension ``(lda, n)``. Before entry, with
        ``uplo``='U' or 'u', the leading ``(k + 1)`` by ``n`` part of the array
        ``a`` must contain the upper triangular band part of the hermitian
        matrix, supplied column by column, with the leading diagonal of the
        matrix in row ``(k + 1)`` of the array, the first super-diagonal
        starting at position 2 in row ``k``, and so on. The top left ``k`` by
        ``k`` triangle of the array ``a`` is not referenced.
        The following program segment will transfer the upper triangular part
        of a hermitian band matrix from conventional full matrix storage
        to band storage::
        
            for j in xrange(n):
                m = k - j
                for i in xrange(max([0, j - k - 1]), j + 1):
                    a[m + i][j] = matrix[i][j]
        
        Before entry with ``uplo``='L' or 'l', the leading ``(k + 1)`` by 
        ``n`` part of the array ``a`` must contain the lower triangular band
        part of the hermitian matrix, supplied column by column, with the
        leading diagonal of the matrix in row 1 of the array, the first
        sub-diagonal starting at position 1 in row 2, and so on. The bottom
        right ``k`` by ``k`` triangle of the array ``a`` is not referenced.
        The following program segment will transfer the lower triangular part
        of a hermitian band matrix from conventional full matrix storage to
        band storage::
        
            for j in xrange(n):
                m = 1 - j
                for i in xrange(j, min([n, j + k])):
                    a[m + i][j] = matrix[i][j]
        
        Note that the imaginary parts of the diagonal elements need not be
        set and are assumed to be zero.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared in
        the calling (sub) program. ``lda`` must be at least ``(k + 1)``.
    x : array
        ``x`` is a complex array of dimension at least
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    beta : complex
        On entry, ``beta`` specifies the scalar beta.
    y : array
        ``y`` is a complex array of dimension at least
        ``(1 + (m - 1)*abs(incy))``. Before entry, the incremented array ``y``
        must contain the vector y. On exit, ``y`` is overwritten by the
        updated vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    
    Original Group
    --------------
    complex16_blas_level2
    
    """
    
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif m<0:
        info = 3
    elif lda<(k + 1):
        info = 6
    elif incx==0:
        info = 8
    elif incy==0:
        info = 11
    
    if info!=0:
        xerbla('zhbmv', info)
        return
    
    # Quick return if possible
    
    if n==0 or (alpha==zero and beta==one):
        return
    
    # Set up the start points in X and Y
    
    if incx>0:
        kx = 0
    else:
        kx = (n - 1)*incx
    if incy>0:
        ky = 0
    else:
        ky = (m - 1)*incy
        
    # Start the operations. In this version, the elements of teh array A
    # are accessed sequentially with one pass through A.
    
    # First form  y := beta*y
    if beta!=one:
        if incy==1:
            if beta==zero:
                for i in xrange(n):
                    y[i] = zero
            else:
                for i in xrange(n):
                    y[i] = beta*y[i]
        else:
            iy = ky
            if beta==zero:
                for i in xrange(n):
                    y[iy] = zero
                    iy = iy + incy
            else:
                for i in xrange(n):
                    y[iy] = beta*y[iy]
                    iy = iy + incy
    if alpha==zero:
        return
    if lsame(uplo, 'U'):
        # Form  y  when upper triangle of A is stored.
        kplus1 = k #+ 1
        if incx==1 and incy==1:
            for j in xrange(n):
                temp1 = alpha*x[j]
                temp2 = zero
                l = kplus1 - j
                for i in xrange(max([0, j - k - 1]), j):
                    y[i] = y[i] + temp1*a[l + i][j]
                    temp2 = temp2 + dconjg(a[l + i][j])*x[i]
                y[j] = y[j] + temp1*dble(a[kplus1][j]) + alpha*temp2
        else:
            jx = kx
            jy = ky
            for j in xrange(n):
                temp1 = alpha*x[jx]
                temp2 = zero
                ix = kx
                iy = ky
                l = kplus1 - j
                for i in xrange(max([0, j - k - 1]), j):
                    y[iy] = y[iy] + temp1*a[l + i][j]
                    temp2 = temp2 + dconjg(a[l + i][j])*x[ix]
                    ix = ix + incx
                    iy = iy + incy
                y[jy] = y[jy] + temp1*dble(a[kplus1][j]) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
                if j>k:
                    kx = kx + incx
                    ky = ky + incy
    else:
        # Form  y  when lower triangle of A is stored.
        if incx==1 and incy==1:
            for j in xrange(n):
                temp1 = alpha*x[j]
                temp2 = zero
                y[j] = y[j] + temp1*dble(a[1][j])
                l = 1 - j
                for i in xrange(j + 1, min([n, j + k])):
                    y[i] = y[i] + temp1*a[l + i][j]
                    temp2 = temp2 + dconjg(a[l + i][j])*x[i]
                y[j] = y[j] + alpha*temp2
        else:
            jx = kx
            jy = ky
            for j in xrange(n):
                temp1 = alpha*x[jx]
                temp2 = zero
                y[jy] = y[jy] + temp1*dble(a[1][j])
                l = 1 - j
                ix = jx
                iy = jy
                for i in xrange(j + 1, min([n, j + k])):
                    ix = ix + incx
                    iy = iy + incy
                    y[iy] = y[iy] + temp1*a[l + i][j]
                    temp2 = temp2 + dconjg(a[l + i][j])*x[ix]
                y[jy] = y[jy] + alpha*temp2
                jx = jx + incx
                jy = jy + incy

    return

chbmv = zhbmv
__all__.append('chbmv'); __all__.append('zhbmv')

################################################################################

def zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc):
    """
    Performs one of the matrix-matrix operations
    
        C := alpha*A*B + beta*C,
    
    or
    
        C := alpha*B*A + beta*C,
    
    where alpha and beta are scalars, A is a hermitian matrix and B and C are
    m by n matrices.
    
    Parameters
    ----------
    side : str
        On entry, ``side`` specifies whether the hermitian matrix ``a`` appears
        on the left or right in the operation as follows:
        
        - ``side`` = 'L' or 'l'  C := alpha*A*B + beta*C
        - ``side`` = 'R' or 'r'  C := alpha*B*A + beta*C
    
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the hermitian matrix ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part.
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part.
    
    m : int
        On entry, ``m`` specifies the number of rows of the matrix ``c``.
        ``m`` must be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of the matrix ``c``.
        ``n`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, ka)``, where ``ka``
        is ``m`` when ``side`` = 'L' or 'l' and is ``n`` otherwise. Before
        entry, with ``side`` = 'L' or 'l', the ``m`` by ``m`` part of the
        array ``a`` must contain the hermitian matrix, such that when 
        ``uplo`` = 'U' or 'u', the leading ``m`` by ``m`` upper triangular
        part of the hermitian matrix and the strictly lower triangular part
        of ``a`` is not referenced, and when ``uplo`` = 'L' or 'l', the
        leading ``m`` by ``m``m lower triangular part of the array ``a`` must
        contain the lower triangular part of the hermitian matrix and the 
        strictly upper triangular part of ``a`` is not referenced.
        Before entry, with ``side`` = 'R' or 'r', the ``n`` by ``n`` part of
        the array ``a`` must contain the hermitian matrix, such that when
        ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``a`` must contain the uppert triangular part of
        the hermitian matrix and the strictly lower triangular part of ``a``
        is not referenced, and ``uplo`` = 'L' or 'l', the leading ``n`` by
        ``n`` lower triangular part of the array ``a`` must contain the lower
        triangular part of the hermitian matrix and the strictly upper
        triangular part of ``a`` is not referenced.
        Note that the imaginary parts of the diagonal elements need not be set,
        they are assumed to be zero.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``side`` = 'L' or 'l', then ``lda``
        must be at least ``max([1, m])``, otherwise ``lda`` must be at least
        ``max([1, n])``.
    b : 2d-array
        ``b`` is a complex array of dimension ``(ldb, n)``. Before entry, the
        leading ``m`` by ``n`` part of the array ``b`` must contain the matrix
        B.
    ldb : int
        On entry, ``ldb`` specifies the first dimension of ``b`` as declared
        in the calling (sub) program. ``ldb`` must be at least ``max([1, m])``.
    c : 2d-array
        ``c`` is a complex array of dimension ``(ldc, n)``. Before entry, the
        leading ``m`` by ``n`` part of the array ``c`` must contain the matrix
        C, except when beta is zero, in which case ``c`` need not be set on
        entry. On exit, the array ``c`` is overwritten by the ``m`` by ``n``
        updated matrix.
    ldc : int
        On entry, ``ldc`` specifies the first dimension of ``c`` as declared
        in the calling (sub) program. ``ldc`` must be at least ``max([1, m])``.
    
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Set NROWA as the number of rows of A.
    if lsame(side, 'L'):
        nrowa = m
    else:
        nrowa = n
    upper = lsame(uplo, 'U')
    
    # Test the input parameters
    info = 0
    if not lsame(side, 'L') and not lsame(side, 'R'):
        info = 1
    elif not upper and not lsame(uplo, 'L'):
        info = 2
    elif m<0:
        info = 3
    elif n<0:
        info = 4
    elif lda<max([1, nrowa]):
        info = 7
    elif ldb<max([1, m]):
        info = 9
    elif ldc<max([1, m]):
        info = 12
    
    if info!=0:
        xerbla('zhemm', info)
        return
    
    # Quick return if possible
    if m==0 or n==0 or (alpha==zero and beta==1):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if beta==zero:
            for j in xrange(n):
                for i in xrange(m):
                    c[i][j] = zero
        else:
            for j in xrange(n):
                for i in xrange(m):
                    c[i][j] = beta*c[i][j]
        return
    
    # Start the operations.
    if lsame(side, 'L'):
        # Form  C := alpha*A*B + beta*C
        if upper:
            for j in xrange(n):
                for i in xrange(m):
                    temp1 = alpha*b[i][j]
                    temp2 = zero
                    for k in xrange(i - 1):
                        c[k][j] = c[k][j] + temp1*a[k][i]
                        temp2 = temp2 + b[k][j]*dconjg(a[k][i])
                    if beta==zero:
                        c[i][j] = temp1*dble(a[i][i]) + alpha*temp2
                    else:
                        c[i][j] = beta*c[i][j] + temp1*dble(a[i][i]) + \
                            alpha*temp2
        else:
            for j in xrange(n):
                for i in reversed(xrange(m)):
                    temp1 = alpha*b[i][j]
                    temp2 = zero
                    for k in xrange(i + 1, m):
                        c[k][j] = c[k][j] + temp1*a[k][i]
                        temp2 = temp2 + b[k][j]*dconjg(a[k][i])
                    if beta==zero:
                        c[i][j] = temp1*dble(a[i][i]) + alpha*temp2
                    else:
                        c[i][j] = beta*c[i][j] + temp1*dble(a[i][i]) + \
                            alpha*temp2
    else:
        # Form  C := alpha*B*A + beta*C
        for j in xrange(n):
            temp1 = alpha*dble(a[j][j])
            if beta==zero:
                for i in xrange(m):
                    c[i][j] = temp1*b[i][j]
            else:
                for i in xrange(m):
                    c[i][j] = beta*c[i][j] + temp1*b[i][j]
            for k in xrange(j - 1):
                if upper:
                    temp1 = alpha*a[k][j]
                else:
                    temp1 = alpha*dconjg(a[j][k])
                for i in xrange(m):
                    c[i][j] = c[i][j] + temp1*c[i][k]
            for k in xrange(j + 1, n):
                if upper:
                    temp1 = alpha*dconjg(a[j][k])
                else:
                    temp1 = alpha*a[k][j]
                for i in xrange(m):
                    c[i][j] = c[i][j] + temp1*b[i][k]
    
    return

chemm = zhemm
__all__.append('chemm'); __all__.append('zhemm')

################################################################################

def zhemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy):
    """
    Performs the matrix-vector operation
    
        y := alpha*A*x + beta*y,
    
    where alpha and beta are scalars, x and y are n element vectors and A
    is an n by n hermitian matrix.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` spcifies whether the upper or lower triangular part
        of the array ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part
        
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``.  ``n`` must
        be at least zero.
    alpha : complex
        On entry, ``alpha`` spcifies the scalar alpha
    a : 2d-array
        ``a`` is a `complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo``='U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``a`` must contain the upper triangular part of the
        hermitian matrix and the strictly lower triangular part of ``a`` is
        not referenced. Before entry, with ``uplo`` = 'L' or 'l', the leading
        ``n`` by ``n`` lower triangular part of the array ``a`` must contain
        the lower triangular part of the hermitian matrix and the strictly
        upper triangular part of ``a`` is not referenced. Note that the 
        imaginary parts of the diagonal elements need not be set and are
        assumed to be zero.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``max([1, n])``.
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    beta : complex
        On entry, ``beta`` specifies the scalar beta. When ``beta`` is
        supplied as zero, then ``y`` need not be set on input.
    y : array
        ``y`` is a ``complex`` array of dimension at least
        ``(1 + (n - 1)*abs(incy))``. Before entry, the incremented array ``y``
        must contain the ``n`` element vector y. On exit, ``y`` is overwritten
        by the updated vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    
    Original Group
    --------------
    complex16_blas_level2
        
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif lda<max([1, n]):
        info = 5
    elif incx==0:
        info = 7
    elif incy==0:
        info = 10
    
    if info!=0:
        xerbla('zhemv', info)
        return
    
    # Quick return if possible
    if n==0 or (alpha==zero and beta==one):
        return
    
    # Set up the start points in X and Y
    if incx>0:
        kx = 0
    else:
        kx = (n - 1)*incx
    if incy>0:
        ky = 0
    else:
        ky = (n - 1)*incy
    
    # Start the operations. In this version, the elements of A are accessed
    # sequentially with one pass through the triangular part of A.
    
    # First form  y := beta*y
    if beta!=one:
        if incy==1:
            if beta==zero:
                for i in xrange(n):
                    y[i] = zero
            else:
                for i in xrange(n):
                    y[i] = beta*y[i]
        else:
            iy = ky
            if beta==zero:
                for i in xrange(n):
                    y[iy] = zero
                    iy = iy + incy
            else:
                for i in xrange(n):
                    y[iy] = beta*y[iy]
                    iy = iy + incy
    
    if alpha==zero:
        return
    if lsame(uplo, 'U'):
        # Form  y  when A is stored in upper triangle.
        if incx==1 and incy==1:
            for j in xrange(n):
                temp1 = alpha*x[j]
                temp2 = zero
                for i in xrange(j - 1):
                    y[i] = y[i] + temp1*a[i][j]
                    temp2 = temp2 + dconjg(a[i][j])*x[i]
                y[j] = y[j] + temp1*dble(a[j][j]) + alpha*temp2
        else:
            jx = kx
            jy = ky
            for j in xrange(n):
                temp1 = alpha*x[jx]
                temp2 = zero
                ix = kx
                iy = ky
                for i in xrange(j - 1):
                    y[iy] = y[iy] + temp1*a[i][j]
                    temp2 = temp2 + dconjg(a[i][j])*x[ix]
                    ix = ix + incx
                    iy = iy + incy
                y[jy] = y[jy] + temp1*dble(a[j][j]) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
    else:
        # Form  y  when A is stored in lower triangle
        if incx==1 and incy==1:
            for j in xrange(n):
                temp1 = alpha*x[j]
                temp2 = zero
                y[j] = y[j] + temp1*dble(a[j][j])
                for i in xrange(j + 1, n):
                    y[i] = y[i] + temp1*a[i][j]
                    temp2 = temp2 + conjg(a[i][j])*x[i]
                y[j] = y[j] + alpha*temp2
        else:
            jx = kx
            jy = ky
            for j in xrange(n):
                temp1 = alpha*x[jx]
                temp2 = zero
                y[jy] = y[jy] + temp1*dble(a[j][j])
                ix = jx
                iy = jy
                for i in xrange(j + 1, n):
                    ix = ix + incx
                    iy = iy + incy
                    y[iy] = y[iy] + temp1*a[i][j]
                    temp2 = temp2 + dconjg(a[i][j])*x[ix]
                y[jy] = y[jy] + alpha*temp2
                jx = jx + incx
                jy = jy + incy
    
    return

chemv = zhemv
__all__.append('chemv'); __all__.append('zhemv')

################################################################################

def zher(uplo, n, alpha, x, incx, a, lda):
    """
    Performs the hermitian rank 1 operation
    
        A := alpha*x*x**H + A,
    
    where alpha is a real scalar, x is an n element vector and A is an n by n
    hermitian matrix.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` spcifies whether the upper or lower triangular part
        of the array ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part
        
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``.  ``n`` must
        be at least zero.
    alpha : complex
        On entry, ``alpha`` spcifies the scalar alpha
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    a : 2d-array
        ``a`` is a `complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo``='U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``a`` must contain the upper triangular part of the
        hermitian matrix and the strictly lower triangular part of ``a`` is
        not referenced. On exit, the upper triangular part of the array ``a``
        is overwritten by the upper triangular part of the updated matrix.
        Before entry, with ``uplo`` = 'L' or 'l', the leading
        ``n`` by ``n`` lower triangular part of the array ``a`` must contain
        the lower triangular part of the hermitian matrix and the strictly
        upper triangular part of ``a`` is not referenced. On exit, the lower
        triangular part of the array ``a`` is overwritten by the lower
        triangular part of the updated matrix. Note that the 
        imaginary parts of the diagonal elements need not be set and are
        assumed to be zero, and on exit they are set to zero.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``max([1, n])``.
    
    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 5
    elif lda<max([1, n]):
        info = 7
    
    if info!=0:
        xerbla('zher', info)
        return
    
    # Quick return if possible
    if n==0 or alpha==dble(zero):
        return
    
    # Set the start point in X if the increment is not unity.
    if incx<=0:
        kx = (n - 1)*incx
    elif incx!=0:
        kx = 0
    
    # Start the operations. In this version, the elements of A are accessed
    # sequentially with one pass through the triangular part of A.
    if lsame(uplo, 'U'):
        # Form  A  when A is stored in upper triangle
        if incx==1:
            for j in xrange(n):
                if x[j]!=zero:
                    temp = alpha*dconjg(x[j])
                    for i in xrange(j-1):
                        a[i][j] = a[i][j] + x[i]*temp
                    a[j][j] = dble(a[j][j]) + dble(x[j]*temp)
                else:
                    a[j][j] = dble(a[j][j])
        else:
            jx = kx
            for j in xrange(n):
                if x[j]!=zero:
                    temp = alpha*dconjg(x[jx])
                    ix = kx
                    for i in xrange(j - 1):
                        a[i][j] = a[i][j] + x[ix]*temp
                        ix = ix + incx
                    a[j][j] = dble(a[j][j]) + dble(x[jx]*temp)
                else:
                    a[j][j] = dble(a[j][j])
                jx = jx + incx
    else:
        # Form  A  when A is stored in lower triangle.
        if incx==1:
            for j in xrange(n):
                if x[j]!=zero:
                    temp = alpha*dconjg(x[j])
                    a[j][j] = dble(a[j][j]) + dble(temp*x[j])
                    for i in xrange(j + 1, n):
                        a[i][j] = a[i][j] + x[i]*temp 
                else:
                    a[j][j] = dble(a[j][j])
        else:
            jx = kx
            for j in xrange(n):
                if x[jx]!=zero:
                    temp = alpha*dconjg(x[jx])
                    a[j][j] = dble(a[j][j]) + dble(temp*x[jx])
                    ix = jx
                    for i in xrange(j + 1, n):
                        ix = ix + incx
                        a[i][j] = a[i][j] + x[ix]*temp 
                else:
                    a[j][j] = dble(a[j][j])
                jx = jx + incx
                
    return

cher = zher
__all__.append('cher'); __all__.append('zher')

################################################################################

def zher2(uplo, n, alpha, x, incx, y, incy, a, lda):
    """
    Performs the hermitian rank 2 operation
    
        A := alpha*x*y**H + conjg(alpha)*y*x**H + A,
    
    where alpha is a real scalar, x and y are n element vectors and A is an 
    n by n hermitian matrix.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` spcifies whether the upper or lower triangular part
        of the array ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part
        
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``.  ``n`` must
        be at least zero.
    alpha : complex
        On entry, ``alpha`` spcifies the scalar alpha
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    y : array
        ``y`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incy))``. Before entry, the incremented array ``y``
        must contain the ``n`` element vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    a : 2d-array
        ``a`` is a `complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo``='U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``a`` must contain the upper triangular part of the
        hermitian matrix and the strictly lower triangular part of ``a`` is
        not referenced. On exit, the upper triangular part of the array ``a``
        is overwritten by the upper triangular part of the updated matrix.
        Before entry, with ``uplo`` = 'L' or 'l', the leading
        ``n`` by ``n`` lower triangular part of the array ``a`` must contain
        the lower triangular part of the hermitian matrix and the strictly
        upper triangular part of ``a`` is not referenced. On exit, the lower
        triangular part of the array ``a`` is overwritten by the lower
        triangular part of the updated matrix. Note that the 
        imaginary parts of the diagonal elements need not be set and are
        assumed to be zero, and on exit they are set to zero.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``max([1, n])``.
    
    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 5
    elif incy==0:
        info = 7
    elif lda<max([1, n]):
        info = 9
    
    if info!=0:
        xerbla('zher2', info)
        return
    
    # Quick return if possible
    if n==0 or alpha==dble(zero):
        return
    
    # Set the start points in X and Y if the increment are not both unity.
    if incx!=1 or incy!=1:
        if incx>0:
            kx = 0
        else:
            kx = (n - 1)*incx
        if incy>0:
            ky = 0
        else:
            ky = (n - 1)*incy
        jx = kx
        jy = ky
    
    # Start the operations. In this version, the elements of A are accessed
    # sequentially with one pass through the triangular part of A.
    if lsame(uplo, 'U'):
        # Form  A  when A is stored in upper triangle
        if incx==1 and incy==1:
            for j in xrange(n):
                if x[j]!=zero and y[j]!=zero:
                    temp1 = alpha*dconjg(y[j])
                    temp2 = dconj(alpha*x[j])
                    for i in xrange(j-1):
                        a[i][j] = a[i][j] + x[i]*temp1 + y[i]*temp2
                    a[j][j] = dble(a[j][j]) + dble(x[j]*temp1 + y[j]*temp2)
                else:
                    a[j][j] = dble(a[j][j])
        else:
            for j in xrange(n):
                if x[jx]!=zero and y[jy]!=zero:
                    temp1 = alpha*dconjg(y[jy])
                    temp2 = dconjg(alpha*x[jx])
                    ix = kx
                    iy = ky
                    for i in xrange(j - 1):
                        a[i][j] = a[i][j] + x[ix]*temp1 + y[iy]*temp2
                        ix = ix + incx
                        iy = iy + incy
                    a[j][j] = dble(a[j][j]) + dble(x[jx]*temp1 + y[jy]*temp2)
                else:
                    a[j][j] = dble(a[j][j])
                jx = jx + incx
                jy = jy + incy
    else:
        # Form  A  when A is stored in lower triangle.
        if incx==1 and incy==1:
            for j in xrange(n):
                if x[j]!=zero or y[j]!=zero:
                    temp1 = alpha*dconjg(y[j])
                    temp2 = dconjg(alpha*x[j])
                    a[j][j] = dble(a[j][j]) + dble(x[j]*temp1 + y[j]*temp2)
                    for i in xrange(j + 1, n):
                        a[i][j] = a[i][j] + x[i]*temp 
                else:
                    a[j][j] = dble(a[j][j])
        else:
            for j in xrange(n):
                if x[jx]!=zero or y[jy]!=zero:
                    temp1 = alpha*dconjg(y[jy])
                    temp2 = dconjg(alpha*x[jx])
                    a[j][j] = dble(a[j][j]) + dble(x[jx]*temp1 + y[jy]*temp2)
                    ix = jx
                    iy = jy
                    for i in xrange(j + 1, n):
                        ix = ix + incx
                        iy = iy + incy
                        a[i][j] = a[i][j] + x[ix]*temp1 + y[iy]*temp2 
                else:
                    a[j][j] = dble(a[j][j])
                jx = jx + incx
                jy = jy + incy
                
    return

cher2 = zher2
__all__.append('cher2'); __all__.append('zher2')

################################################################################

def zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    """
    Performs one of the hermitian rank 2k operations
    
        C := alpha*A*B**H + conjg(alpha)*B*A**H + beta*C,
    
    or
    
        C := alpha*A**H*B + conjg(alpha)*B**H*A + beta*C,
    
    where alpha and beta are scalars with beta real, C is an n by n hermitian
    matrix and A and B are n by k matrices in the first case and k by n
    matrices in the second case.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the array ``c`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part of ``c``
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part of ``c``
    
    trans : str 
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  C := alpha*A*B**H + conjg(alpha)*B*A**H + beta*C
        - ``trans`` = 'C' or 'c'  C := alpha*A**H*B + conjg(alpha)*B**H*A + beta*C
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``c``. ``n`` must be
        at least zero
    k : int
        On entry with ``trans`` = 'N' or 'n', ``k`` specifies the number of
        columns of the matrices ``a`` and ``b``, and on entry with ``trans`` = 
        'C' or 'c', ``k`` specifies the number of rows of the matrices ``a``
        and ``b``. ``k`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha
    a : 2d-array
        ``a`` is a complex array of dimension ``(lda, ka)``, where ka is ``k``
        when ``trans`` = 'N' or 'n', and is ``n`` otherwise. Before entry,
        with ``trans`` = 'N' or 'n', the leading ``n`` by ``k`` part of the
        array ``a`` must contain the matrix A, otherwise the leading ``k``
        by ``n`` part of the array ``a`` must contain the matrix A.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``trans`` = 'N' or 'n', then ``lda``
        must be at least ``max([1, n])``, otherwise ``lda`` must be at least 
        ``max([1, k])``.
    b : 2d-array
        ``b`` is a complex array of dimension ``(ldb, kb)``, where kb is ``k``
        when ``trans`` = 'N' or 'n', and is ``n`` otherwise. Before entry,
        with ``trans`` = 'N' or 'n', the leading ``n`` by ``k`` part of the
        array ``a`` must contain the matrix A, otherwise the leading ``k``
        by ``n`` part of the array ``a`` must contain the matrix A.
    ldb : int
        On entry, ``ldb`` specifies the first dimension of ``b`` as declared
        in the calling (sub) program. When ``trans`` = 'N' or 'n', then ``ldb``
        must be at least ``max([1, n])``, otherwise ``ldb`` must be at least 
        ``max([1, k])``. Unchanged on exit.
    beta : float
        On entry, ``beta`` specifies the scalar beta.
    c : 2d-array
        ``c`` is a ``complex`` array of dimension ``(ldc, n)``. Before entry,
        with ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper
        triangular part of the array ``c`` must contain the upper triangular
        part of th hermitian matrix and the strictly lower triangular part of
        ``c`` is not referenced. On exit, the upper triangular part of the 
        array ``c`` is overwritten by the upper triangular part of the updated
        matrix.
        Before entry with ``uplo`` = 'L' or 'l', the leading ``n`` by ``n``
        lower triangular part of the array ``c`` must contain the lower
        triangular part of the hermitian matrix and the strictly upper
        triangular part of ``c`` is not referenced. On exit, the lower
        triangular part of the array ``c`` is overwritten by the lower
        triangular part of the updated matrix.
        Note that the imaginary parts of the diagonal elements need not be set,
        they are assumed to be zero, and on exit they are set to zero.
    ldc : int
        On entry, ``ldc`` specifies the first dimension of ``c`` as declared
        in the calling (sub) program. ``ldc`` must be at least ``max([1, n])``.
        
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = 1.0
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    if lsame(trans, 'N'):
        nrowa = n
    else:
        nrowa = k
    upper = lsame(uplo, 'U')
    
    info = 0
    if not upper and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'C'):
        info = 2
    elif n<0:
        info = 3
    elif k<0:
        info = 4
    elif lda<max([1, nrowa]):
        info = 7
    elif ldb<max([1, nrowa]):
        info = 9
    elif ldc<max([1, n]):
        info = 12
    
    if info!=0:
        xerbla('zher2k', info)
        return
    
    # Quick return if possible
    if n==0 or ((alpha==zero or k==0) and beta==one):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if upper:
            if beta==dble(zero):
                for j in xrange(n):
                    for i in xrange(j):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    for i in xrange(j - 1):
                        c[i][j] = beta*c[i][j]
                    c[j][j] = beta*dble(c[j][j])
        else:
            if beta==dble(zero):
                for j in xrange(n):
                    for i in xrange(j, n):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    c[j][j] = beta*dble(c[j][j])
                    for i in xrange(j + 1, n):
                        c[i][j] = beta*c[i][j]
        return
    
    # Start the operations
    if lsame(trans, 'N'):
        # Form  C := alpha*A*B**H = conjg(alpha)*B*A**H + C
        if upper:
            for j in xrange(n):
                if beta==dble(zero):
                    for i in xrange(j):
                        c[i][j] = zero
                elif beta!=one:
                    for i in xrange(j - 1):
                        c[i][j] = beta*c[i][j]
                    c[j][j] = beta*dble(c[j][j])
                else:
                    c[j][j] = dble(c[j][j])
                for l in xrange(k):
                    if a[j][l]!=zero or b[j][l]!=zero:
                        temp1 = alpha*dconjg(b[j][l])
                        temp2 = dconjg(alpha*a[j][l])
                        for i in xrange(j - 1):
                            c[i][j] = c[i][j] + a[i][l]*temp1 + b[i][l]*temp2
                        c[j][j] = dble(c[j][j]) + dble(a[j][l]*temp1 + \
                            b[j][l]*temp2)
        else:
            for j in xrange(n):
                if beta==dble(zero):
                    for i in xrange(j + 1, n):
                        c[i][j] = zero
                elif beta!=one:
                    for i in xrange(j, n):
                        c[i][j] = beta*c[i][j]
                    c[j][j] = beta*dble(c[j][j])
                else:
                    c[j][j] = dble(c[j][j])
                for l in xrange(k):
                    if a[j][l]!=zero or b[j][l]!=zero:
                        temp1 = alpha*dconjg(b[j][l])
                        temp2 = dconjg(alpha*a[j][l])
                        for i in xrange(j, n):
                            c[i][j] = c[i][j] + a[i][l]*temp1 + b[i][l]*temp2
                        c[j][j] = dble(c[j][j]) + dble(a[j][l]*temp1 + \
                            b[j][l]*temp2)
    else:
        # Form C := alpha*A**H*B + conjg(alpha)*B**H*A + C
        if upper:
            for j in xrange(n):
                for i in xrange(j):
                    temp1 = zero
                    temp2 = zero
                    for l in xrange(k):
                        temp1 = temp1 + dconjg(a[l][i])*b[l][j]
                        temp2 = temp2 + dconjg(b[l][i])*a[l][j]
                    if i==j:
                        if beta==dble(zero):
                            c[j][j] = dble(alpha*temp1 + dconjg(alpha)*temp2)
                        else:
                            c[j][j] = beta*dble(c[j][j]) + dble(alpha*temp1 + \
                                dconjg(alpha)*temp2)
                    else:
                        if beta==dble(zero):
                            c[i][j] = alpha*temp1 + dconjg(alpha)*temp2
                        else:
                            c[i][j] = beta*c[i][j] + alpha*temp1 + \
                                dconjg(alpha)*temp2
        else:
            for j in xrange(n):
                for i in xrange(j, n):
                    temp1 = zero
                    temp2 = zero
                    for l in xrange(k):
                        temp1 = temp1 + dconjg(a[l][i])*b[l][j]
                        temp2 = temp2 + dconjg(b[l][i])*a[l][j]
                    if i==j:
                        if beta==dble(zero):
                            c[j][j] = dble(alpha*temp1 + dconjg(alpha)*temp2)
                        else:
                            c[j][j] = beta*dble(c[j][j]) + dble(alpha*temp1 + \
                                dconjg(alpha)*temp2)
                    else:
                        if beta==dble(zero):
                            c[i][j] = alpha*temp1 + dconjg(alpha)*temp2
                        else:
                            c[i][j] = beta*c[i][j] + alpha*temp1 + \
                                dconjg(alpha)*temp2
    
    return

cher2k = zher2k
__all__.append('cher2k'); __all__.append('zher2k')

################################################################################

def zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc):
    """
    Performs one of the hermitian rank k operations
    
        C := alpha*A*A**H + beta*C,
    
    or
    
        C := alpha*A**H*A + beta*C,
    
    where alpha and beta are real scalars, C is an n by n hermitian matrix and
    A is an n by k matrix in the first case and a k by n matrix in the second
    case.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the array ``c`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part of ``c``
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part of ``c``
    
    trans : str 
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  C := alpha*A*A**H + beta*C
        - ``trans`` = 'C' or 'c'  C := alpha*A**H*A + beta*C
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``c``. ``n`` must be
        at least zero
    k : int
        On entry with ``trans`` = 'N' or 'n', ``k`` specifies the number of
        columns of the matrix ``a``, and on entry with ``trans`` = 
        'C' or 'c', ``k`` specifies the number of rows of the matrix ``a``.
        ``k`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha
    a : 2d-array
        ``a`` is a complex array of dimension ``(lda, ka)``, where ka is ``k``
        when ``trans`` = 'N' or 'n', and is ``n`` otherwise. Before entry,
        with ``trans`` = 'N' or 'n', the leading ``n`` by ``k`` part of the
        array ``a`` must contain the matrix A, otherwise the leading ``k``
        by ``n`` part of the array ``a`` must contain the matrix A.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``trans`` = 'N' or 'n', then ``lda``
        must be at least ``max([1, n])``, otherwise ``lda`` must be at least 
        ``max([1, k])``.
    beta : float
        On entry, ``beta`` specifies the scalar beta.
    c : 2d-array
        ``c`` is a ``complex`` array of dimension ``(ldc, n)``. Before entry,
        with ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper
        triangular part of the array ``c`` must contain the upper triangular
        part of th hermitian matrix and the strictly lower triangular part of
        ``c`` is not referenced. On exit, the upper triangular part of the 
        array ``c`` is overwritten by the upper triangular part of the updated
        matrix.
        Before entry with ``uplo`` = 'L' or 'l', the leading ``n`` by ``n``
        lower triangular part of the array ``c`` must contain the lower
        triangular part of the hermitian matrix and the strictly upper
        triangular part of ``c`` is not referenced. On exit, the lower
        triangular part of the array ``c`` is overwritten by the lower
        triangular part of the updated matrix.
        Note that the imaginary parts of the diagonal elements need not be set,
        they are assumed to be zero, and on exit they are set to zero.
    ldc : int
        On entry, ``ldc`` specifies the first dimension of ``c`` as declared
        in the calling (sub) program. ``ldc`` must be at least ``max([1, n])``.
        
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = 1.0
    zero = 0.0
    
    # Test the input parameters
    if lsame(trans, 'N'):
        nrowa = n
    else:
        nrowa = k
    upper = lsame(uplo, 'U')
    
    info = 0
    if not upper and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'C'):
        info = 2
    elif n<0:
        info = 3
    elif k<0:
        info = 4
    elif lda<max([1, nrowa]):
        info = 7
    elif ldc<max([1, n]):
        info = 10
    
    if info!=0:
        xerbla('zherk', info)
        return
    
    # Quick return if possible
    if n==0 or ((alpha==zero or k==0) and beta==one):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if upper:
            if beta==zero:
                for j in xrange(n):
                    for i in xrange(j):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    for i in xrange(j - 1):
                        c[i][j] = beta*c[i][j]
                    c[j][j] = beta*dble(c[j][j])
        else:
            if beta==zero:
                for j in xrange(n):
                    for i in xrange(j, n):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    c[j][j] = beta*dble(c[j][j])
                    for i in xrange(j + 1, n):
                        c[i][j] = beta*c[i][j]
        return
        
    # Start the operations
    if lsame(trans, 'N'):
        # Form  C := alpha*A*A**H + beta*C
        if upper:
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(j):
                        c[i][j] = zero
                elif beta!=one:
                    for i in xrange(j - 1):
                        c[i][j] = beta*c[i][j]
                    c[j][j] = beta*dble(c[j][j])
                else:
                    c[j][j] = dble(c[j][j])
                for l in xrange(k):
                    if a[j][l]!=dcomplx(zero):
                        temp = alpha*dconjg(a[j][l])
                        for i in xrange(j - 1):
                            c[i][j] = c[i][j] + temp*a[i][l]
                        c[j][j] = dble(c[j][j]) + dble(temp*a[i][l])
        else:
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(j, n):
                        c[i][j] = zero
                elif beta!=one:
                    c[j][j] = beta*dble(c[j][j])
                    for i in xrange(j + 1, n):
                        c[i][j] = beta*c[i][j]
                else:
                    c[j][j] = dble(c[j][j])
                for l in xrange(k):
                    if a[j][l]!=dcomplx(zero):
                        temp = alpha*dconjg(a[j][l])
                        c[j][j] = dble(c[j][j]) + dble(temp*a[j][l])
                        for i in xrange(j + 1, n):
                            c[i][j] = c[i][j] + temp*a[i][l]
    else:
        # Form  C := alpha*A**H*A + beta*C
        if upper:
            for j in xrange(n):
                for i in xrange(j - 1):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + dconjg(a[l][i])*a[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
                rtemp = zero
                for l in xrange(k):
                    rtemp = rtemp + dconjg(a[l][j])*a[l][j]
                if beta==zer0:
                    c[j][j] = alpha*rtemp
                else:
                    c[j][j] = alpha*rtemp + beta*dble(c[j][j])
        else:
            for j in xrange(n):
                rtemp = zero
                for l in xrange(k):
                    rtemp = rtemp + dconjg(a[l][j])*a[l][j]
                if beta==zero:
                    c[j][j] = alpha*rtemp
                else:
                    c[j][j] = alpha*rtemp + beta*dble(c[j][j])
                for i in xrange(j + 1, n):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + dconjg(a[l][i])*a[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
                    
    return

cherk = zherk
__all__.append('cherk'); __all__.append('zherk')

################################################################################

def zhpmv(uplo, n, alpha, ap, x, incx, beta, y, incy):
    """
    Performs the matrix-vector operation
    
        y := alpha*A*x + beta*y,
    
    where alpha and beta are scalars, x, and y are n element vectors and
    A is an n by n hermitian matrix, supplied in packed form.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the array ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  The upper triangular part of ``a`` is supplied
          in ``ap``.
        - ``uplo`` = 'L' or 'l'  The lower triangular part of ``a`` is supplied
          in ``ap``.
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must be
        at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    ap : array
        ``ap`` is a ``complex`` array of dimension at least ``((n*(n + 1))/2)``.
        Before entry, with ``uplo`` = 'U' or 'u', the array ``ap`` must contain
        the upper triangular part of the hermitian matrix packed sequentially,
        column by column, so that ``ap[0]`` conatins ``a[0][0]``, ``ap[1]`` and
        ``ap[2]`` contain ``a[0][1]`` and ``a[0][2]`` respectively, and so on.
        Before entry, with ``uplo`` = 'L' or 'l', the array ``ap`` must contain
        the lower triangular part of the hermitian matrix packed sequentially,
        column by column, so that ``ap[0]`` contains ``a[0][0]``, ``ap[1]`` and
        ``ap[2]`` contain ``a[1][0]`` and ``a[2][0]`` respectively, and so on.
        Note that the imaginary parts of the diagonal elements need not be set
        and are assumed to be zero.
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    beta : complex
        On entry, ``beta`` specifies the scalar beta. When ``beta`` is supplied
        as zero, then ``y`` need not be set on input.
    y : array
        ``y`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incy))``. Before entry, the incremented array ``y``
        must contain the ``n`` element vector y. On exit, ``y`` is overwritten
        by the updated vector y
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    
    Original Group
    --------------
    complex16_blas_level2
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 6
    elif incy==0:
        info = 9
    
    if info!=0:
        xerbla('zhpmv', info)
        return
    
    # Quick return if possible
    if n==0 or (alpha==zero and beta==one):
        return
    
    # Set up the start points of X and Y
    if incx>0:
        kx = 0
    else:
        kx = (n - 1)*incx
    if incy>0:
        ky = 0
    else:
        ky = (n - 1)*incy
        
    # Start the operations. In this version, the elements of the array AP
    # are accessed sequentially with one pass through AP.
    
    # First form  y := beta*y.
    if beta!=one:
        if incy==1:
            if beta==zero:
                for i in xrange(n):
                    y[i] = zero
            else:
                for i in xrange(n):
                    y[i] = beta*y[i]
        else:
            iy = ky
            if beta==zero:
                for i in xrange(n):
                    y[iy] = zero
                    iy = iy + incy
            else:
                for i in xrange(n):
                    y[iy] = beta*y[iy]
                    iy = iy + incy
    
    if alpha==zero:
        return
    kk = 0
    if lsame(uplo, 'U'):
        # Form  y  when AP contains the upper triangle.
        if incx==1 and incy==1:
            for j in xrange(n):
                temp1 = alpha*x[j]
                temp2 = zero
                k = kk
                for i in xrange(j - 1):
                    y[i] = y[i] + temp1*ap[k]
                    temp2 = temp2 + dconjg(ap[k])*x[i]
                    k = k + 1
                y[j] = y[j] + temp1*dble(ap[kk + j - 1]) + alpha*temp2
                kk = kk + j
        else:
            jx = kx
            jy = ky
            for j in xrange(n):
                temp1 = alpha*x[jx]
                temp2 = zero
                ix = kx
                ky = ky
                for k in xrange(kk, kk + j - 2):
                    y[iy] = y[iy] + temp1*ap[k]
                    temp2 = temp2 + dconjg(ap[k])*x[ix]
                    ix = ix + incx
                    iy = iy + incy
                y[jy] = y[jy] + temp1*dble(ap[kk + j - 1]) + alpha*temp2
                jx = jx + incx
                jy = jy + incy
                kk = kk + j
    else:
        # Form  y  when AP contains the lower triangle
        if incx==1 and incy==1:
            for j in xrange(n):
                temp1 = alpha*x[j]
                temp2 = zero
                y[j] = y[j] + temp1*dble(ap[kk])
                k = kk + 1
                for i in xrange(j + 1, n):
                    y[i] = y[i] + temp1*ap[k]
                    temp2 = temp2 + dconjg(ap[k])*x[i]
                    k = k + 1
                y[j] = y[j] + alpha*temp2
                kk = kk + (n - j + 1)
        else:
            jx = kx
            jy = ky
            for j in xrange(n):
                temp1 = alpha*x[jx]
                temp2 = zero
                y[jy] = y[jy] + temp1*dble(ap[kk])
                ix = jx
                ky = jy
                for k in xrange(kk + 1, kk + n - j):
                    ix = ix + incx
                    iy = iy + incy
                    y[iy] = y[iy] + temp1*ap[k]
                    temp2 = temp2 + dconnjg(ap[k])*x[ix]
                y[jy] = y[jy] + alpha*temp2
                jx = jx + incx
                jy = jy + incy
                kk = kk + (n - j + 1)
    
    return

chpmv = zhpmv
__all__.append('chpmv'); __all__.append('zhpmv')

################################################################################

def zhpr(uplo, n, alpha, x, incx, ap):
    """
    Performs the hermitian rank 1 operation
    
        A := alpha*x*x**H + A,
    
    where alpha is a real scalar, x is an n element vector and A is an
    n by n hermitian matrix, supplied in packed form.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the array ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  The upper triangular part of ``a`` is supplied
          in ``ap``.
        - ``uplo`` = 'L' or 'l'  The lower triangular part of ``a`` is supplied
          in ``ap``.
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must be
        at least zero.
    alpha : float
        On entry, ``alpha`` specifies the scalar alpha.
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    ap : array
        ``ap`` is a ``complex`` array of dimension at least ``((n*(n + 1))/2)``.
        Before entry, with ``uplo`` = 'U' or 'u', the array ``ap`` must contain
        the upper triangular part of the hermitian matrix packed sequentially,
        column by column, so that ``ap[0]`` conatins ``a[0][0]``, ``ap[1]`` and
        ``ap[2]`` contain ``a[0][1]`` and ``a[0][2]`` respectively, and so on.
        Before entry, with ``uplo`` = 'L' or 'l', the array ``ap`` must contain
        the lower triangular part of the hermitian matrix packed sequentially,
        column by column, so that ``ap[0]`` contains ``a[0][0]``, ``ap[1]`` and
        ``ap[2]`` contain ``a[1][0]`` and ``a[2][0]`` respectively, and so on.
        Note that the imaginary parts of the diagonal elements need not be set
        and are assumed to be zero.

    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 5
    
    if info!=0:
        xerbla('zhpr', info)
        return
    
    # Quick return if possible
    if n==0 or alpha==dble(zero):
        return
    
    # Set the start point in X if the increment is not unity.
    if incx<=0:
        kx = (n - 1)*incx
    else:
        kx = 0
    
    # Start the operations. In this version, the elements of the array AP are
    # accessed sequentially with one pass through AP.
    
    kk = 0
    if lsame(uplo, 'U'):
        # Form  A  when upper triangle is stored in AP
        if incx==1:
            for j in xrange(n):
                if x[j]!=zero:
                    temp = alpha*dconjg(x[j])
                    k = kk
                    for i in xrange(j - 1):
                        ap[k] = a[k] + x[i]*temp
                        k = k + 1
                    ap[kk + j - 1] = dble(ap[kk + j - 1]) + dble(x[j]*temp)
                else:
                    ap[kk + j - 1] = dble(ap[kk + j - 1])
                kk = kk + j
        else:
            jx = kx
            for j in xrange(n):
                if x[jx]!=zero:
                    temp = alpha*dconjg(x[jx])
                    ix = kx
                    for k in xrange(kk, kk + j - 2):
                        ap[k] = ap[k] + x[ix]*temp
                        ix = ix + incx
                    ap[kk + j - 1] = dble(ap[kk + j - 1]) + dble(x[jx]*temp)
                else:
                    ap[kk + j - 1] = dble(ap[kk + j - 1])
                jx = jx + incx
                kk = kk + j
    else:
        # Form  A  when lower triangle is stored in AP
        if incx==1:
            for j in xrange(n):
                if x[j]!=zero:
                    temp = alpha*dconjg(x[j])
                    ap[kk] = dble(ap[kk]) + dble(temp*x[j])
                    k = kk + 1
                    for i in xrange(j + 1, n):
                        ap[k] = ap[k] + x[i]*temp
                        k = k + 1
                else:
                    ap[kk] = dble(ap[kk])
                kk = kk + n - j + 1
        else:
            for j in xrange(n):
                if x[jx]!=zero:
                    temp = alpha*dconjg(x[jx])
                    ap[kk] = dble(ap[kk]) + dble(temp*x[jx])
                    ix = jx
                    for k in xrange(kk + 1, kk + n - j):
                        ix = i + incx
                        ap[k] = ap[k] + x[ix]*temp
                else:
                    ap[kk] = dble(ap[kk])
                jx = jx + incx
                kk = kk + n - j + 1
    
    return

chpr = zhpr
__all__.append('chpr'); __all__.append('zhpr')

################################################################################

def zhpr2(uplo, n, alpha, x, incx, y, incy, ap):
    """
    Performs the hermitian rank 2 operation
    
        A := alpha*x*y**H + conjg(alpha)*y*x**H + A,
    
    where alpha is a scalar, x and y are n element vectors and A is an n by n
    hermitian matrix, supplied in packed form.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the array ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  The upper triangular part of ``a`` is supplied
          in ``ap``.
        - ``uplo`` = 'L' or 'l'  The lower triangular part of ``a`` is supplied
          in ``ap``.
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must be
        at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    y : array
        ``y`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incy))``. Before entry, the incremented array ``y``
        must contain the ``n`` element vector y.
    incy : int
        On entry, ``incy`` specifies the increment for the elements of ``y``.
        ``incy`` must not be zero.
    ap : array
        ``ap`` is a ``complex`` array of dimension at least ``((n*(n + 1))/2)``.
        Before entry, with ``uplo`` = 'U' or 'u', the array ``ap`` must contain
        the upper triangular part of the hermitian matrix packed sequentially,
        column by column, so that ``ap[0]`` conatins ``a[0][0]``, ``ap[1]`` and
        ``ap[2]`` contain ``a[0][1]`` and ``a[0][2]`` respectively, and so on.
        Before entry, with ``uplo`` = 'L' or 'l', the array ``ap`` must contain
        the lower triangular part of the hermitian matrix packed sequentially,
        column by column, so that ``ap[0]`` contains ``a[0][0]``, ``ap[1]`` and
        ``ap[2]`` contain ``a[1][0]`` and ``a[2][0]`` respectively, and so on.
        Note that the imaginary parts of the diagonal elements need not be set
        and are assumed to be zero.

    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif n<0:
        info = 2
    elif incx==0:
        info = 5
    elif incy==0:
        info = 7
    
    if info!=0:
        xerbla('zhpr2', info)
        return
    
    # Quick return if possible
    if n==0 or alpha==zero:
        return
    
    # Set up the start points in X and Y if the increments are not both unity
    if incx!=1 or incy!=1:
        if incx>0:
            kx = 0
        else:
            kx = (n - 1)*incx
        if incy>0:
            ky = 0
        else:
            ky = (n - 1)*incy
        jx = kx
        jy = ky
    
    # Start the operations. In this version, the elements of the array AP are
    # accessed sequentially with one pass through AP.
    kk = 0
    if lsame(uplo, 'U'):
        # Form  A  when upper triangle is stored in AP.
        if incx==1 and incy==1:
            for j in xrange(n):
                if x[j]!=zero or y[j]!=zero:
                    temp1 = alpha*dconjg(y[j])
                    temp2 = dconjg(alpha*x[j])
                    k = kk
                    for i in xrange(j - 1):
                        ap[k] = ap[k] + x[i]*temp1 + y[i]*temp2
                        k = k + 1
                    ap[kk + j - 1] = dble(ap[kk + j - 1]) + \
                        dble(x[j]*temp1 + y[j]*temp2)
                else:
                    ap[kk + j - 1] = dble(ap[kk + j - 1])
                kk = kk + j
        else:
            for j in xrange(n):
                if x[jx]!=zero or y[jy]!=zero:
                    temp1 = alpha*dconjg(y[jy])
                    temp2 = dconjg(alpha*x[jx])
                    ix = kx
                    iy = ky
                    for k in xrange(kk, kk + j - 2):
                        ap[k] = ap[k] + x[ix]*temp1 + y[iy]*temp2
                        ix = ix + incx
                        iy = iy + incy
                    ap[kk + j - 1] = dble(ap[kk + j - 1]) + \
                        dble(x[jx]*temp1 + y[jy]*temp2)
                else:
                    ap[kk + j - 1] = dble(ap[kk + j - 1])
                jx = jx + incx
                jy = jy + incy
                kk = kk + j
    else:
        # Form  A  when lower triangle is stored in AP.
        if incx==1 and incy==1:
            for j in xrange(n):
                if x[j]!=zero or y[j]!=zero:
                    temp1 = alpha*dconjg(y[j])
                    temp2 = dconjg(alpha*x[j])
                    ap[kk] = dble(ap[kk]) + dble(x[j]*temp1 + y[j]*temp2)
                    k = kk + 1
                    for i in xrange(j + 1, n):
                        ap[k] = ap[k] + x[i]*temp1 + y[i]*temp2
                        k = k + 1
                else:
                    ap[kk] = dble(ap[kk])
                kk = kk + n - j + 1
        else:
            for j in xrange(n):
                if x[jx]!=zero or y[jy]!=zero:
                    temp1 = alpha*dconjg(y[jy])
                    temp2 = dconjg(alpha*x[jx])
                    ap[kk] = dble(ap[kk]) + dble(x[jx]*temp1 + y[jy]*temp2)
                    ix = jx
                    iy = jy
                    for k in xrange(kk + 1, kk + n - j):
                        ix = ix + incx
                        iy = iy + incy
                        ap[k] = ap[k] + x[ix]*temp1 + y[iy]*temp2
                else:
                    ap[kk] = dble(ap[kk])
                jx = jx + incx
                jy = jy + incy
                kk = kk + n - j + 1
    
    return

chpr2 = zhpr2
__all__.append('chpr2'); __all__.append('zhpr2')

################################################################################

def zrotg(ca, cb, c, s):
    """
    Determines a double complex Givens rotation.
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    if abs(ca)==0.0:
        c = 0.0
        s = complex(1.0, 0.0)
        ca = cb
    else:
        scale = abs(ca) + abs(cb)
        norm = scale*((abs(ca/complex(scale, 0.0)))**2 + \
            (abs(cb/complex(scale, 0.0)))**2)**0.5
        alpha = ca/abs(ca)
        c = abs(ca)/norm
        s = alpha*dconjg(cb)/norm
        ca = alpha*norm
        
    return

crotg = zrotg
__all__.append('crotg'); __all__.append('zrotg')

################################################################################

def zscal(n, za, zx, incx):
    """
    Scales a vector by a constant.
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    if n<=0 or incx<=0:
        return
    if incx==1:
        # Code for increment equal to 1
        for i in xrange(n):
            zx[i] = za*zx[i]
    else:
        # Code for increment not equal to 1
        nincx = n*incx
        for i in xrange(0, nincx, incx):
            zx[i] = za*zx[i]
    
    return

cscal = zscal
__all__.append('cscal'); __all__.append('zscal')

################################################################################

def zswap(n, zx, incx, zy, incy):
    """
    Interchanges two vectors.
    
    Original Group
    --------------
    complex16_blas_level1
    
    """
    if n<=0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        for i in xrange(n):
            ztemp = zx[i]
            zx[i] = zy[i]
            zy[i] = temp
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 1
        iy = 1
        if incx<0:
            ix = (1 - n)*incx
        if incy<0:
            iy = (1 - n)*incy
        for i in xrange(n):
            ztemp = zx[ix]
            zx[ix] = zy[iy]
            zy[iy] = ztemp
            ix = ix + incx
            iy = iy + incy
    
    return

cswap = zswap
__all__.append('cswap'); __all__.append('zswap')

################################################################################

def zsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc):
    """
    Performs one of the matrix-matrix operations
    
        C := alpha*A*B + beta*C
    
    or
    
        C := alpha*B*A + beta*C,
    
    where alpha and beta are scalars, A is a symmetric matrix and B and C are
    m by n matrices.
    
    Parameters
    ----------
    side : str
        On entry, ``side`` specifies whether the symmetric matrix ``a`` appears
        on the lest or rith in the operation as follows:
        
        - ``side`` = 'L' or 'l'  C := alpha*A*B + beta*C
        - ``side`` = 'R' or 'r'  C := alpha*B*A + beta*C
    
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular
        part of the symmetric matrix ``a`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part
    
    m : int
        On entry, ``m`` specifies the number of rows of the matrix ``c``.
        ``m`` must be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of the matrix ``c``.
        ``n`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, ka)``, where ``ka``
        is ``m`` when ``side`` = 'L' or 'l' and ``n`` otherwise. Before entry
        with ``side`` = 'L' or 'l', the ``m`` by ``m`` part of the array ``a``
        must contain the symmetric matrix, such that when ``uplo`` = 'U' or 
        'u', the leading ``m`` by ``m`` upper triangular part of ``a`` must
        contain the upper triangular part of the symmetric matrix and the 
        strictly lower triangular part of ``a`` is not referenced, and when 
        ``uplo`` = 'L' or 'l', the leading ``m`` by ``m`` lower triangular 
        part of the array ``a`` must contain the lower triangular part of the 
        symmetric matrix and the strictly upper triangular part of ``a`` is
        not referenced.
        Before entry with ``side`` = 'R' or 'r', the ``n`` by ``n`` part of
        the array ``a`` must contain the symmetric matrix, such that when
        ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``a`` must contain the upper triangular part of the
        symmetric matrix and the strictly lower triangular part of ``a`` is
        not referenced, and when ``uplo`` = 'L' or 'l', the leading ``n`` by 
        ``n`` lower triangular part of the array ``a`` must contain the lower
        triangular part of the symmetric matrix and the strictly upper 
        triangular part of ``a`` is not referenced.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``side`` = 'L' or 'l' then ``lda``
        must be at least ``max([1, m])``, otherwise ``lda`` must be at least
        ``max([1, n])``.
    b : 2d-array
        ``b`` is a ``complex`` array of dimension ``(ldb, n)``. Before entry,
        the leading ``m`` by ``n`` part of the array ``b`` must contain the
        matrix B.
    ldb : int
        On entry, ``ldb`` specifies the first dimension as declared in the
        calling (sub) program. ``ldb`` must be at least ``max([1, m])``.
    beta : complex
        On entry, ``beta`` specifies the scalar beta. When ``beta`` is supplied
        as zero, then ``c`` need not be set on input.
    c : 2d-array
        ``c`` is a ``complex`` array of dimension ``(ldc, n)``. Before entry,
        the leading ``m`` by ``n`` part of the array ``c`` must contain the
        matrix C, except when ``beta`` is zero, in which case ``c`` need not
        be set on entry. On exit, the array ``c`` is overwritten by the ``m``
        by ``n`` updated matrix.
    ldc : int
        On entry, ``ldc`` specifies the first dimension of ``c`` as declared
        in the calling (sub) program. ``ldc`` must be at least ``max([1, m])``.
    
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Set NROWA as the number of rows of A
    if lsame(side, 'L'):
        nrowa = m
    else:
        nrowa = n
    upper = lsame(uplo, 'U')
    
    # Test the input parameters
    info = 0
    if not lsame(side, 'L') and not lsame(side, 'R'):
        info = 1
    elif not upper and not lsame(uplo, 'L'):
        info = 2
    elif m<0:
        info = 3
    elif n<0:
        info = 4
    elif lda<max([1, nrowa]):
        info = 7
    elif ldb<max([1, m]):
        info = 9
    elif ldc<max([1, m]):
        info = 12
    
    if info!=0:
        xerbla('zsymm', info)
        return
    
    # Quick return if possible
    if m==0 or n==0 or (alpha==zero and beta==one):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if beta==zero:
            for j in xrange(n):
                for i in xrange(m):
                    c[i][j] = zero
        else:
            for j in xrange(n):
                for i in xrange(m):
                    c[i][j] = beta*c[i]*c[j]
    
        return
    
    # Start the operations
    if lsame(side, 'L'):
        # Form C := alpha*A*B + beta*C
        if upper:
            for j in xrange(n):
                for i in xrange(m):
                    temp1 = alpha*b[i][j]
                    temp2 = zero
                    for k in xrange(i - 1):
                        c[k][j] = c[k][j] + temp1*a[k][i]
                        temp2 = temp2 + b[k][j]*a[k][i]
                    if beta==zero:
                        c[i][j] = temp1*a[i][i] + alpha*temp2
                    else:
                        c[i][j] = beta*c[i][j] + temp1*a[i][i] + alpha*temp2
        else:
            for j in xrange(n):
                for i in xrange(m - 1, -1, -1):
                    temp1 = alpha*b[i][j]
                    temp2 = zero
                    for k in xrange(i + 1, m):
                        c[k][j] = c[k][j] + temp1*a[k][i]
                        temp2 = temp2 + b[k][j]*a[k][i]
                    if beta==zero:
                        c[i][j] = temp1*a[i][i] + alpha*temp2
                    else:
                        c[i][j] = beta*c[i][j] + temp1*a[i][i] + alpha*temp2
    else:
        # Form C := alpha*B*A + beta*C
        for j in xrange(n):
            temp1 = alpha*a[j][j]
            if beta==zero:
                for i in xrange(m):
                    c[i][j] = temp1*b[i][j]
            else:
                for i in xrange(m):
                    c[i][j] = beta*c[i][j] + temp1*b[i][j]
            for k in xrange(j - 1):
                if upper:
                    temp1 = alpha*a[k][j]
                else:
                    temp1 = alpha*a[j][k]
                for i in xrange(m):
                    c[i][j] = c[i][j] + temp1*b[i][k]
            for k in xrange(j + 1, n):
                if upper:
                    temp1 = alpha*a[j][k]
                else:
                    temp1 = alpha*a[k][j]
                for i in xrange(m):
                    c[i][j] = c[i][j] + temp1*b[i][k]
                    
    return

csymm = zsymm
__all__.append('csymm'); __all__.append('zsymm')

################################################################################

def zsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    """
    Performs one of the symmetric rank 2k operations
    
        C := alpha*A*B**T + alpha*B*A**T + beta*C
    
    or
    
        C := alpha*A**T*B + alpha*B**T*A + beta*C,
    
    where alpha and beta are scalars, C is an n by n symmetric matrix and A and
    B are n by k matrices in the first case and k by n matrices in the second
    case.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular part
        of the array ``c`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part of ``c``
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part of ``c``
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  C := alpha*A*B**T + alpha*B*A**T + beta*C
        - ``trans`` = 'T' or 't'  C := alpha*A**T*B + alpha*B**T*A + beta*C
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``c``. ``n`` must
        be at least zero.
    k : int
        On entry with ``trans`` = 'N' or 'n', ``k`` specifies the number of
        columns of the matrices ``a`` and ``b``, and on entry with ``trans`` =
        'T' or 't', ``k`` specifies the number of rows of the matrices ``a``
        and ``b``. ``k`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, ka)``, where ``ka`` 
        is ``k`` when ``trans`` = 'N' or 'n', the leading ``n`` by ``k`` part
        of the array ``a`` must contain the matrix A, otherwise the leading
        ``k`` by ``n`` part of the array ``a`` must contain the matrix A.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``trans`` = 'N' or 'n', then ``lda``
        must be at least ``max([1, n])``, otherwise ``lda`` must be at least
        ``max([1, k])``.
    b : 2d-array
        ``b`` is a ``complex`` array of dimension ``(ldb, kb)``, where ``kb`` 
        is ``k`` when ``trans`` = 'N' or 'n', the leading ``n`` by ``k`` part
        of the array ``b`` must contain the matrix B, otherwise the leading
        ``k`` by ``n`` part of the array ``b`` must contain the matrix B.
    ldb : int
        On entry, ``lda`` specifies the first dimension of ``b`` as declared
        in the calling (sub) program. When ``trans`` = 'N' or 'n', then ``ldb``
        must be at least ``max([1, n])``, otherwise ``ldb`` must be at least
        ``max([1, k])``.
    beta : complex
        On entry, ``beta`` specifies the scalar beta.
    c : 2d-array
        ``c`` is a ``complex`` array of dimension ``(ldc, n)``. Before entry
        with ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``c`` must contain the upper triangular part of the
        symmetric matrix and the strictly lower triangular part of ``c`` is
        not referenced. On exit, the upper triangular part of the array ``c``
        is overwritten by the upper triangular part of the updated matrix.
        Before entry with ``uplo`` = 'L' or 'l', the leading ``n`` by ``n``
        lower triangular part of the array ``c`` must contain the lower
        triangular part of the symmetric matrix and the strictly upper
        triangular part of ``c`` is not referenced. On exit, the lower
        triangular part of the array ``c`` is overwritten by the lower 
        triangular part of the updated matrix.
    ldc : int
        On entry, ``ldc`` specifies the first dimension of ``c`` as declared in 
        the calling (sub) program. ``ldc`` must be at least ``max([1, n])``.
        
    Original Group
    --------------
    complex16_blas_level3
        
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    if lsame(trans, 'N'):
        nrowa = n
    else:
        nrowa = k
    
    upper = lsame(uplo, 'U')
    
    info = 0
    if not upper and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'T'):
        info = 2
    elif n<0:
        info = 3
    elif k<0:
        info = 4
    elif lda<max([1, nrowa]):
        info = 7
    elif ldb<max([1, nrowa]):
        info = 9
    elif ldc<max([1, n]):
        info = 12
    
    if info!=0:
        xerbla('zsyr2k', info)
        return
    
    # Quick return if possible
    if n==0 or ((alpha==zero or k==0) and beta==one):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if upper:
            if beta==zero:
                for j in xrange(n):
                    for i in xrange(j):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    for i in xrange(j):
                        c[i][j] = beta*c[i][j]
        else:
            if beta==zero:
                for j in xrange(n):
                    for i in xrange(j, n):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    for i in xrange(j, n):
                        c[i][j] = beta*c[i][j]
        return
    
    # Start the operations
    if lsame(trans, 'N'):
        # Form C := alpha*A*B**T + alpha*B*A**T + C.
        if upper:
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(j):
                        c[i][j] = zero
                else:
                    for i in xrange(j):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if a[j][l]!=zero or b[j][l]!=zero:
                        temp1 = alpha*b[j][l]
                        temp2 = alpha*a[j][l]
                        for i in xrange(j):
                            c[i][j] = c[i][j] + a[i][l]*temp1 + b[i][l]*temp2
        else:
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(j, n):
                        c[i][j] = zero
                else:
                    for i in xrange(j, n):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if a[j][l]!=zero or b[j][l]!=zero:
                        temp1 = alpha*b[j][l]
                        temp2 = alpha*a[j][l]
                        for i in xrange(j, n):
                            c[i][j] = c[i][j] + a[i][l]*temp1 + b[i][l]*temp2
    else:
        # Form  C := alpha*A**T*B + alpha*B**T*A + C
        if upper:
            for j in xrange(n):
                for i in xrange(j):
                    temp1 = zero
                    temp2 = zero
                    for l in xrange(k):
                        temp1 = temp1 + a[l][i]*b[l][j]
                        temp2 = temp2 + b[l][i]*a[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp1 + alpha*temp2
                    else:
                        c[i][j] = beta*c[i][j] + alpha*temp1 + alpha*temp2
        else:
            for j in xrange(n):
                for i in xrange(j, n):
                    temp1 = zero
                    temp2 = zero
                    for l in xrange(k):
                        temp1 = temp1 + a[l][i]*b[l][j]
                        temp2 = temp2 + b[l][i]*a[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp1 + alpha*temp2
                    else:
                        c[i][j] = beta*c[i][j] + alpha*temp1 + alpha*temp2
    
    return

csyr2k = zsyr2k
__all__.append('csyr2k'); __all__.append('zsyr2k')

################################################################################

def zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc):
    """
    Performs one of the symmetric rank k operations
    
        C := alpha*A*A**T + beta*C
    
    or
        
        C := alpha*A**T*A + beta*C,
    
    where alpha and beta are scalars, C is an n by n symmetric matrix and A is
    an n by k matrix in the first case and a k by n matrix in the second case.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the upper or lower triangular part
        of the array ``c`` is to be referenced as follows:
        
        - ``uplo`` = 'U' or 'u'  Only the upper triangular part of ``c``
        - ``uplo`` = 'L' or 'l'  Only the lower triangular part of ``c``
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  C := alpha*A*B**T + alpha*B*A**T + beta*C
        - ``trans`` = 'T' or 't'  C := alpha*A**T*B + alpha*B**T*A + beta*C
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``c``. ``n`` must
        be at least zero.
    k : int
        On entry with ``trans`` = 'N' or 'n', ``k`` specifies the number of
        columns of the matrices ``a`` and ``b``, and on entry with ``trans`` =
        'T' or 't', ``k`` specifies the number of rows of the matrices ``a``
        and ``b``. ``k`` must be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, ka)``, where ``ka`` 
        is ``k`` when ``trans`` = 'N' or 'n', the leading ``n`` by ``k`` part
        of the array ``a`` must contain the matrix A, otherwise the leading
        ``k`` by ``n`` part of the array ``a`` must contain the matrix A.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``trans`` = 'N' or 'n', then ``lda``
        must be at least ``max([1, n])``, otherwise ``lda`` must be at least
        ``max([1, k])``.
    beta : complex
        On entry, ``beta`` specifies the scalar beta.
    c : 2d-array
        ``c`` is a ``complex`` array of dimension ``(ldc, n)``. Before entry
        with ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper triangular
        part of the array ``c`` must contain the upper triangular part of the
        symmetric matrix and the strictly lower triangular part of ``c`` is
        not referenced. On exit, the upper triangular part of the array ``c``
        is overwritten by the upper triangular part of the updated matrix.
        Before entry with ``uplo`` = 'L' or 'l', the leading ``n`` by ``n``
        lower triangular part of the array ``c`` must contain the lower
        triangular part of the symmetric matrix and the strictly upper
        triangular part of ``c`` is not referenced. On exit, the lower
        triangular part of the array ``c`` is overwritten by the lower 
        triangular part of the updated matrix.
    ldc : int
        On entry, ``ldc`` specifies the first dimension of ``c`` as declared in 
        the calling (sub) program. ``ldc`` must be at least ``max([1, n])``.
        
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    if lsame(trans, 'N'):
        nrowa = n
    else:
        nrowa = k
    
    upper = lsame(uplo, 'U')
    
    info = 0
    if not upper and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'T'):
        info = 2
    elif n<0:
        info = 3
    elif k<0:
        info = 4
    elif lda<max([1, nrowa]):
        info = 7
    elif ldc<max([1, n]):
        info = 10
    
    if info!=0:
        xerbla('zsyrk', info)
        return
    
    # Quick return if possible
    if n==0 or ((alpha==zero or k==0) and beta==one):
        return
    
    # And when alpha==zero
    if alpha==zero:
        if upper:
            if beta==zero:
                for j in xrange(n):
                    for i in xrange(j):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    for i in xrange(j):
                        c[i][j] = beta*c[i][j]
        else:
            if beta==zero:
                for j in xrange(n):
                    for i in xrange(j, n):
                        c[i][j] = zero
            else:
                for j in xrange(n):
                    for i in xrange(j, n):
                        c[i][j] = beta*c[i][j]
        return
    
    # Start the operations
    if lsame(trans, 'N'):
        # Form C := alpha*A*A**T + beta*C
        if upper:
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(j):
                        c[i][j] = zero
                else:
                    for i in xrange(j):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if a[j][l]!=zero:
                        temp = alpha*a[j][l]
                        for i in xrange(j):
                            c[i][j] = c[i][j] + temp*a[i][l]
        else:
            for j in xrange(n):
                if beta==zero:
                    for i in xrange(j, n):
                        c[i][j] = zero
                else:
                    for i in xrange(j, n):
                        c[i][j] = beta*c[i][j]
                for l in xrange(k):
                    if a[j][l]!=zero:
                        temp = alpha*a[j][l]
                        for i in xrange(j, n):
                            c[i][j] = c[i][j] + temp*a[i][l]
    else:
        # Form  C := alpha*A**T*A + beta*C
        if upper:
            for j in xrange(n):
                for i in xrange(j):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + a[l][i]*a[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
        else:
            for j in xrange(n):
                for i in xrange(j, n):
                    temp = zero
                    for l in xrange(k):
                        temp = temp + a[l][i]*a[l][j]
                    if beta==zero:
                        c[i][j] = alpha*temp
                    else:
                        c[i][j] = alpha*temp + beta*c[i][j]
    
    return

csyrk = zsyrk
__all__.append('csyrk'); __all__.append('zsyrk')

################################################################################

def ztbmv(uplo, trans, diag, n, k, a, lda, x, incx):
    """
    Performs one of the matrix-vector operations
    
        x := A*x,  or  x := A**T*x,  or  x := A**H*x,
        
    where x is an n element vector and A is an n by n unit, or non-unit,
    upper or lower triangular band matrix, with (k + 1) diagonals.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the matrix is an upper or lower 
        triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is an lower triangular matrix
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  x := A*x
        - ``trans`` = 'T' or 't'  x := A**T*x
        - ``trans`` = 'C' or 'c'  x := A**H*x
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must
        be at least zero.
    k : int
        On entry with ``uplo`` = 'U' or 'u', ``k`` specifies the number of
        super-diagonals of the matrix ``a``. On entry with ``uplo`` = 'L' or
        'l', ``k`` specifies the number of sub-diagonals of the matrix ``a``.
        ``k`` must satisfy 0<=k.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo`` = 'U' or 'u', the leading ``(k + 1)`` by ``n`` part of
        the array ``a`` must contain the upper triangular band part of the
        matrix of coefficients, supplied column by column with the leading
        diagonal of the matrix in row ``(k + 1)`` of the array, the first 
        super-diagonal starting at position 1 in row ``k``, and so on. The top
        left ``k`` by ``k`` triangle of the array ``a`` is not referenced. The
        following program segment will transfer an upper triangular band
        matrix from conventional full matrix storage to band storage::
        
            for j in xrange(n):
                m = k - j
                for i in xrange(max([0, j - k - 1]), j):
                    a[m + i][j] = matrix[i][j]
        
        Before entry with ``uplo`` = 'L' or 'l', the leading ``(k + 1)`` by
        ``n`` part of the array ``a`` must contain the lower triangular band
        part of the matrix of coefficients, supplied column by column, with
        the leading diagonal of the matrix in row 0 of the array, the first
        sub-diagonal starting at position 0 of row 1, and so on. The bottom
        right ``k`` by ``k`` triangle of the array ``a`` is not referenced.
        The following program segment will transfer a lower triangular band
        matrix from conventional full matrix storage to band storage::
        
            for j in xrange(n):
                m = 1 - j
                for i in xrange(j, min([n, j + k])):
                    a[m + i][j] = matrix[i][j]
        
        Note that when ``diag`` = 'U' or 'u', the elements of the array ``a``
        corresponding to the diagonal elements of the matrix are not 
        referenced, but are assumed to be unity.
    lda: int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``(k + 1)``.
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x. One exit, ``x`` is overwritten
        with the transformed vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    
    Original Group
    --------------
    complex16_blas_level2

    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'T') \
        and not lsame(trans, 'C'):
        info = 2
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 3
    elif n<0:
        info = 4
    elif k<0:
        info = 5
    elif lda<(k + 1):
        info = 7
    elif incx==0:
        info = 9
    
    if info!=0:
        xerbla('ztbmv', info)
        return
    
    # Quick return if possible
    if n==0:
        return
    
    noconj = lsame(trans, 'T')
    nounit = lsame(diag, 'N')
    
    # Set up the start point in X if the increment is not unit. This will be
    # (N-1)*INCX too small for descending loops.
    if incx<=0:
        kx = (n - 1)*incx
    else:
        kx = 0
    
    # Start the operation. In this version, the elements of A are accessed
    # sequentially with one pass through A.
    
    if lsame(trans, 'N'):
        # Form  x := A*x.
        if lsame(uplo, 'U'):
            kplus1 = k + 1
            if incx==1:
                for j in xrange(n):
                    if x[j]!=zero:
                        temp = x[j]
                        l = kplus1 - j
                        for i in xrange(max([0, j - k - 1]), j - 1):
                            x[i] = x[i] + temp*a[l + i][j]
                        if nounit:
                            x[j] = x[j]*a[kplus1][j]
            else:
                jx = kx
                for j in xrange(n):
                    if x[jx]!=zero:
                        temp = x[jx]
                        ix = kx
                        l = kplus1 - j
                        for i in xrange(max([0, j - k - 1]), j - 1):
                            x[ix] = x[i] + temp*a[l + i][j]
                            ix = ix + incx
                        if nounit:
                            x[jx] = x[jx]*a[kplus1, j]
                    jx = jx + incx
                    if j>k:
                        kx = kx + incx
        else:
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    if x[j]!=zero:
                        temp = x[j]
                        l = 1 - j
                        for i in xrange(min([n - 1, j + k - 1]), j - 1, -1):
                            x[i] = x[i] + temp*a[l + i][j]
                        if nounit:
                            x[j] = x[j]*a[0][j]
            else:
                kx = kx + (n - 1)*incx
                jx = kx
                for j in xrange(n - 1, -1, -1):
                    if x[jx]!=zero:
                        temp = x[jx]
                        ix = kx
                        l = 1 - j
                        for i in xrange(min([n - 1, j + k - 1]), j - 1, -1):
                            x[ix] = x[ix] + temp*a[l + i][j]
                            ix = ix - incx
                        if nounit:
                            x[jx] = x[jx]*a[0][j]
                    jx = jx - incx
                    if (n - j)>=k:
                        kx = kx - incx
    else:
        # Form  x := A**T*x  or  x := A**H*x
        if lsame(uplo, 'U'):
            kplus1 = k + 1
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    temp = x[j]
                    l = kplus1 - j
                    if noconj:
                        if nounit:
                            temp = temp*a[kplus1][j]
                        for i in xrange(j - 2, max([-1, j - k - 2]), -1):
                            temp = temp + a[l + i][j]*x[i]
                    else:
                        if nounit:
                            temp = temp*dconjg(a[kplus1][j])
                        for i in xrange(j - 2, max([-1, j - k - 2]), -1):
                            temp = temp + dconjg(a[l + i][j])*x[i]
                    x[j] = temp
            else:
                kx = kx + (n - 1)*incx
                jx = kx
                for j in xrange(n - 1, -1, -1):
                    temp = x[jx]
                    kx = kx - incx
                    ix = kx
                    l = kplus1 - j
                    if noconj:
                        if nounit:
                            temp = temp*a[kplus1, j]
                        for i in xrange(j - 2, max([-1, j - k - 2]), -1):
                            temp = temp + a[l + i][j]*x[ix]
                            ix = ix - incx
                    else:
                        if nounit:
                            temp = dconjg(a[kplus1][j])
                        for i in xrange(j - 2, max([-1, j - k - 2]), -1):
                            temp = temp + dconjg(a[l + i][j])*x[ix]
                            ix = ix - incx
                    x[jx] = temp
                    jx = jx - incx
        else:
            if incx==1:
                for j in xrange(n):
                    temp = x[j]
                    l = 1 - j
                    if noconj:
                        if nounit:
                            temp = temp*a[0][j]
                        for i in xrange(j + 1, min([n - 1, j + k - 1])):
                            temp = temp + a[l + i][j]*x[i]
                    else:
                        if nounit:
                            temp = temp*dconjg(a[0][j])
                            for i in xrange(j + 1, min([n - 1, j + k - 1])):
                                temp = temp + dconjg(a[l + i][j])*x[i]
                    x[j] = temp
            else:
                jx = kx
                for j in xrange(n):
                    temp = x[jx]
                    kx = kx + incx
                    ix = kx
                    l = 1 - j
                    if noconj:
                        if nounit:
                            temp = temp*a[0][j]
                        for i in xrange(j + 1, min([n - 1, j + k - 1])):
                            temp = temp + a[l + i][j]*x[ix]
                            ix = ix + incx
                    else:
                        if nounit:
                            temp = temp*dconjg(a[0][j])
                        for i in xrange(j + 1, min([n - 1, j + k - 1])):
                            temp = temp + dconjg(a[l + i][j])*x[ix]
                            ix = ix + incx
                    x[jx] = temp
                    jx = jx + incx
                    
    return

ctbmv = ztbmv
__all__.append('ctbmv'); __all__.append('ztbmv')

################################################################################

def ztbsv(uplo, trans, diag, n, k, a, lda, x, incx):
    """
    Solves one of the systems of equations
    
        A*x = b,  or  A**T*x = b,  or  A**H*x = b,
        
    where b and x are n element vectors and A is an n by n unit, or non-unit,
    upper or lower triangular band matrix with (k + 1) diagonals.
    
    No test for singularity or near-singularity is included in this routine.
    Such tests must be performed before calling this routine.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the matrix is an upper or lower 
        triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is an lower triangular matrix
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  x := A*x
        - ``trans`` = 'T' or 't'  x := A**T*x
        - ``trans`` = 'C' or 'c'  x := A**H*x
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must
        be at least zero.
    k : int
        On entry with ``uplo`` = 'U' or 'u', ``k`` specifies the number of
        super-diagonals of the matrix ``a``. On entry with ``uplo`` = 'L' or
        'l', ``k`` specifies the number of sub-diagonals of the matrix ``a``.
        ``k`` must satisfy 0<=k.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo`` = 'U' or 'u', the leading ``(k + 1)`` by ``n`` part of
        the array ``a`` must contain the upper triangular band part of the
        matrix of coefficients, supplied column by column with the leading
        diagonal of the matrix in row ``(k + 1)`` of the array, the first 
        super-diagonal starting at position 1 in row ``k``, and so on. The top
        left ``k`` by ``k`` triangle of the array ``a`` is not referenced. The
        following program segment will transfer an upper triangular band
        matrix from conventional full matrix storage to band storage::
        
            for j in xrange(n):
                m = k - j
                for i in xrange(max([0, j - k - 1]), j):
                    a[m + i][j] = matrix[i][j]
        
        Before entry with ``uplo`` = 'L' or 'l', the leading ``(k + 1)`` by
        ``n`` part of the array ``a`` must contain the lower triangular band
        part of the matrix of coefficients, supplied column by column, with
        the leading diagonal of the matrix in row 0 of the array, the first
        sub-diagonal starting at position 0 of row 1, and so on. The bottom
        right ``k`` by ``k`` triangle of the array ``a`` is not referenced.
        The following program segment will transfer a lower triangular band
        matrix from conventional full matrix storage to band storage::
        
            for j in xrange(n):
                m = 1 - j
                for i in xrange(j, min([n, j + k])):
                    a[m + i][j] = matrix[i][j]
        
        Note that when ``diag`` = 'U' or 'u', the elements of the array ``a``
        corresponding to the diagonal elements of the matrix are not 
        referenced, but are assumed to be unity.
    lda: int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``(k + 1)``.
    x : array
        ``x`` is a ``complex`` array of dimension at least 
        ``(1 + (n - 1)*abs(incx))``. Before entry, the incremented array ``x``
        must contain the ``n`` element right-hand side vector b. One exit, 
        ``x`` is overwritten with the solution vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
    
    Original Group
    --------------
    complex16_blas_level2

    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'T') \
        and not lsame(trans, 'C'):
        info = 2
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 3
    elif n<0:
        info = 4
    elif k<0:
        info = 5
    elif lda<(k + 1):
        info = 7
    elif incx==0:
        info = 9
    
    if info!=0:
        xerbla('ztbsv', info)
        return
    
    # Quick return if possible
    if n==0:
        return
    
    noconj = lsame(trans, 'T')
    nounit = lsame(diag, 'N')
    
    # Set up the start point in X if the increment is not unit. This will be
    # (N-1)*INCX too small for descending loops.
    if incx<=0:
        kx = (n - 1)*incx
    else:
        kx = 0
    
    # Start the operation. In this version, the elements of A are accessed
    # sequentially with one pass through A.
    if lsame(trans, 'N'):
        # Form  x := inv(A)*x
        if lsame(uplo, 'U'):
            kplus1 = k + 1
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    if x[j]!=zero:
                        l = kplus1 - j
                        if nounit:
                            x[j] = x[j]/a[kplus1][j]
                        temp = x[j]
                        for i in xrange(j - 2, max([-1, j - k - 2]), -1):
                            x[i] = x[i] - temp*a[l + i][j]
                    else:
                        kx = kx + (n - 1)*incx
                        jx = kx
                        for j in xrange(j - 2, max([-1, j - k - 2]), -1):
                            kx = kx - incx
                            if x[jx]!=zero:
                                ix = kx
                                l = kplus1 - j
                                if nounit:
                                    x[jx] = x[jx]/a[kplus1][j]
                                temp = x[jx]
                                for i in xrange(j - 2, max([-1, j - k - 2]), -1):
                                    x[ix] = x[ix] - temp*a[l + i][j]
                                    ix = ix - incx
                            jx = jx - incx
        else:
            if incx==1:
                for j in xrange(n):
                    if x[j]!=zero:
                        l = 1 - j
                        if nounit:
                            x[j] = x[j]/a[0][j]
                        temp = x[j]
                        for i in xrange(j + 1, min([n, j + k])):
                            x[i] = x[i] - temp*a[l + i][j]
            else:
                jx = kx
                for j in xrange(n):
                    kx = kx + incx
                    if x[jx]!=zero:
                        ix = kx
                        l = 1 - j
                        if nounit:
                            x[jx] = x[jx]/a[0][j]
                        temp = x[jx]
                        for i in xrange(j + 1, min([n, j + k])):
                            x[ix] = x[ix] - temp*a[l + i][j]
                            ix = ix + incx
                    jx = jx + incx
    else:
        # Form  x := inv(A**T)*x  or x := inv(A**H)*x
        if lsame(uplo, 'U'):
            kplus1 = k + 1
            if incx==1:
                for j in xrange(n):
                    temp = x[j]
                    l = kplus1 - j
                    if noconj:
                        for i in xrange(max([0, j - k - 1]), j - 1):
                            temp = temp - a[l + i][j]*x[i]
                        if nounit:
                            temp = temp/a[kplus1][j]
                    else:
                        for i in xrange(max([0, j - k - 1]), j - 1):
                            temp = temp - dconjg(a[l + i][j])*x[i]
                        if nounit:
                            temp = temp/dconjg(a[kplus1][j])
                    x[j] = temp
            else:
                jx = kx
                for j in xrange(n):
                    temp = x[jx]
                    ix = kx
                    l = kplus1 - j
                    if noconj:
                        for i in xrange(max([0, j - k - 1]), j - 1):
                            temp = temp - a[l + i][j]*x[ix]
                            ix = ix + incx
                        if nounit:
                            temp = temp/a[kplus1][j]
                    else:
                        for i in xrange(max([0, j - k - 1]), j - 1):
                            temp = temp - dconjg(a[l + i][j])*x[ix]
                            ix = ix + incx
                        if nounit:
                            temp = temp/dconjg(a[kplus1][j])
                    x[jx] = temp
                    jx = jx + incx
                    if j>k:
                        kx = kx + incx
        else:
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    temp = x[j]
                    l = 1 - j
                    if noconj:
                        for i in xrange(min([n - 1, j + k - 1]), j - 1, -1):
                            temp = temp - a[l + i][j]*x[i]
                        if nounit:
                            temp = temp/a[0][j]
                    else:
                        for i in xrange(min([n - 1, j + k - 1]), j - 1, -1):
                            temp = temp - dconjg(a[l + i][j])*x[ix]
                            ix = ix - incx
                        if nounit:
                            temp = temp/dconjg(a[0][j])
                    x[jx] = temp
                    jx = jx - incx
                    if (n - j)>=k:
                        kx = kx - incx
    
    return

ctbsv = ztbsv
__all__.append('ctbsv'); __all__.append('ztbsv')

################################################################################

def ztpmv(uplo, trans, diag, n, ap, x, incx):
    """
    Performs one of the matrix-vector operations
    
        x := A*x,  or  x := A**T*x,  or x := A**H*x,
        
    where x ix an n element vector and A is an n by n unit, or non-unit,
    upper or lower triangular matrix, supplied in packed form.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the matrix is an upper or lower 
        triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is a lower triangular matrix
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  x := A*x
        - ``trans`` = 'T' or 't'  x := A**T*x
        - ``trans`` = 'C' or 'c'  x := A**H*x
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must
        be at least zero.
    ap : array
        ``ap`` is a ``complex`` array of dimension at least ``(n*(n + 1))/2``.
        Before entry with ``uplo`` = 'U' or 'u', the array ``ap`` must contain
        the upper triangular matrix packed sequentially, column by column, so
        that ``ap[0]`` contains ``a[0][0]``, ``ap[1]`` and ``a[2]`` contain
        ``a[0][1]`` and ``a[0][2]`` respectively, and so on. Before entry with
        ``uplo`` = 'L' or 'l', the array ``ap`` must contain the lower 
        triangular marix packed sequentially, column by column, so that 
        ``ap[0]`` contains ``a[0][0]``, ``ap[1]`` and ``ap[2]`` contain
        ``a[0][1]`` and ``a[0][2]`` respectively, and so on. Note that when
        ``diag`` = 'U' or 'u', the diagonal elements of ``a`` are not 
        referenced, but are assumed to be unity.
    x : array
        ``x`` is a ``complex`` array of dimension at least
        ``(1 + (n - 1)*abs(incx)``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x. On exit, ``x`` is overwritten
        with the transformed vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
        
    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if not lsame(uplo, 'U') and not lsame(uplo, 'L'):
        info = 1
    elif not lsame(trans, 'N') and not lsame(trans, 'T') and \
        not lsame(trans, 'C'):
        info = 2
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 3
    elif n<0:
        info = 4
    elif incx==0:
        info = 7
    
    if info!=0:
        xerbla('ztpmv', info)
        return
     
    # Quick return if possible.
    if n==0:
        return
    
    noconj = lsame(trans, 'T')
    nounit = lsame(diag, 'U')
    
    # Set up the start point in X if the increment is not unity. This will be
    # (N - 1)*INCX too small for descending loops.
    if incx<=0:
        kx = (n - 1)*incx
    elif incx!=1:
        kx = 0
    
    # Start the operations. In this version the elements of AP are accessed
    # sequentially with one pass through AP.
    if lsame(trans, 'N'):
        # Form  x := A*x.
        if lsame(uplo, 'U'):
            kk = 0
            if incx==1:
                for j in xrange(n):
                    if x[j]!=zero:
                        temp = x[j]
                        k = kk
                        for i in xrange(j - 1):
                            x[i] = x[i] + temp*ap[k]
                            k = k + 1
                    kk = kk + j
            else:
                jx = kx
                for j in xrange(n):
                    if x[jx]!=zero:
                        temp = x[jx]
                        ix = kx
                        for k in xrange(kk, kk + j - 2):
                            x[i] = x[i] + temp*ap[k]
                            ix = ix + incx
                        if nounit:
                            x[j] = x[j]*ap[kk + j - 1]
                    jx = jx + incx
                    kk = kk + j
        else:
            kk = (n*(n + 1))/2
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    if x[j]!=zero:
                        temp = x[j]
                        k = kk
                        for i in xrange(n - 1, j - 1, -1):
                            x[i] = x[i] + temp*ap[k]
                            k = k - 1
                        if nounit:
                            x[j] = x[j]*ap[kk - n + j]
                    kk = kk - (n - j + 1)
            else:
                kx = kx + (n - 1)*incx
                jx = kx
                for j in xrange(n - 1, -1, -1):
                    if x[jx]!=zero:
                        temp = x[jx]
                        ix = kx
                        for k in xrange(kk - 1, kk - (n - (j + 1)) - 2, -1):
                            x[ix] = x[ix] + temp*ap[k]
                            ix = ix - incx
                        if nounit:
                            x[jx] = x[jx]*ap[kk - n - j]
                    jx = jx - incx
                    kk = kk - (n - j + 1)
    else:
        # Form  x := A**T*x  or  x := A**H*x.
        if lsame(uplo, 'U'):
            kk = (n*(n + 1))/2
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    temp = x[j]
                    k = kk - 1
                    if noconj:
                        if nounit:
                            temp = temp*ap[kk]
                        for i in xrange(j - 2, -1, -1):
                            temp = temp + ap[k]*x[i]
                            k = k - 1
                    else:
                        if nounit:
                            temp = temp*dconjg(ap[kk])
                        for i in xrange(j - 2, -1, -1):
                            temp = temp + dconjg(ap[k])*x[i]
                            k = k - 1
                    x[j] = temp
                    kk = kk - j
            else:
                jx = kx + (n - 1)*incx
                for j in xrange(n - 1, -1, -1):
                    temp = x[jx]
                    ix = jx
                    if noconj:
                        if nounit:
                            temp = temp*ap[kk]
                        for k in xrange(kk - 2, kk - j - 1, -1):
                            ix = ix - incx
                            temp = temp + ap[k]*x[ix]
                    else:
                        if nounit:
                            temp = temp*dconjg(ap[kk])
                        for k in xrange(kk - 2, kk - j - 1, -1):
                            ix = ix - incx
                            temp = temp + dconjg(ap[k])*x[ix]
                    x[jx] = temp
                    jx = jx - incx
                    kk = kk - j
        else:
            kk = 0
            if incx==1:
                for j in xrange(n):
                    temp = x[j]
                    k = kk + 1
                    if noconj:
                        if nounit:
                            temp = temp*ap[kk]
                        for i in xrange(j + 1, n):
                            temp = temp + ap[k]*x[i]
                            k = k + 1
                    else:
                        if nounit:
                            temp = temp*dconjg(ap[kk])
                        for i in xrange(j + 1, n):
                            temp = temp + dconjg(ap[k])*x[i]
                            k = k + 1
                    x[j] = temp
                    kk = kk + (n - j + 1)
            else:
                jx = kx
                for j in xrange(n):
                    temp = x[jx]
                    ix = jx
                    if noconj:
                        if nounit:
                            temp = temp*ap[kk]
                        for k in xrange(kk + 1, kk + n - j):
                            ix = ix + incx
                            temp = temp + ap[k]*x[ix]
                    else:
                        if nounit:
                            temp = temp*dconjg(ap[kk])
                        for k in xrange(kk + 1, kk + n - j):
                            ix = ix + incx
                            temp = temp + dconjg(ap[k])*x[ix]
                    x[jx] = temp
                    jx = jx + incx
                    kk = kk + (n - j + 1)
    
    return

ctpmv = ztpmv
__all__.append('ctpmv'); __all__.append('ztpmv')

################################################################################

def ztrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb):
    """
    Performs one of the matrix-matrix operations
    
        B := alpha*op(A)*B,  or  B := alpha*B*op(A),
    
    where alpha is a scalar, B is an m by n matrix, A is a unit, or non-unit,
    upper or lower triangular matrix and op(A) is one of
    
        op(A) = A  or  op(A) = A**T  or  op(A) = A**H.
    
    Parameters
    ----------
    side : str
        On entry, ``side`` specifies whether ``op(a)`` multiplies ``b`` from
        the left side or right as follows:
        
        - ``side`` = 'L' or 'l'  B := alpha*op(A)*B.
        - ``side`` = 'R' or 'r'  B := alpha*B*op(A).

    uplo : str
        On entry, ``uplo`` specifies whether the matrix ``a`` is an upper or 
        lower triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is a lower triangular matrix
    
    transa : str
        On entry, ``transa`` specifies the operation to be performed as follows:
        
        - ``transa`` = 'N' or 'n'  op(A) := A
        - ``transa`` = 'T' or 't'  op(A) := A**T
        - ``transa`` = 'C' or 'c'  op(A) := A**H
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
    
    m : int
        On entry, ``m`` specifies the number of rows of ``b``. ``m`` must
        be at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of ``b``. ``n`` must
        be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha. When ``alpha`` is zero,
        then ``a`` is not referenced and ``b`` need not be set before entry.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, k`)``, where ``k``
        is ``m`` when ``side`` = 'L' or 'l' and is ``n`` when ``side`` = 'R'
        or 'r'. Before entry with ``uplo`` = 'U' or 'u', the leading ``k`` by
        ``k`` upper triangular part of the array ``a`` must contain the upper
        triangular matrix and the strictly lower triangular part of ``a`` is
        not referenced. Before entry with ``uplo`` = 'L' or 'l', the leading
        ``k`` by ``k`` lower triangular part of the array ``a`` must contain 
        the lower triangular matrix and the strictly upper triangular part of
        ``a`` is not referenced. Note that when ``diag`` = 'U' or 'u', the 
        diagonal elements of ``a`` are not referenced either, but are assumed
        to be unity.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``side`` = 'L' or 'l' then ``lda``
        must be at least ``max([1, m])``, when ``side`` = 'R' or 'r' then 
        ``lda`` must be at least ``max([1, n])
    b : 2d-array
        ``b`` is a ``complex`` array of dimension ``(ldb, n)``. Before entry,
        the leading ``m`` by ``n`` part of the array ``b`` must contain the
        matrix ``b``, and on exit is written by the transformed matrix.
    ldb : int
        On entry, ``ldb`` specifies the first dimension of ``b`` as declared
        in the calling (sum) program. ``ldb`` must be at least ``max([1, m])``.
    
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    lside = lsame(side, 'L')
    if lside:
        nrowa = m
    else:
        nrowa, n
    noconj = lsame(transa, 'T')
    nounit = lsame(diag, 'N')
    upper = lsame(uplo, 'U')
    
    info = 0
    if not lside and not lsame(side, 'R'):
        info = 1
    elif not upper and not lsame(uplo, 'L'):
        info = 2
    elif not lsame(transa, 'N') and not lsame(transa, 'T') and \
        not lsame(transa, 'C'):
        info = 3
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 4
    elif m< 0:
        info = 5
    elif n<0:
        info = 6
    elif lda<max([1, nrowa]):
        info = 9
    elif ldb<max([1, m]):
        info = 11
    
    if info!=0:
        xerbla('ztrmm', info)
        return
    
    # Quick return if possible.
    if m==0 or n==0:
        return
    
    # And when alpha==zero
    if alpha==zero:
        for j in xrange(n):
            for i in xrange(m):
                b[i][j] = zero
        return
    
    # Start the operations
    if lside:
        if lsame(transa, 'N'):
            # Form  B := alpha*A*B
            if upper:
                for j in xrange(n):
                    for k in xrange(m):
                        if b[k][j]!=zero:
                            temp = alpha*b[k][j]
                            for i in xrange(k - 1):
                                b[i][j] = b[i][j] + temp*a[i][k]
                            if nounit:
                                temp = temp*a[k][k]
                            b[k][j] = temp
            else:
                for j in xrange(n):
                    for k in xrange(m - 1, -1, -1):
                        if b[k][j]!=zero:
                            temp = alpha*b[k][j]
                            b[k][j] = temp
                            if nounit:
                                b[k][j] = b[k][j]*a[k][k]
                            for i in xrange(k + 1, m):
                                b[i][j] = b[i][j] + temp*a[i][k]
        else:
            # Form  b := alpha*A**T*B  or  B := alpha*A**H*B.
            if upper:
                for j in xrange(n):
                    for i in xrange(m - 1, -1, -1):
                        temp = b[i][j]
                        if noconj:
                            if nounit:
                                temp = temp*a[i][i]
                            for k in xrange(i - 1):
                                temp = temp + a[k][i]*b[k][j]
                        else:
                            if nounit:
                                temp = temp*dconjg(a[i][i])
                            for k in xrange(i - 1):
                                temp = temp + dconjg(a[k][i])*b[k][j]
                        b[i][j] = alpha*temp
            else:
                for j in xrange(n):
                    for i in xrange(m):
                        temp = b[i][j]
                        if noconj:
                            if nounit:
                                temp = temp*a[i][i]
                            for k in xrange(i + 1, m):
                                temp = temp + a[k][i]*b[k][j]
                        else:
                            if nounit:
                                temp = temp*dconjg(a[i][i])
                            for k in xrange(i + 1, m):
                                temp = temp + dconjg(a[k][i])*b[k][j]
                        b[i][j] = alpha*temp
    else:
        if lsame(trans, 'N'):
            # Form  B := alpha*B*A
            if upper:
                for j in xrange(n - 1, -1, -1):
                    temp = alpha
                    if nounit:
                        temp = temp*a[j][j]
                    for i in xrange(m):
                        b[i][j] = temp*b[i][j]
                    for k in xrange(j - 1):
                        if a[k][j]!=zero:
                            temp = alpha*a[k][j]
                            for i in xrange(m):
                                b[i][j] = b[i][j] + temp*b[i][k]
            else:
                for j in xrange(n):
                    temp = alpha
                    if nounit:
                        temp = temp*a[j][j]
                    for i in xrange(m):
                        b[i][j] = temp*b[i][j]
                    for k in xrange(j + 1, n):
                        if a[k][j]!=zero:
                            temp = alpha*a[k][j]
                            for i in xrange(m):
                                b[i][j] = b[i][j] + temp*b[i][k]
        else:
            # Form  B := alpha*B*A**T  or  B := alpha*B*A**H
            if upper:
                for k in xrange(n):
                    for j in xrange(k - 1):
                        if a[j][k]!=zero:
                            if noconj:
                                temp = alpha*a[j][k]
                            else:
                                temp = alpha*dconjg(a[j][k])
                            for i in xrange(m):
                                b[i][j] = b[i][j] + temp*b[i][k]
                    temp = alpha
                    if nounit:
                        if noconj:
                            temp = temp*a[k][k]
                        else:
                            temp = temp*dconjg(a[k][k])
                    if temp!=one:
                        for i in xrange(m):
                            b[i][k] = temp*b[i][k]
            else:
                for k in xrange(n - 1, -1, -1):
                    for j in xrange(k + 1, n):
                        if a[j][k]!=zero:
                            if noconj:
                                temp = alpha*a[j][k]
                            else:
                                temp = alpha*dconjg(a[j][k])
                            for i in xrange(m):
                                b[i][j] = b[i][j] + temp*b[i][k]
                    temp = alpha
                    if nounit:
                        if noconj:
                            temp = temp*a[k][k]
                        else:
                            temp = temp*dconjg(a[k][k])
                    if temp!=one:
                        for i in xrange(m):
                            b[i][k] = temp*b[i][k]
    
    return

ctrmm = ztrmm
__all__.append('ctrmm'); __all__.append('ztrmm')

################################################################################

def ztrmv(uplo, trans, diag, n, a, lda, x, incx):
    """
    Performs one of the matrix-vector operations
    
        x := A*x,  or  x := A**T*x,  or  x := A**H*x,
    
    where x is an n element vector and A is an n by n unit, or non-unit,
    upper or lower triangular matrix.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the matrix is an upper or lower 
        triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is a lower triangular matrix
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  x := A*x
        - ``trans`` = 'T' or 't'  x := A**T*x
        - ``trans`` = 'C' or 'c'  x := A**H*x
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must
        be at least zero.
    a : array
        ``a`` is a ``complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper
        triangular part of the array ``a`` must contain the upper triangular
        matrix and the strictly lower triangular part of ``a`` is not 
        referenced. Before engty with ``uplo`` = 'L' or 'l', the leading ``n``
        by ``n`` lower triangular part of the array ``a`` must contain the
        lower triangular matrix and the strictly upper triangular part of
        ``a`` is not referenced. Note that when ``diag`` = 'U' or 'u', the
        diagonal elements of ``a`` are not referenced either, but are assumed
        to be unity.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``max([1, n])``.
    x : array
        ``x`` is a ``complex`` array of dimension at least
        ``(1 + (n - 1)*abs(incx)``. Before entry, the incremented array ``x``
        must contain the ``n`` element vector x. On exit, ``x`` is overwritten
        with the transformed vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
        
    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if lsame(uplo, 'U') and not lsame(uplo, 'l'):
        info = 1
    elif lsame(trans, 'N') and not lsame(trans, 'T') and not lsame(trans, 'C'):
        info = 2
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 3
    elif n<0:
        info = 4
    elif lda<max([1, n]):
        info = 6
    elif incx==0:
        info = 8
    
    if info!=0:
        xerbla('ztrmv', info)
        return
    
    # Quick return if possible
    if n==0:
        return
    
    noconj = lsame(trans, 'T')
    nounit = lsame(diag, 'N')
    
    # Set up the start point in X if the increment is not unity. This will be
    # (N-1)*INCX too small for descending loops
    if incx<=0:
        kx = (n - 1)*incx
    elif incx!=1:
        kx = 0
    
    # Start the operation. In this version, the elements of A are accessed
    # sequentially with one pass through A.
    if lsame(trans, 'N'):
        # Form  x := A*x.
        if lsame(uplo, 'U'):
            if incx==1:
                for j in xrange(n):
                    if x[j]!=zero:
                        temp = x[j]
                        for i in xrange(j - 1):
                            x[i] = x[i] + temp*a[i][j]
                        if nounit:
                            x[j] = x[j]*a[j][j]
            else:
                jx = kx
                for j in xrange(n):
                    if x[jx]!=zero:
                        temp = x[jx]
                        ix = kx
                        for i in xrange(j - 1):
                            x[ix] = x[ix] + temp*a[i][j]
                            ix = ix + incx
                        if nounit:
                            x[jx] = x[jx]*a[j][j]
                    jx = jx + incx
        else:
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    if x[j]!=zero:
                        temp = x[j]
                        for i in xrange(n - 1, j - 1, -1):
                            x[i] = x[i] + temp*a[i][j]
                        if nounit:
                            x[j] = x[j]*a[j][j]
            else:
                kx = kx + (n - 1)*incx
                jx = kx
                for j in xrange(n - 1, -1, -1):
                    if x[jx]!=zero:
                        temp = x[jx]
                        ix = kx
                        for i in xrange(n - 1, j - 1, -1):
                            x[ix] = x[ix] + temp*a[i][j]
                            ix = ix - incx
                        if nounit:
                            x[jx] = x[jx]*a[j][j]
                    jx = jx - incx
    else:
        # Form  x := A**T*x  or  x := A**H*x.
        if lsame(uplo, 'U'):
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    temp = x[j]
                    if noconj:
                        if nounit:
                            temp = temp*a[j][j]
                        for i in xrange(j - 1, -1, -1):
                            temp = temp + a[i][j]*x[i]
                    else:
                        if nounit:
                            temp = temp*dconjg(a[j][j])
                        for i in xrange(j - 1, -1, -1):
                            temp = temp + dconjg(a[i][j])*x[i]
                    x[j] = temp
            else:
                jx = kx + (n - 1)*incx
                for j in xrange(n - 1, -1, -1):
                    temp = x[jx]
                    ix = jx
                    if noconj:
                        if nounit:
                            temp = temp*a[j][j]
                        for i in xrange(j - 1, -1, -1):
                            ix = ix - incx
                            temp = temp + a[i][j]*x[ix]
                    else:
                        if nounit:
                            temp = temp*dconjg(a[j][j])
                        for i in xrange(j - 1, -1, -1):
                            ix = ix - incx
                            temp = temp + dconjg(a[i][j])*x[ix]
                    x[jx] = temp
                    jx = jx - incx
        else:
            if incx==1:
                for j in xrange(n):
                    temp = x[j]
                    if noconj:
                        if nounit:
                            temp = temp*a[j][j]
                        for i in xrange(j + 1, n):
                            temp = temp + a[i][j]*x[i]
                    else:
                        if nounit:
                            temp = temp*dconjg(a[j][j])
                        for i in xrange(j + 1, n):
                            temp = temp + dconjg(a[i][j])*x[i]
                    x[j] = temp
            else:
                jx = kx
                for j in xrange(n):
                    temp = x[jx]
                    ix = jx
                    if noconj:
                        if nounit:
                            temp = temp*a[j][j]
                        for i in xrange(j + 1, n):
                            ix = ix + incx
                            temp = temp + a[i][j]*x[ix]
                    else:
                        if nounit:
                            temp = temp*dconjg(a[j][j])
                        for i in xrange(j + 1, n):
                            ix = ix + incx
                            temp = temp + dconjg(a[i][j])*x[ix]
                    x[jx] = temp
                    jx = jx + incx
    
    return

ctrmv = ztrmv
__all__.append('ctrmv'); __all__.append('ztrmv')

################################################################################

def ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb):
    """
    Solves one of the matrix equations
    
        op(A)*X = alpha*B  or  X*op(A) = alpha*B,
    
    where alpha is a scalar, X and B are m by n matrices, A is a unit, or 
    non-unit, upper or lower triangular matrix and op(A) is one of
    
        op(A) = A  or  op(A) = A**T  or  op(A) = A**H.
    
    The matrix X is overwritten on B.
    
    Parameters
    ----------
    side : str
        On entry, ``side`` specifies wither op(A) appears on the left or right
        side of ``x`` as follows:
        
        - ``side`` = 'L' or 'l'  op(A)*X = alpha*B.
        - ``side`` = 'R' or 'r'  X*op(A) = alpha*B.

    uplo : str
        On entry, ``uplo`` specifies whether the matrix is an upper or lower 
        triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is a lower triangular matrix
    
    transa : str
        On entry, ``transa`` specifies the operation to be performed as follows:
        
        - ``transa`` = 'N' or 'n'  op(A) = A
        - ``transa`` = 'T' or 't'  op(A) = A**T
        - ``transa`` = 'C' or 'c'  op(A) = A**H
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
        
    m : int
        On entry, ``m`` specifies the number of rows of ``b``. ``m`` must be
        at least zero.
    n : int
        On entry, ``n`` specifies the number of columns of ``b``. ``n`` must
        be at least zero.
    alpha : complex
        On entry, ``alpha`` specifies the scalar alpha. When ``alpha`` is zero
        then ``a`` is not referenced and ``b`` need not be set before entry.
    a : 2d-array
        ``a`` is a ``complex`` array of dimension ``(lda, k)``, where ``k`` is
        ``m`` when ``side`` = 'L' or 'l' and ``k`` is ``n`` when ``side`` =
        'R' or 'r'. Before entry with ``uplo`` = 'U' or 'u', the leading
        ``k`` by ``k`` upper triangular part of the array ``a`` must contain
        the upper triangular matrix and the strictly lower triangular part of
        ``a`` is not referenced. Before entry with ``uplo`` = 'L' or 'l', the
        leading ``k`` by ``k`` lower triangular part of the array ``a`` must
        contain the lower triangular matrix and the strictly upper triangular
        part of ``a`` is not referenced. Note that when ``diag`` = 'U' or 'u',
        the diagonal elements of `` are not referenced either, but are assumed
        to be unity.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. When ``side`` = 'L' or 'l', then ``lda``
        must be at least ``max([1, m])``, when ``side`` = 'R' or 'r' then
        ``lda`` must be at least ``max([1, n])``.
    b : 2d-array
        ``b`` is a ``complex`` array of dimension ``(ldb, n)``. Before entry,
        the leading ``m`` by ``n`` part of the array ``b`` must contain the
        right-hand side matrix B, and on exit is overwritten by the solution
        matrix ``x``.
    ldb : int
        On entry, ``ldb`` specifies the first dimension of ``b`` as declared
        in the calling (sub) program. ``ldb`` must be at least ``max([1, m])``.
    
    Original Group
    --------------
    complex16_blas_level3
    
    """
    one = complex(1.0, 0.0)
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    lside = lsame(side, 'L')
    if lside:
        nrowa = m
    else:
        nrowa = n
    noconj = lsame(transa, 'T')
    nounit = lsame(diag, 'N')
    upper = lsame(uplo, 'U')
    
    info = 0
    if not lside and not lsame(side, 'R'):
        info = 1
    elif not upper and not lsame(uplo, 'L'):
        info = 2
    elif not lsame(transa, 'N') and not lsame(transa, 'T') and \
        not lsame(transa, 'C'):
        info = 3
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 4
    elif m<0:
        info = 5
    elif n<0:
        info = 6
    elif lda<max([1, nrowa]):
        info = 9
    elif ldb<max([1, m]):
        info = 11
    
    if info!=0:
        xerbla('ztrsm', info)
        return
    
    # Quick return if possible.
    if m==0 or n==0:
        return
    
    # And when alpha==zero
    if alpha==zero:
        for j in xrange(n):
            for i in xrange(m):
                b[i][j] = zero
        return
    
    # Start the operations.
    if lside:
        if lsame(transa, 'N'):
            # Form  B := alpha*inv(A)*B.
            if upper:
                for j in xrange(n):
                    if alpha!=one:
                        for i in xrange(m):
                            b[i][j] = alpha*b[i][j]
                    for k in xrange(m - 1, -1, -1):
                        if b[k][j]!=zero:
                            if nounit:
                                b[k][j] = b[k][j]/a[k][k]
                            for i in xrange(k - 1):
                                b[i][j] = b[i][j] - b[k][j]*a[i][k]
            else:
                for j in xrange(n):
                    if alpha!=one:
                        for i in xrange(m):
                            b[i][j] = alpha*b[i][j]
                    for k in xrange(m):
                        if b[k][j]!=zero:
                            if nounit:
                                b[k][j] = b[k][j]/a[k][k]
                            for i in xrange(k + 1, m):
                                b[i][j] = b[i][j] - b[k][j]*a[i][k]
        else:
            # Form  B := alpha*inv(A**T)*B  or  B := alpha*inv(A**H)*B
            if upper:
                for j in xrange(n):
                    for i in xrange(m):
                        temp = alpha*b[i][j]
                        if noconj:
                            for k in xrange(i - 1):
                                temp = temp - a[k][i]*b[k][j]
                            if nounit:
                                temp = temp/a[i][i]
                        else:
                            for k in xrange(i - 1):
                                temp = temp - dconjg(a[k][i])*b[k][j]
                            if nounit:
                                temp = temp/dconjg(a[i][i])
                        b[i][j] = temp
            else:
                for j in xrange(n):
                    for i in xrange(m - 1, -1, -1):
                        temp = alpha*b[i][j]
                        if noconj:
                            for k in xrange(i + 1, m):
                                temp = temp - a[k][i]*b[k][j]
                            if nounit:
                                temp = temp/a[i][i]
                        else:
                            for k in xrange(i + 1, m):
                                temp = temp - dconjg(a[k][i])*b[k][j]
                            if nounit:
                                temp = temp/dconjg(a[i][i])
                        b[i][j] = temp
    else:
        if lsame(transa, 'N'):
            # Form  B := alpha*B*inv(A)
            if upper:
                for j in xrange(n):
                    if alpha!=one:
                        for i in xrange(m):
                            b[i][j] = alpha*b[i][j]
                    for k in xrange(j - 1):
                        if a[k][j]!=zero:
                            for i in xrange(m):
                                b[i][j] = b[i][j] - a[k][j]*b[i][k]
                    if nounit:
                        temp = one/a[j][j]
                        for i in xrange(m):
                            b[i][j] = temp*b[i][j]
            else:
                for j in xrange(n - 1, -1, -1):
                    if alpha!=one:
                        for i in xrange(m):
                            b[i][j] = alpha*b[i][j]
                    for k in xrange(j + 1, n):
                        if a[k][j]!=zero:
                            for i in xrange(m):
                                b[i][j] = b[i][j] - a[k][j]*b[i][k]
                    if nounit:
                        temp = one/a[j][j]
                        for i in xrange(m):
                            b[i][j] = temp*b[i][j]
        else:
            # Form  B := alpha*B*inv(A**T)  or  B := alpha*B*inv(A**H)
            if upper:
                for k in xrange(n - 1, -1, -1):
                    if nounit:
                        if noconj:
                            temp = one/a[k][k]
                        else:
                            temp = one/dconjg(a[k][k])
                        for i in xrange(m):
                            b[i][k] = temp*b[i][k]
                    for j in xrange(k - 1):
                        if a[j][k]!=zero:
                            if noconj:
                                temp = a[j][k]
                            else:
                                temp = dconjg(a[j][k])
                        for i in xrange(m):
                            b[i][j] = b[i][j] - temp*b[i][k]
                    if alpha!=one:
                        for i in xrange(m):
                            b[i][k] = alpha*b[i][k]
            else:
                for k in xrange(n):
                    if nounit:
                        if noconj:
                            temp = one/a[k][k]
                        else:
                            temp = one/dconjg(a[k][k])
                        for i in xrange(m):
                            b[i][k] = temp*b[i][k]
                    for j in xrange(k + 1, n):
                        if a[j][k]!=zero:
                            if noconj:
                                temp = a[j][k]
                            else:
                                temp = dconjg(a[j][k])
                            for i in xrange(m):
                                b[i][j] = b[i][j] - temp*b[i][k]
                    if alpha!=one:
                        for i in xrange(m):
                            b[i][k] = alpha*b[i][k]
    
    return

ctrsm = ztrsm
__all__.append('ctrsm'); __all__.append('ztrsm')

################################################################################

def ztrsv(uplo, trans, diag, n, a, lda, x, incx):
    """
    Solves one of the systems of equations
    
        A*x = b,  or  A**T*x = b,  or  A**H*x = b,
    
    where b and x are n element vectors and A is an n by n unit, or non-unit,
    upper or lower triangular matrix.
    
    No test for singularity or near-singularity is included in this routine.
    Such tests must be performed before calling this routine.
    
    Parameters
    ----------
    uplo : str
        On entry, ``uplo`` specifies whether the matrix is an upper or lower 
        triangular matrix as follows:
        
        - ``uplo`` = 'U' or 'u'  A is an upper triangular matrix
        - ``uplo`` = 'L' or 'l'  A is a lower triangular matrix
    
    trans : str
        On entry, ``trans`` specifies the operation to be performed as follows:
        
        - ``trans`` = 'N' or 'n'  A*x = b.
        - ``trans`` = 'T' or 't'  A**T*x = b.
        - ``trans`` = 'C' or 'c'  A**H*x = b.
    
    diag : str
        On entry, ``diag`` specifies whether or not ``a`` is unit triangular
        as follows:
        
        - ``diag`` = 'U' or 'u'  A is assumed to be unit triangular
        - ``diag`` = 'N' or 'n'  A is not assumed to be unit triangular
    
    n : int
        On entry, ``n`` specifies the order of the matrix ``a``. ``n`` must
        be at least zero.
    a : array
        ``a`` is a ``complex`` array of dimension ``(lda, n)``. Before entry
        with ``uplo`` = 'U' or 'u', the leading ``n`` by ``n`` upper
        triangular part of the array ``a`` must contain the upper triangular
        matrix and the strictly lower triangular part of ``a`` is not 
        referenced. Before engty with ``uplo`` = 'L' or 'l', the leading ``n``
        by ``n`` lower triangular part of the array ``a`` must contain the
        lower triangular matrix and the strictly upper triangular part of
        ``a`` is not referenced. Note that when ``diag`` = 'U' or 'u', the
        diagonal elements of ``a`` are not referenced either, but are assumed
        to be unity.
    lda : int
        On entry, ``lda`` specifies the first dimension of ``a`` as declared
        in the calling (sub) program. ``lda`` must be at least ``max([1, n])``.
    x : array
        ``x`` is a ``complex`` array of dimension at least
        ``(1 + (n - 1)*abs(incx)``. Before entry, the incremented array ``x``
        must contain the ``n`` element right-hand side vector b. On exit, 
        ``x`` is overwritten with the transformed vector x.
    incx : int
        On entry, ``incx`` specifies the increment for the elements of ``x``.
        ``incx`` must not be zero.
        
    Original Group
    --------------
    complex16_blas_level2
    
    """
    zero = complex(0.0, 0.0)
    
    # Test the input parameters
    info = 0
    if lsame(uplo, 'U') and not lsame(uplo, 'l'):
        info = 1
    elif lsame(trans, 'N') and not lsame(trans, 'T') and not lsame(trans, 'C'):
        info = 2
    elif not lsame(diag, 'U') and not lsame(diag, 'N'):
        info = 3
    elif n<0:
        info = 4
    elif lda<max([1, n]):
        info = 6
    elif incx==0:
        info = 8
    
    if info!=0:
        xerbla('ztrsv', info)
        return
    
    # Quick return if possible
    if n==0:
        return
    
    noconj = lsame(trans, 'T')
    nounit = lsame(diag, 'N')
    
    # Set up the start point in X if the increment is not unity. This will be
    # (N-1)*INCX too small for descending loops
    if incx<=0:
        kx = (n - 1)*incx
    elif incx!=1:
        kx = 0
    
    # Start the operation. In this version, the elements of A are accessed
    # sequentially with one pass through A.
    if lsame(trans, 'N'):
        # Form  x := inv(A)*x.
        if lsame(uplo, 'U'):
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    if x[j]!=zero:
                        if nounit:
                            x[j] = x[j]/a[j][j]
                        temp = x[j]
                        for i in xrange(j - 1, -1, -1):
                            x[i] = x[i] - temp*a[i][j]
            else:
                jx = kx + (n - 1)*incx
                for j in xrange(n - 1, -1, -1):
                    if x[jx]!=zero:
                        if nounit:
                            x[jx] = x[jx]/a[j][j]
                        temp = x[jx]
                        ix = jx
                        for i in xrange(j - 2, -1, -1):
                            ix = ix - incx
                            x[ix] = x[ix] - temp*a[i][j]
                    jx = jx - incx
        else:
            if incx==1:
                for j in xrange(n):
                    if x[j]!=zero:
                        if nounit:
                            x[j] = x[j]/a[j][j]
                        temp = x[j]
                        for i in xrange(j + 1, n):
                            x[i] = x[i] - temp*a[i][j]
            else:
                jx = kx
                for j in xrange(n):
                    if x[jx]!=zero:
                        if nounit:
                            x[jx] = x[jx]/a[j][j]
                        temp = x[jx]
                        ix = jx
                        for i in xrange(j + 1, n):
                            ix = ix + incx
                            x[ix] = x[ix] - temp*a[i][j]
                    jx = jx + incx
    else:
        # Form  x := inv(A**T)*x  or  x := inv(A**H)*x.
        if lsame(uplo, 'U'):
            if incx==1:
                for j in xrange(n):
                    temp = x[j]
                    if noconj:
                        for i in xrange(j - 1):
                            temp = temp - a[i][j]*x[i]
                        if nounit:
                            temp = temp/a[j][j]
                    else:
                        for i in xrange(j - 1):
                            temp = temp - dconjg(a[i][j])*x[i]
                        if nounit:
                            temp = temp/dconjg(a[j][j])
                    x[j] = temp
            else:
                jx = kx
                for j in xrange(n):
                    ix = kx
                    temp = x[jx]
                    if noconj:
                        for i in xrange(j - 1):
                            temp = temp - a[i][j]*x[ix]
                            ix = ix + incx
                        if nounit:
                            temp = temp/a[j][j]
                    else:
                        for i in xrange(j - 1):
                            temp = temp - dconjg(a[i][j])*x[i]
                            ix = ix + incx
                        if nounit:
                            temp = temp/dconjg(a[j][j])
                    x[jx] = temp
                    jx = jx + incx
        else:
            if incx==1:
                for j in xrange(n - 1, -1, -1):
                    temp = x[j]
                    if noconj:
                        for i in xrange(n - 1, j - 1, -1):
                            temp = temp - a[i][j]*x[i]
                        if nounit:
                            temp = temp/a[j][j]
                    else:
                        for i in xrange(n - 1, j - 1, -1):
                            temp = temp - dconjg(a[i][j])*x[i]
                        if nounit:
                            temp = temp/dcongj(a[j][j])
                    x[j] = temp
            else:
                kx = kx + (n - 1)*incx
                jx = kx
                for j in xrange(n - 1, -1, -1):
                    ix = kx
                    temp = x[jx]
                    if noconj:
                        for i in xrange(n - 1, j - 1, -1):
                            temp = temp - a[i][j]*x[ix]
                            ix = ix - incx
                        if nounit:
                            temp = temp/a[j][j]
                    else:
                        for i in xrange(n - 1, j - 1, -1):
                            temp = temp - dconjg(a[i][j])*x[ix]
                            ix = ix - incx
                        if nounit:
                            temp = temp/dconjg(a[j][j])
                    x[jx] = temp
                    jx = jx - incx
    
    return

ctrsv = ztrsv
__all__.append('ctrsv'); __all__.append('ztrsv')

################################################################################

