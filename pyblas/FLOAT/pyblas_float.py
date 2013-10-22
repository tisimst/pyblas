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
from pyblas.AUXILIARY import *

__all__ = []

################################################################################
 
def dasum(n, dx, incx):
    """
    Take the sum of the absolute values
    
    Parameters
    ----------
    n : int
    dx : array
    inx : int
    
    Original Group
    --------------
    double_blas_level1
    
    """
    dasum = 0.0
    dtemp = 0.0
    if n<=0 or incx<=0:
        return
    if incx==1:
        # Code for increment equal to 1
        m = n%6
        if m!=0:
            for i in xrange(m):
                dtemp = dtemp + abs(dx[i])
            if n<6:
                dasum = dtemp
                return dasum
        for i in xrange(m, n, 6):
            dtemp = dtemp + abs(dx[i]) + abs(dx[i + 1]) + abs(dx[i + 2]) + \
                    abs(dx[i + 3]) + abs(dx[i + 4]) + abs(dx[i + 5])
    
    else:
        # Code for increment not equal to 1
        nincx = n*incx
        for i in xrange(0, nincx, incx):
            dtemp = dtemp + abs(dx[i])
    
    dasum = dtemp
    return dasum

__all__.append('dasum')

################################################################################
 
def daxpy(n, da, dx, incx, dy, incy):
    """
    Constant types a vector plus a vector ``dy + da*dx``, uses unrolled 
    loops for increments equal to one.
    
    Parameters
    ----------
    n : int
    da : scalar
    dx : array
    incx : int
    dy : array
    incy : int
    
    Original Group
    --------------
    double_blas_level1
    
    """
    if n<=0:
        return
    if da==0.0:
        return
    if incx==1 and incx==1:
        # Code for both increments equal to 1
        m = n%4
        if m!=0:
            for i in xrange(m):
                dy[i] = dy[i] + da*dx[i]
        if n<4:
            return
        for i in xrange(m, n, 4):
            dy[i] = dy[i] + da*dx[i]
            dy[i + 1] = dy[i + 1] + da*dx[i + 1]
            dy[i + 2] = dy[i + 2] + da*dx[i + 2]
            dy[i + 3] = dy[i + 3] + da*dx[i + 3]
    
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 1
        iy = 1
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy + (-n + 1)*incy
        for i in xrange(n):
            dy[iy] = dy[iy] + da*dx[ix]
            ix = ix + incx
            iy = ix + incy
    
    return

__all__.append('daxpy')

################################################################################

def dcopy(n, dx, incx, dy, incy):
    """
    Copies a vector, x, to a vector, y. Uses unrolled loops for increments 
    equal to 1.
    
    Parameters
    ----------
    n : int
    dx : array
    incx : int
    dy : array
    incy : int
    
    Original Group
    --------------
    double_blas_level1
    
    """
    if n<=0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        m = n%7
        if m!=0:
            for i in xrange(m):
                dy[i] = dx[i]
            if n<7:
                return
        for i in xrange(m, n, 7):
            dy[i] = dx[i]
            dy[i + 1] = dx[i + 1]
            dy[i + 2] = dx[i + 2]
            dy[i + 3] = dx[i + 3]
            dy[i + 4] = dx[i + 4]
            dy[i + 5] = dx[i + 5]
            dy[i + 6] = dx[i + 6]
    
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 1
        iy = 1
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incy
        for i in xrange(n):
            dy[iy] = dx[ix]
            ix = ix + incx
            iy = iy + incy

    return

__all__.append('dcopy')

################################################################################

def ddot(n, dx, incx, dy, incy):
    """
    Forms the dot product of two vectors
    
    Parameters
    ----------
    n : int
    dx : array
    incx : int
    dy : array
    incy : int
    
    Original Group
    --------------
    double_blas_level1
    
    """
    dtemp = 0.0
    if n<=0:
        return
    if incx==1 and incy==1:
        # Code for both increments equal to 1
        m = n%5
        if m!=0:
            for i in xrange(m):
                dtemp = dtemp + dx[i]*dy[i]
            if n<5:
                return dtemp
        for i in xrange(m, n, 5):
            dtemp = dtemp + dx[i]*dy[i] + dx[i + 1]*dy[i + 1] + \
                 dx[i + 2]*dy[i + 2] + dx[i + 3]*dy[i + 3] + \
                 dx[i + 4]*dy[i + 4]
             
    else:
        # Code for unequal increments or equal increments not equal to 1
        ix = 0
        iy = 0
        if incx<0:
            ix = (-n + 1)*incx
        if incy<0:
            iy = (-n + 1)*incy
        for i in xrange(n):
            dtemp = dtemp + dx[ix]*dy[iy]
            ix = ix + incx
            iy = iy + incy
            
    return dtemp

__all__.append('ddot')

################################################################################

def dgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy):
    """
    Performs one of the matrix-vector operations:
    
        y := alpha*A*x + beta*y  or  y := alpha*A**T*x + beta*y
    
    where alpha and beta are scalars, x and y are vectors and A is an m
    by n band matrix, with kl sub-diagonals and ku super-diagonals.
    
    Parameters
    ----------
    trans : str
        On entry, TRANS specifies the operation to be performed as follows:
            
            TRANS = 'N' or 'n'  y := alpha*A*x + beta*y
            TRANS = 'T' or 't'  y := alpha*A**T*x + beta*y
            TRANS = 'C' or 'c'  y := alpha*A**T*x + beta*y
    m : int
        On entry, M specifies the number of rows of the matrix A. M must
        be at least zero.
    n : int
        On entry, N specifies the number of columns of the matrix A. N
        must be at least zero
    kl : int
        On entry, KL specifies the number of sub-diagonals of the matrix
        A. KL must satisfy KL>=0.
    ku : int
        On entry, KU specifies the number of super-diagonals of the matrix
        A. KU must satisfy KU>=0.
    alpha : scalar
        On entry, ALPHA specifies the scalar alpha.
    A : 2d-array
        A matrix of dimension (LDA, N). Before entry, the leading 
        (KL + KU + 1) by N part of the array A must contain the matrix
        coefficients, supplied column by column, with the leading diagonal
        of the matrix in row (KU + 1) of the array, the first super-
        diagonal starting at position 1 in row KU, the first sub-diagonal
        starting at position 0 in row (KU + 2), and so on. Elements in the
        array A that do not correspond to elements in the band matrix 
        (such as the top left KU by KU triangle) are not referenced. The
        following program segment will transfer a band matrix from 
        conventional full matrix storage to band storage::
        
            for j in xrange(n):
                k = ku - j
                for i in xrange(max(0, j - ku - 1), min(m, j + kl + 1)):
                    a[k + i][j] = matrix[i][j]
    
    lda : int
        On entry, LDA specifies the first dimension of A as declared in
        the calling (sub) program. LDA must be at least (KL + KU + 1).
    x : array
        X is an array of dimension at least (1 + (N - 1)*abs(incx)) when
        TRANS = 'N' or 'n' and at least (1 + (M - 1)*abs(incx)) otherwise.
        Before entry, the incremented array X must contain the vector x.
    incx : int
        On entry, INCX specifies the increment for the elements of X. INCX
        must not be zero.
    beta : scalar
        On entry, BETA specifies the scalar beta. When BETA is supplied as
        zero then Y need not be set on input.
    y : array
        Y is an array of dimension at least (1 + (m - 1)*abs(incy)) when
        TRANS = 'N' or 'n' and at least (1 + (n - 1)*abs(incy)) otherwise.
        Before entry, the incremented array Y must contain the vector y.
        On exit, Y is overwritten by the updated vector y.
    incy : int
        On entry, incy specifies the increment for the elements of Y. INCY
        must not be zero.
        
    Original Group
    --------------
    double_blas_level2
    
    """
    
    one = 1.0
    zero = 0.0
    
    info = 0
    if not lsame(trans, 'N') and not lsame(trans, 'T') and \
        not lsame(trans, 'C'):
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
        xerbla('dgbmv', info)
        return
    
    # Quick return if possible
    if m==0 or n==0 or (alpha==zero and beta==one):
        return
    
    # Set LENX and LENY, the lengths of the vectors x and y, and set
    # up the start points in X and Y.
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
                for i in xrange(max([0, j - ku]), min([m, j + kl])):
                    temp = temp + a[k + i][j]*x[i]
                y[jy] = y[jy] + alpha*temp
                jy = jy + incy
        else:
            for j in xrange(n):
                temp = zero
                ix = kx
                k = kup1 - j
                for i in xrange(max([0, j - ku]), min([m, j + kl])):
                    temp = temp + a[k + i][j]*x[i]
                    ix = ix + incx
                y[jy] = y[jy] + alpha*temp
                jy = jy + incy
                if j>ku:
                    kx = kx + incx

    return

__all__.append('dgbmv')

################################################################################

def dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc):
    """
    Performs one of the matrix-matrix operations
    
        C := alpha*op(A)*op(B) + beta*C
    
    where op(X) is one of
    
        op(X) = X  or op(X) = X**T
    
    alpha and beta are scalars, and A, B, and C are matrices, op(A) an m
    by k matrix, op(B) a k by n matrix and C an m by n matrix.
    
    Parameters
    ----------
    transa : str
        On entry, TRANSA specifies the form of op(A) to be used in the 
        matrix multiplication as follows:
        
        - TRANSA = 'N' or 'n',  op(A) = A
        - TRANSA = 'T' or 't',  op(A) = A**T
        - TRANSA = 'C' or 'c',  op(A) = A**T
        
    transb : str
        On entry, TRANSB specifies the form of op(B) to be used in the 
        matrix multiplication as follows:
        
        - TRANSB = 'N' or 'n',  op(B) = B
        - TRANSB = 'T' or 't',  op(B) = B**T
        - TRANSB = 'C' or 'c',  op(B) = B**T
        
    m : int
        On entry, M specifies the number of rows in the matrix op(A)
        and of the matrix C. M must be at least zero.
    n : int
        On entry, N specifies the number of columns in the matrix op(B)
        and of the number of columns of the matrix C. N must be at 
        least zero.
    k : int
        On entry, K specifies the number of columns of the matrix op(A)
        and the number fo rows of the matrix op(B). K must be at least
        zero.
    alpha : scalar
        On entry, ALPHA specifies the scalar alpha.
    a : 2d-array
        An array of dimension (LDA, ka), where ka is k when TRANSA = 'N' 
        or 'n', and is m otherwise. Before with TRANSA = 'N' or 'n', the 
        leading m by k part of the array A must contain the matrix A, 
        otherwise the leading k by m part of the array A must contain the 
        matrix A.
    lda : int
        On entry, LDA specifies the first dimension of A as declared 
        in calling (sub) program. When TRANS = 'N' or 'n' then LDA must be 
        at least max([1, m]), otherwise LDA must be at least max([1, k]).
    b : complex array
        An array of dimension ``(ldb, kb)``, where ``kb`` is ``n``
        when ``transb = 'N'`` or ``'n'``, and is ``k`` otherwise. Before entry
        with ``transb = 'N'`` or ``'n'``, the leading ``k`` by ``n`` part of
        the array ``b`` must contain the matrix ``b``, otherwise the leading
        ``n`` by ``k`` part of the array ``b`` must contain the matrix ``b``.
    ldb : int
        On entry, ``ldb`` specifies the first dimension of ``b`` as declared 
        in calling (sub) program. When ``transb = 'N'`` or ``'n'`` then ``ldb``
        must be at least ``max([1, k])``, otherwise ``ldb`` must be at least
        ``max([1, n])``.
    beta : scalar
        On entry, ``beta`` specifies the scalar beta. When ``beta`` is supplied
        as zero, then ``c`` need not be set on input.
    c : array
        An array of dimension ``(ldc, n)``. Before entry, the leading
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

__all__.append('dgemm')

################################################################################
