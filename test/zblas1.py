"""
Test program for the COMPLEX Level 1 BLAS.
"""
import sys

sys.path.insert(0, os.path.abspath('..'))

from pyblas.COMPLEX import *
from pyblas.AUXILIARY import *

SFAC = 9.765625e-4

EPSILON = sys.float_info.epsilon
    
################################################################################

def header():
    global icase, n, incx, incy, mode, PASS
    L = [
        'zdotc',
        'zdotu',
        'zaxpy',
        'zcopy',
        'zswap',
        'dznrm2',
        'dzasum',
        'zscal',
        'zdscal',
        'izamax'
        ]
    
    print ' Test of subprogram number', icase, L[icase]
    
################################################################################

def check1(sfac):
    global icase, n, incx, incy, mode, PASS
    sa = 0.3
    ca = 0.4-0.7j
    
    cv = [[[0.0 for k in xrange(2)] for j in xrange(5)] for i in xrange(8)]
    cv_data1 = [
        0.1+0.1j,  1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 
        0.3-0.4j,  3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 
        0.1-0.3j,  0.5-0.1j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 
        0.1+0.1j, -0.6+0.1j, 0.1-0.3j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 
        0.3+0.1j,  0.5+0.0j, 0.0+0.5j, 0.0+0.2j, 2.0+3.0j, 2.0+3.0j, 2.0+3.0j, 2.0+3.0j
        ]
    cv_data2 = [
        0.1+0.1j, 4.0+5.0j,  4.0+5.0j, 4.0+5.0j, 4.0+5.0j, 4.0+5.0j, 4.0+5.0j, 4.0+5.0j, 
        0.3-0.4j, 6.0+7.0j,  6.0+7.0j, 6.0+7.0j, 6.0+7.0j, 6.0+7.0j, 6.0+7.0j, 6.0+7.0j, 
        0.1-0.3j, 8.0+9.0j,  0.5-0.1j, 2.0+5.0j, 2.0+5.0j, 2.0+5.0j, 2.0+5.0j, 2.0+5.0j, 
        0.1+0.1j, 3.0+6.0j, -0.6+0.1j, 4.0+7.0j, 0.1-0.3j, 7.0+2.0j, 7.0+2.0j, 7.0+2.0j, 
        0.3+0.1j, 5.0+8.0j,  0.5+0.0j, 6.0+9.0j, 0.0+0.5j, 8.0+3.0j, 0.0+0.2j, 9.0+4.0j
        ]
    for j in xrange(5):
        for i in xrange(8):
            cv[i][j][0] = cv_data1[i + 8*j]
            cv[i][j][1] = cv_data2[i + 8*j]
    for i in xrange(8):
        for j in xrange(5):
            cv[i][j] = tuple(cv[i][j])
        cv[i] = tuple(cv[i])
    cv = tuple(cv)

    strue2 = (0.0, 0.5, 0.6, 0.7, 0.8)
    strue4 = (0.0, 0.7, 1.0, 1.3, 1.6)

    ctrue5 = [[[0.0 for k in xrange(2)] for j in xrange(5)] for i in xrange(8)]
    ctrue5_data1 = [
           0.1+0.1j,    1.0+2.0j,    1.0+2.0j,   1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j,
        -0.16-0.37j,    3.0+4.0j,    3.0+4.0j,   3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j,
        -0.17-0.19j,  0.13-0.39j,    5.0+6.0j,   5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j,
          0.11-0.3j, -0.17+0.46j, -0.17-0.19j,   7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j,
         0.19-0.17j,  0.20-0.35j,  0.35+0.20j, 0.14+0.08j, 2.0+3.0j, 2.0+3.0j, 2.0+3.0j, 2.0+3.0j
        ]
    ctrue5_data2 = [
            0.1+0.1j,   4.0+5.0j,    4.0+5.0j, 4.0+5.0j,    4.0+5.0j, 4.0+5.0j,   4.0+5.0j, 4.0+5.0j,
         -0.16-0.37j,   6.0+7.0j,    6.0+7.0j, 6.0+7.0j,    6.0+7.0j, 6.0+7.0j,   6.0+7.0j, 6.0+7.0j,
         -0.17-0.19j,   8.0+9.0j,  0.13-0.39j, 2.0+5.0j,    2.0+5.0j, 2.0+5.0j,   2.0+5.0j, 2.0+5.0j,
          0.11-0.03j,   3.0+6.0j, -0.17+0.46j, 4.0+7.0j, -0.17-0.19j, 7.0+2.0j,   7.0+2.0j, 7.0+2.0j,
          0.19-0.17j,   5.0+8.0j,  0.20-0.35j, 6.0+9.0j,  0.35+0.20j, 8.0+3.0j, 0.14+0.08j, 9.0+4.0j
        ]
    for j in xrange(5):
        for i in xrange(8):
            ctrue5[i][j][0] = ctrue5_data1[i + 8*j]
            ctrue5[i][j][1] = ctrue5_data2[i + 8*j]
    for i in xrange(8):
        for j in xrange(5):
            ctrue5[i][j] = tuple(ctrue5[i][j])
        ctrue5[i] = tuple(ctrue5[i])
    ctrue5 = tuple(ctrue5)
            
    ctrue6 = [[[0.0 for k in xrange(2)] for j in xrange(5)] for i in xrange(8)]
    ctrue6_data1 = [
          0.1+0.1j,    1.0+2.0j,   1.0+2.0j,   1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j, 1.0+2.0j,
        0.09-0.12j,    3.0+4.0j,   3.0+4.0j,   3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j, 3.0+4.0j,
        0.03-0.09j,  0.15-0.03j,   5.0+6.0j,   5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j, 5.0+6.0j,
        0.03+0.03j, -0.18+0.03j, 0.03-0.09j,   7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j, 7.0+8.0j,
        0.09+0.03j,  0.15+0.00j, 0.00+0.15j, 0.00+0.06j, 2.0+3.0j, 2.0+3.0j, 2.0+3.0j, 2.0+3.0j
        ]
    ctrue6_data2 = [
          0.1+0.1j, 4.0+5.0j,    4.0+5.0j, 4.0+5.0j,   4.0+5.0j, 4.0+5.0j,   4.0+5.0j, 4.0+5.0j,
        0.09-0.12j, 6.0+7.0j,    6.0+7.0j, 6.0+7.0j,   6.0+7.0j, 6.0+7.0j,   6.0+7.0j, 6.0+7.0j,
        0.03-0.09j, 8.0+9.0j,  0.15-0.03j, 2.0+5.0j,   2.0+5.0j, 2.0+5.0j,   2.0+5.0j, 2.0+5.0j,
        0.03+0.03j, 3.0+6.0j, -0.18+0.03j, 4.0+7.0j, 0.03-0.09j, 7.0+2.0j,   7.0+2.0j, 7.0+2.0j,
        0.09+0.03j, 5.0+8.0j,  0.15+0.00j, 6.0+9.0j, 0.00+0.15j, 8.0+3.0j, 0.00+0.06j, 9.0+4.0j
        ]
    for j in xrange(5):
        for i in xrange(8):
            ctrue6[i][j][0] = ctrue6_data1[i + 8*j]
            ctrue6[i][j][1] = ctrue6_data2[i + 8*j]
    for i in xrange(8):
        for j in xrange(5):
            ctrue6[i][j] = tuple(ctrue6[i][j])
        ctrue6[i] = tuple(ctrue6[i])
    ctrue6 = tuple(ctrue6)

    itrue3 = (-1, 0, 1, 1, 1)
    
    cx = [0.0]*8
    for incx in [1, 2]:
        for np1 in xrange(4):
            n = np1
            LEN = 2*max([n, 1])
            
            for i in xrange(LEN):
                cx[i] = cv[i][np1][incx - 1]
            
            if icase==5:
                # DZNRM2
                stest1(dznrm2(n, cx, incx), strue2[np1], strue2[np1], sfac)
            elif icase==6:
                # DZASUM
                stest1(dzasum(n, cx, incx), strue4[np1], strue4[np1], sfac)
            elif icase==7:
                # ZSCAL
                zscal(n, ca, cx, incx)
                ctest(LEN, cx, ctrue5[0][np1][incx - 1], ctrue5[0][np1][incx - 1], sfac)
            elif icase==8:
                # ZDSCAL
                zdscal(n, sa, cx, incx)
                ctest(LEN, cx, ctrue6[0][np1][incx - 1], ctrue6[0][np1][incx - 1], sfac)
            elif icase==9:
                # IZAMAX
                itest1(izamax(n, cx, incx), itrue3[np1])
            else:
                print " Shouldn't be here in CHECK1"
                exit()
    
    incx = 1
    if icase==7:
        # ZSCAL
        # Add a test for alpha equal to zero.
        ca = 0j
        mwpct = tuple([0.0j for i in xrange(5)])
        mwpcs = tuple([1.0j for i in xrange(5)])
        zscal(5, ca, cx, incx)
        ctest(5, cx, mwpct, mwpcs, sfac)
    elif icase==8:
        # ZDSCAL
        # Add a test for alpha equal to zero.
        sa = 0.0
        mwpct = tuple([0.0j for i in xrange(5)])
        mwpcs = tuple([1.0j for i in xrange(5)])
        zdscal(5, sa, cx, incx)
        ctest(5, cx, mwpct, mwpcs, sfac)
        
        # Add a test for alpha equal to one.
        sa = 1.0
        mwpct = tuple([cx[i] for i in xrange(5)])
        mwpcs = tuple([cx[i] for i in xrange(5)])
        zdscal(5, sa, cx, incx)
        ctest(5, cx, mwpct, mwpcs, sfac)
        
        # Add a test for alpha equal to minus one.
        sa = -1.0
        mwpct = tuple([-cx[i] for i in xrange(5)])
        mwpcs = tuple([-cx[i] for i in xrange(5)])
        zdscal(5, sa, cx, incx)
        ctest(5, cx, mwpct, mwpcs, sfac)
    
################################################################################

def check2(sfac):
    global icase, n, incx, incy, mode, PASS
    ca = 0.4-0.7j
    incxs = (1, 2, -2, -1)
    incys = (1, -2, 1, -2)
    lens = ((1, 1), (1, 1), (2, 3), (4, 7))
    ns = (0, 1, 2, 4)
    cx1 = (0.7-0.8j, -0.4-0.7j, -0.1-0.9j, 0.2-0.8j, -0.9-0.4j, 0.1+0.4j, -0.6+0.6j)
    cy1 = (0.6-0.6j, -0.9+0.5j, 0.7-0.6j, 0.1-0.5j, -0.1-0.2j, -0.5-0.3j, 0.8-0.7j)
    
    ct8 = [[[0.0 for k in xrange(4)] for j in xrange(4)] for i in xrange(7)]
    ct8_data1 = [
          0.6-0.6j,   0.0+0.0j,   0.0+0.0j,    0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.32-1.41j,   0.0+0.0j,   0.0+0.0j,    0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.32-1.41j, -1.55+0.5j,   0.0+0.0j,    0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.32-1.41j, -1.55+0.5j, 0.03-0.89j, -0.38-0.96j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j
        ]
    ct8_data2 = [
           0.6-0.6j,  0.0+0.0j,   0.0+0.0j, 0.0+0.0j,    0.0+0.0j,  0.0+0.0j,   0.0+0.0j,
         0.32-1.41j,  0.0+0.0j,   0.0+0.0j, 0.0+0.0j,    0.0+0.0j,  0.0+0.0j,   0.0+0.0j,
        -0.07-0.89j, -0.9+0.5j, 0.42-1.41j, 0.0+0.0j,    0.0+0.0j,  0.0+0.0j,   0.0+0.0j,
          0.78+0.6j, -0.9+0.5j, 0.06-0.13j, 0.1-0.5j, -0.77-0.49j, -0.5-0.3j, 0.52-1.51j
        ]
    ct8_data3 = [
           0.6-0.6j,    0.0+0.0j,   0.0+0.0j,    0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
         0.32-1.41j,    0.0+0.0j,   0.0+0.0j,    0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        -0.07-0.89j, -1.18-0.31j,   0.0+0.0j,    0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
         0.78+0.06j, -1.54+0.97j, 0.03-0.89j, -0.18-1.31j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j
        ]
    ct8_data4 = [
          0.6-0.6j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,    0.0+0.0j,  0.0+0.0j,   0.0+0.0j,
        0.32-1.41j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,    0.0+0.0j,  0.0+0.0j,   0.0+0.0j,
        0.32-1.41j, -0.9+0.5j, 0.05-0.6j, 0.0+0.0j,    0.0+0.0j,  0.0+0.0j,   0.0+0.0j,
        0.32-1.41j, -0.9+0.5j, 0.05-0.6j, 0.1-0.5j, -0.77-0.49j, -0.5-0.3j, 0.32-1.16j
        ]
    for j in xrange(4):
        for i in xrange(7):
            ct8[i][j][0] = ct8_data1[i + 7*j]
            ct8[i][j][1] = ct8_data2[i + 7*j]
            ct8[i][j][2] = ct8_data3[i + 7*j]
            ct8[i][j][3] = ct8_data4[i + 7*j]
    for i in xrange(7):
        for j in xrange(4):
            ct8[i][j] = tuple(ct8[i][j])
        ct8[i] = tuple(ct8[i])
    ct8 = tuple(ct8)

    ct7 = [[0.0 for j in xrange(4)] for i in xrange(4)]
    ct7_data1 = [
        0.0+0.0j, -0.06-0.9j,  0.65-0.47j, -0.34-1.22j,
        0.0+0.0j, -0.06-0.9j, -0.59-1.46j, -1.04-0.04j,
        0.0+0.0j, -0.06-0.9j, -0.83+0.59j,  0.07-0.37j,
        0.0+0.0j, -0.06-0.9j, -0.76-1.15j, -1.33-1.82j
        ]
    for j in xrange(4):
        for i in xrange(4):
            ct7[i][j] = ct7_data1[i + 4*j]
    for i in xrange(4):
        ct7[i] = tuple(ct7[i])
    ct7 = tuple(ct7)
    
    ct6 = [[0.0 for j in xrange(4)] for i in xrange(4)]
    ct6_data1 = [
        0.0+0.0j, 0.90+0.06j,  0.91-0.77j, 1.80-0.10j,
        0.0+0.0j, 0.90+0.06j,  1.45+0.74j, 0.20+0.90j,
        0.0+0.0j, 0.90+0.06j, -0.55+0.23j, 0.83-0.39j,
        0.0+0.0j, 0.90+0.06j,  1.04+0.79j, 1.95+1.22j
        ]
    for j in xrange(4):
        for i in xrange(4):
            ct6[i][j] = ct6_data1[i + 4*j]
    for i in xrange(4):
        ct6[i] = tuple(ct6[i])
    ct6 = tuple(ct6)
    
    ct10x = [[[0.0 for k in xrange(4)] for j in xrange(4)] for i in xrange(7)]
    ct10x_data1 = [
        0.7-0.8j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j, -0.9+0.5j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j, -0.9+0.5j, 0.7-0.6j, 0.1-0.5j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j
        ]
    ct10x_data2 = [
        0.7-0.8j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.7-0.6j, -0.4-0.7j,  0.6-0.6j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.8-0.7j, -0.4-0.7j, -0.1-0.2j, 0.2-0.8j, 0.7-0.6j, 0.1+0.4j, 0.6-0.6j
        ]
    ct10x_data3 = [
         0.7-0.8j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
         0.6-0.6j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        -0.9+0.5j, -0.4-0.7j, 0.6-0.6j, 0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
         0.1-0.5j, -0.4-0.7j, 0.7-0.6j, 0.2-0.8j, -0.9+0.5j, 0.1+0.4j, 0.6-0.6j
        ]
    ct10x_data4 = [
        0.7-0.8j, 0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j, 0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j, 0.7-0.6j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.6-0.6j, 0.7-0.6j, -0.1-0.2j, 0.8-0.7j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j
        ]
    for j in xrange(4):
        for i in xrange(7):
            ct10x[i][j][0] = ct10x_data1[i + 7*j]
            ct10x[i][j][1] = ct10x_data2[i + 7*j]
            ct10x[i][j][2] = ct10x_data3[i + 7*j]
            ct10x[i][j][3] = ct10x_data4[i + 7*j]
    for i in xrange(7):
        for j in xrange(4):
            ct10x[i][j] = tuple(ct10x[i][j])
        ct10x[i] = tuple(ct10x[i])
    ct10x = tuple(ct10x)

    ct10y = [[[0.0 for k in xrange(4)] for j in xrange(4)] for i in xrange(7)]
    ct10y_data1 = [
        0.6-0.6j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.7-0.8j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.7-0.8j, -0.4-0.7j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.7-0.8j, -0.4-0.7j, -0.1-0.9j, 0.2-0.8j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j
        ]
    ct10y_data2 = [
         0.6-0.6j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,
         0.7-0.8j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,
        -0.1-0.9j, -0.9+0.5j,  0.7-0.8j, 0.0+0.0j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,
        -0.6+0.6j, -0.9+0.5j, -0.9-0.4j, 0.1-0.5j, -0.1-0.9j, -0.5-0.3j, 0.7-0.8j
        ]
    ct10y_data3 = [
         0.6-0.6j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
         0.7-0.8j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        -0.1-0.9j,  0.7-0.8j,  0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        -0.6+0.6j, -0.9-0.4j, -0.1-0.9j, 0.7-0.8j, 0.0+0.0j, 0.0+0.0j, 0.0+0.0j
        ]
    ct10y_data4 = [
        0.6-0.6j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,
        0.7-0.8j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,
        0.7-0.8j, -0.9+0.5j, -0.4-0.7j, 0.0+0.0j,  0.0+0.0j,  0.0+0.0j, 0.0+0.0j,
        0.7-0.8j, -0.9+0.5j, -0.4-0.7j, 0.1-0.5j, -0.1-0.9j, -0.5-0.3j, 0.2-0.8j
        ]
    for j in xrange(4):
        for i in xrange(7):
            ct10y[i][j][0] = ct10y_data1[i + 7*j]
            ct10y[i][j][1] = ct10y_data2[i + 7*j]
            ct10y[i][j][2] = ct10y_data3[i + 7*j]
            ct10y[i][j][3] = ct10y_data4[i + 7*j]
    for i in xrange(7):
        for j in xrange(4):
            ct10y[i][j] = tuple(ct10y[i][j])
        ct10y[i] = tuple(ct10y[i])
    ct10y = tuple(ct10y)

    csize1 = (0.0+0.0j, 0.9+0.9j, 1.63+1.73j, 2.90+2.78j)
    csize3 = (
          0.0+0.0j,   0.0+0.0j,   0.0+0.0j,   0.0+0.0j,   0.0+0.0j,   0.0+0.0j,   0.0+0.0j,
        1.17+1.17j, 1.17+1.17j, 1.17+1.17j, 1.17+1.17j, 1.17+1.17j, 1.17+1.17j, 1.17+1.17j
        )
    csize2 = ((0.0+0.0j, 1.54+1.54j),
              (0.0+0.0j, 1.54+1.54j),
              (0.0+0.0j, 1.54+1.54j),
              (0.0+0.0j, 1.54+1.54j),
              (0.0+0.0j, 1.54+1.54j),
              (0.0+0.0j, 1.54+1.54j),
              (0.0+0.0j, 1.54+1.54j))
    
    for ki in xrange(4):
        incx = incxs[ki]
        incy = incys[ki]
        mx = abs(incx) - 1
        my = abs(incy) - 1
        
        for kn in xrange(4):
            n = ns[kn]
            ksize = min([1, kn])
            lenx = lens[kn][mx]
            leny = lens[kn][my]
            
            cx = [0.0]*7
            cy = [0.0]*7
            for i in xrange(7):
                cx[i] = cx1[i]
                cy[i] = cy1[i]
            
            cdot = [0.0]
            if icase==0:
                # ZDOTC
                cdot[0] = zdotc(n, cx, incx, cy, incy)
                ctest(1, cdot, ct6[kn][ki], csize1[kn], sfac)
            elif icase==1:
                # ZDOTU
                cdot[0] = zdotu(n, cx, incx, cy, incy)
                ctest(1, cdot, ct7[kn][ki], csize1[kn], sfac)
            elif icase==2:
                # ZAXPY
                zaxpy(n, ca, cx, incx, cy, incy)
                ctest(leny, cy, ct8[0][kn][ki], csize2[0][ksize], sfac)
            elif icase==3:
                # ZCOPY
                zcopy(n, cx, incx, cy, incy)
                ctest(leny, cy, ct10y[0][kn][ki], csize3, 1.0)
            elif icase==4:
                # ZSWAP
                zswap(n, cx, incx, cy, incy)
                ctest(lenx, cx, ct10x[0][kn][ki], csize3, 1.0)
                ctest(leny, cy, ct10y[0][kn][ki], csize3, 1.0)
            else:
                print " Shouldn't be here in CHECK2"
                exit()
                
################################################################################

def stest(LEN, scomp, strue, ssize, sfac):
    """
    This function compares arrays SCOMP and STRUE of length LEN to see if the
    term by term differences, multiplied by SFAC, are negligible.
    """
    global icase, n, incx, incy, mode, PASS
        
    zero = 0.0
    
    for i in xrange(LEN):
        sd = scomp[i] - strue[i]
        if abs(sfac*sd)<=abs(ssize[i])*EPSILON:
            continue
        # Here SCOMP[i] is not close to STRUE[i]
        if not PASS:
            print '{:5d}{:3d}{:5d}{:5d}{:5d}{:3d}{:36.8f}{:36.8f}{:12.4f}{:12.4f}'.format(
                icase, n, incx, incy, mode, i, scomp[i], strue[i], sd, ssize[i])
        
        PASS = False
        print '                                       FAIL'
        head = ' CASE  N INCX INCY MODE  I                            ' + \
            ' COMP(I)                             TRUE(I)  DIFFERENCE' + \
            '     SIZE(I)'
        print head
        print '{:5d}{:3d}{:5d}{:5d}{:5d}{:3d}{:36.8f}{:36.8f}{:12.4f}{:12.4f}'.format(
            icase, n, incx, incy, mode, i, scomp[i], strue[i], sd, ssize[i])

################################################################################

def stest1(scomp1, strue1, ssize, sfac):
    """
    This is an interface subroutine to accomodate the Fortran requirement
    that when a dummy argument is an array, the actual arguement must also
    be an array or an array element.
    """
    scomp = [scomp1]
    strue = [strue1]
    stest(1, scomp, strue, [ssize], sfac)

################################################################################

def sdiff(sa, sb):
    """
    Computes the difference of two numbers
    """
    return sa - sb

################################################################################

def ctest(LEN, ccomp, ctrue, csize, sfac):
    
    ccomp = [0.0]*LEN
    csize = [0.0]*LEN
    ctrue = [0.0]*LEN
    scomp = [0.0]*20
    ssize = [0.0]*20
    strue = [0.0]*20
    
    for i in xrange(LEN):
        scomp[2*i - 1] = dble(ccomp[i])
        scomp[2*i] = dimag(ccomp[i])
        strue[2*i - 1] = dble(ctrue[i])
        strue[2*i] = dimag(ctrue[i])
        ssize[2*i - 1] = dble(csize[i])
        ssize[2*i] = dimag(csize[i])
    
    stest(2*LEN, scomp, strue, ssize, sfac)
    
################################################################################

def itest1(icomp, itrue):
    """
    This routine compares the variables ICOMP and ITRUE for equality
    """
    global icase, n, incx, incy, mode, PASS
    
    #print 'icomp:', icomp
    #print 'itrue:', itrue
    #print 'n:', n
    #print 'incx:', incx
    #print 'incy:', incy
    
    if icomp==itrue:
        return
    # Here ICOMP is not equal to ITRUE
    if not PASS:
        ID = icomp - itrue
        print '{:5d}{:3d}{:5d}{:5d}{:5d}{:36d}{:36d}{:12d}'.format(icase, n,
            incx, incy, mode, icomp, itrue, ID)
    
    PASS = False
    print '                                       FAIL'
    head = ' CASE  N INCX INCY MODE                               ' + \
        ' COMP                                TRUE     DIFFERENCE'
    print head
    ID = icomp - itrue
    print '{:5d}{:3d}{:5d}{:5d}{:5d}{:36d}{:36d}{:12d}'.format(icase, n,
        incx, incy, mode, icomp, itrue, ID)

################################################################################

print ' Complex BLAS Test Program Results'
for ic in xrange(10):
    icase = ic
    header()
    
    # Initialize PASS, INCX, INCY, and MODE for a new case.
    # The value for 9999 for INCX, INCY or MODE will appear in the
    # detailed output, if any, for cases that do not involve
    # these parameters.
    
    PASS = True
    incx = 9999
    incy = 9999
    mode = 9999
    
    if icase<=4:
        check2(SFAC)
    elif icase>=5:
        check1(SFAC)
    
    if PASS:
        print '                                    ----- PASS -----'

################################################################################

print ' TESTING COMPLETE.'