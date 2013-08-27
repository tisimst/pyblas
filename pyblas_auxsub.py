"""
This file contains auxiliary subroutines needed by the pyBLAS package
"""

__all__ = []  # add function names as they are defined

################################################################################

def xerbla(srname, info):
    """
    Error handler for BLAS/LAPACK routines. It is called when an input parameter
    is given an invalid value.
    
    Parameters
    ----------
    srname : str
        The name of the routine which called ``xerbla``.
    info : int
        The position of the invalid parameter in the parameter list of the
        calling routine.
        
    """
    print '** On entry to {:} parameter number {:} had an illegal value.'.format(srname, info)

__all__.append('xerbla')

################################################################################

def dcabs1(z):
    return abs(z.real) + abs(z.imag)

__all__.append('dcabs1')
    
################################################################################

def lsame(ca, cb):
    """
    Compare two characters to see if the same regardless of case.
    
    Parameters
    ----------
    ca : str
        A single character to test.
    cb : str
        A single character that is tested against.
    
    Returns
    -------
    test : bool
        ``True`` if ``ca`` is the same letter as ``cb`` regardless of case.
    
    Original Group
    --------------
    aux_blas
    
    Reference
    ---------
    BLAS 3.4.2

    """
    ichar = ord
    
    # Test if the characters are equal
    if ca==cb:
        return True
    
    # Now test for equivalence if both characters are alphabetic.
    zcode = ichar('Z')
    
    # Use 'Z' rather than 'A' so that ASCII can be detected on Prime
    # machines, on which ICHAR returns a value with bit 8 set.
    # ICHAR('A') on Prime machines returns 193 which is the same as
    # ICHAR('A') on an EBCDIC machine.

    inta = ichar(ca)
    intb = ichar(cb)
    
    if zcode==90 or zcode==122:
        # ASCII assumed - ``zcode`` is the ASCII code of either lower or
        # upper case 'Z'.
        if inta>=97 and inta<=122:
            inta = inta - 32
        if intb>=97 and intb<=122:
            intb = intb - 32
    elif zcode==233 or zcode==169:
        # EBCDIC is assumed - ZCODE is EBCDIC code of either lower or
        # upper case 'Z'.
        if inta>=129 and inta<=137 or \
            inta>=145 and inta<=153 or \
            inta>=162 and inta<=169:
            inta = inta + 64
        if intb>=129 and intb<=137 or \
            intb>=145 and intb<=153 or \
            intb>=162 and intb<=169:
            intb = intb + 64
    elif zcode==218 or zcode==250:
        # ASCII is assumed, on Prime machines - ZCODE is the ASCII code
        # plus 128 of either lower or upper case 'Z'.
        if inta>=225 and inta<=250:
            inta = inta - 32
        if intb>=225 and intb<=250:
            intb = intb - 32
        
    return inta==intb

__all__.append('lsame')

################################################################################

def dconjg(x):
    """
    Get the complex conjugate of ``x``
    """
    assert isinstance(x, complex), 'Only complex numbers have a conjugate value'
    return x.conjugate()

__all__.append('dconjg')

################################################################################

def dble(x):
    """
    Convert ``x`` to a float
    """
    return 1.0*x.real

__all__.append('dble')

################################################################################

def dcomplx(x):
    """
    Convert ``x`` to a complex number.
    """
    return 1.0*x.real + 1.0*x.imag*1j
    
__all__.append('dcomplx')

################################################################################


    