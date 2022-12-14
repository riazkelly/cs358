
import qConstants as qc
import qGates as qg
import qUtilities as qu
from itertools import product

# We represent an n-bit string --- that is, an element of {0, 1}^n --- in Python as a tuple of 0s and 1s.

def basis(n):
    '''Returns array of the first 2^n binary numbers'''
    bitStrings = []
    for x in range(2 ** n):
        bitString = string(n, x)
        bitStrings.append(bitString)
    return bitStrings

def vector(bitString):
    '''Converts a bitstring to a vector'''
    for i in range(len(bitString)):
        if i == 0:
            if bitString[i] == 0:
                vec = qc.ket0
            else:
                vec = qc.ket1
        elif bitString[i] == 0:
            vec = qg.tensor(vec, qc.ket0)
        else:
            vec = qg.tensor(vec, qc.ket1)
    return vec

def bitString(vec):
    '''Converts a classical state to a bitstring'''
    if qu.equal(vec, qc.ket0, 0.000001):
        return (0,)
    else:
        return (1,)

def string(n, m):
    '''Converts a non-negative Python integer m to its corresponding bit string.
    As necessary, pads with leading 0s to bring the number of bits up to n.'''
    s = ()
    while m >= 1:
        s = (m % 2,) + s
        m = m // 2
    s = (n - len(s)) * (0,) + s
    return s

def integer(s):
    '''Converts a bit string to its corresponding non-negative Python integer.'''
    m = 0
    for k in range(len(s)):
        m = 2 * m + s[k]
    return m

def next(s):
    '''Given an n-bit string, returns the next n-bit string. The order is lexicographic, except that there is a string after 1...1, namely 0...0.'''
    k = len(s) - 1
    while k >= 0 and s[k] == 1:
        k -= 1
    if k < 0:
        return len(s) * (0,)
    else:
        return s[:k] + (1,) + (len(s) - k - 1) * (0,)

def addition(s, t):
    '''Returns the mod-2 sum of two n-bit strings.'''
    return tuple([(s[i] + t[i]) % 2 for i in range(len(s))])

def dot(s, t):
    '''Returns the mod-2 dot product of two n-bit strings.'''
    return sum([s[i] * t[i] for i in range(len(s))]) % 2

def reduction(a):
    '''A is a list of m >= 1 bit strings of equal dimension n >= 1.
    In other words, A is a non-empty m x n binary matrix.
    Returns the reduced row-echelon form of A. A itself is left unaltered.'''
    b = a.copy()
    m = len(b)
    n = len(b[0])
    rank = 0
    for j in range(n):
        # Try to swap two rows to make b[rank, j] a leading 1.
        i = rank
        while i < m and b[i][j] == 0:
            i += 1
        if i != m:
            # Perform the swap.
            temp = b[i]
            b[i] = b[rank]
            b[rank] = temp
            # Reduce all leading 1s below the one we just made.
            for i in range(rank + 1, m):
                if b[i][j] == 1:
                    b[i] = addition(b[i], b[rank])
            rank += 1
    for j in range(n - 1, -1, -1):
        # Try to find the leading 1 in column j.
        i = m - 1
        while i >= 0 and b[i][j] != 1:
            i -= 1
        if i >= 0:
            # Use the leading 1 at b[i, j] to reduce 1s above it.
            for k in range(i):
                if b[k][j] == 1:
                    b[k] = addition(b[k], b[i])
    return b
