import numpy
import math
import numpy
import random
from itertools import product
import math

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qBitStrings as qb

def groverR3():
    '''Assumes that n = 3. Returns -R, where R is Grover’s n-qbit gate for
    reflection across |rho>. Builds the gate from one- and two-qbit gates,
    rather than manually constructing the matrix.'''

    H = power(qc.h, 3)
    X = power(qc.x, 3)
    I = power(qc.i, 2)
    IH = tensor(I, qc.h)
    H_X = application(H, X)
    X_H = application(X, H)
    first = application(H_X, IH)
    second = application(IH, X_H)
    ccx = qc.ccNot
    half = application(first, ccx)
    R3 = application (half, second)

    return R3

def ccNot():
    '''Returns the three-qbit ccNOT (i.e., Toffoli) gate. The gate is
    implemented using five specific two-qbit gates and some SWAPs.'''

    CV = distant(qc.cv)
    CZ = tensor(qc.i, qc.cz)
    CU = tensor(qc.cu, qc.i)
    input = application(CZ, CV)
    input = application(input, input)
    ccnot = application(input, CU)

    return ccnot

def distant(gate):
    '''Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is
    n-qbit), performs swaps to insert one extra wire between the first qbit and
    the other n qbits. Returns an (n + 2)-qbit gate.'''

    swapIn = qg.tensor(qc.swap, qg.power(qc.i, int(math.log2(gate.shape[0])-1)))
    distant = application(application(swapIn, (tensor(qc.i, gate))), swapIn)

    return distant

def fourierRecursive(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.
    Computes T recursively rather than from the definition.'''

    R = fourierR(n)
    Q = fourierQ(n)
    S = fourierS(n)
    QR = application(Q, R)
    T = application(QR, S)

    return T

def fourierS(k):
    '''Computes the S component of the QFT'''
    layers = [] # matrix of every combination of SWAP and I
    S = []

    if k == 1:
        S = qc.i

    elif k == 2:
        S = qc.swap

    else:
        for i in range(k-1):
            if i == k-2:
                layer = tensor(qc.swap, power(qc.i, i))
                layers.append(layer)
            elif i == 0:
                layer = tensor(power(qc.i, k-2), qc.swap)
                layers.append(layer)
            elif len(layers) == k-1:
                 break
            else:
                layer = tensor(power(qc.i, k-i-2), qc.swap)
                layer = tensor(layer, power(qc.i, i))
                layers.append(layer)


        for layer in layers: # applies each combination SWAP and I
            if len(S) == 0:
                S = layer
            else:
                S = application(layer, S)
    return S

def fourierR(k): #HELP
    '''Computes the R component of the QFT'''

    if k == 1:
        R = qc.i

    else:
        r = fourierRecursive(k-1)
        R = qu.directSum(r, r)

    return R

def fourierD(k):
    '''Computes the D component of the QFT'''

    D = numpy.zeros((2**k, 2**k), dtype = complex)
    omega = math.e**((2*math.pi*1j)/(2**(k+1)))

    for y in range(2**k):
        D[y,y] = omega**y

    return D

def fourierQ(k):
    '''Computes the Q component of the QFT'''

    if k == 1:
        Q = fourier(1)
    else:
        I = power(qc.i, k - 1)
        D = fourierD(k - 1)
        top = numpy.concatenate((I, D),axis = 1)
        bottom = numpy.concatenate((I, -1 * D), axis = 1)
        Q = numpy.concatenate((top, bottom))
        Q = 1 / math.sqrt(2) * Q

    return Q

def fourier(n):
    '''Assumes n >= 1. Returns the n-qbit quantum Fourier transform gate T.'''

    T = numpy.zeros((2 ** n, 2 ** n), dtype = complex)
    omega = math.e ** ((2 * math.pi * 1j) / (2 ** n))
    for i in range(T.shape[1]):
        T[i, 0] = 1
        T[0, i] = 1
    for y in range(1, T.shape[1]):
        for x in range(1, T.shape[0]):
            pow = (y) * (x)
            T[y,x] = omega ** pow

    return numpy.multiply(T, 1 / math.sqrt(2 ** n))

def application(u, ketPsi):
    '''Assumes n >= 1. Applies the n-qbit gate U to the n-qbit state |psi>, returning the n-qbit state U |psi>.'''
    return numpy.matmul(u, ketPsi, dtype = complex)

def tensor(a, b):
    '''Assumes that n, m >= 1. Assumes that a is an n-qbit state and b is an
    m-qbit state, or that a is an n-qbit gate and b is an m-qbit gate. Returns
    the tensor product of a and b, which is an (n + m)-qbit gate or state.'''

    n = a.shape[0]
    m = b.shape[0]

    if len(a.shape) == 1: # a and b are states
        rows = numpy.zeros((n * m), dtype = complex)
        index = 0
        for i in range(int(n)):
            for k in range(int(m)):
                value = a[i] * b[k]
                rows[index] = value
                index += 1
        return rows

    else: # a and b are gates
        rows = numpy.zeros((n * m, n * m), dtype = complex)
        row = 0
        col = 0

        for i in range(int(n)):
            for k in range(int(m)):
                for j in range(int(n)):
                    for l in range(int(m)):
                        value = a[i][j] * b[k][l]
                        rows[row][col] = value
                        col += 1
                row += 1
                col = 0

        return rows

def function(n, m, f):
    '''Assumes n, m >= 1. Given a Python function f : {0, 1}^n -> {0, 1}^m.
    That is, f takes as input an n-bit string and produces as output an m-bit
    string, as defined in qBitStrings.py. Returns the corresponding
    (n + m)-qbit gate F.'''

    length = 2 ** (n + m)
    columns = numpy.zeros((length, length), dtype = complex)
    index = 0
    for a, b in product(qb.basis(n), qb.basis(m)):
        column = qb.vector(a + qb.addition(b, f(a)))
        columns[index] = column
        index += 1
    return numpy.column_stack(columns)

def power(stateOrGate, m):
    '''Assumes n >= 1. Given an n-qbit gate or state and m >= 1, returns the
    mth tensor power, which is an (n * m)-qbit gate or state. For the sake of
    time and memory, m should be small.'''
    input = stateOrGate
    output = stateOrGate
    while m > 1:
        output = tensor(output, input)
        m = m - 1
    return output

def distant(gate):
    '''Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is
    n-qbit), performs swaps to insert one extra wire between the first qbit and
    the other n qbits. Returns an (n + 2)-qbit gate.'''
    gate1 = tensor(qc.swap, qc.i)
    gate2 = tensor(qc.i, gate)
    gate3 = tensor(qc.swap, qc.i)
    basis = power(qc.i, len(gate) + 1)
    basis = application(gate1, basis)
    basis = application(gate2, basis)
    basis = application(gate3, basis)

### DEFINING SOME TESTS ###

def applicationTest():
    # These simple tests detect type errors but not much else.
    answer = application(qc.h, qc.ketMinus)
    if qu.equal(answer, qc.ket1, 0.000001):
        print("passed applicationTest first part")
    else:
        print("FAILED applicationTest first part")
        print("    H |-> = " + str(answer))
    ketPsi = qu.uniform(2)
    answer = application(qc.swap, application(qc.swap, ketPsi))
    if qu.equal(answer, ketPsi, 0.000001):
        print("passed applicationTest second part")
    else:
        print("FAILED applicationTest second part")
        print("    |psi> = " + str(ketPsi))
        print("    answer = " + str(answer))

def tensorTest():
    # Pick two gates and two states.
    u = qc.x
    v = qc.h
    ketChi = qu.uniform(1)
    ketOmega = qu.uniform(1)
    # Compute (U tensor V) (|chi> tensor |omega>) in two ways.
    a = tensor(application(u, ketChi), application(v, ketOmega))
    b = application(tensor(u, v), tensor(ketChi, ketOmega))
    # Compare.
    if qu.equal(a, b, 0.000001):
        print("passed tensorTest")
    else:
        print("FAILED tensorTest")
        print("    a = " + str(a))
        print("    b = " + str(b))

def functionTest(n, m):
    # 2^n times, randomly pick an m-bit string.
    values = [qb.string(m, random.randrange(0, 2**m)) for k in range(2**n)]
    # Define f by using those values as a look-up table.
    def f(alpha):
        a = qb.integer(alpha)
        return values[a]
    # Build the corresponding gate F.
    ff = function(n, m, f)
    # Helper functions --- necessary because of poor planning.
    def g(gamma):
        if gamma == 0:
            return qc.ket0
        else:
            return qc.ket1

    def ketFromBitString(alpha):
        ket = g(alpha[0])
        for gamma in alpha[1:]:
            ket = tensor(ket, g(gamma))
        return ket

    # Check 2^n - 1 values somewhat randomly.
    alphaStart = qb.string(n, random.randrange(0, 2**n))
    alpha = qb.next(alphaStart)
    while alpha != alphaStart:
    # Pick a single random beta to test against this alpha.
        beta = qb.string(m, random.randrange(0, 2**m))
        # Compute |alpha> tensor |beta + f(alpha)>.
        ketCorrect = ketFromBitString(alpha + qb.addition(beta, f(alpha)))
        # Compute F * (|alpha> tensor |beta>).
        ketAlpha = ketFromBitString(alpha)
        ketBeta = ketFromBitString(beta)
        ketAlleged = application(ff, tensor(ketAlpha, ketBeta))
        # Compare.
        if not qu.equal(ketCorrect, ketAlleged, 0.000001):
            print("failed functionTest")
            print(" alpha = " + str(alpha))
            print(" beta = " + str(beta))
            print(" ketCorrect = " + str(ketCorrect))
            print(" ketAlleged = " + str(ketAlleged))
            print(" and here’s F...")
            print(ff)
            return
        else:
            alpha = qb.next(alpha)
    print("passed functionTest")

def fourierTest(n):
    if n == 1:
        # Explicitly check the answer.
        t = fourier(1)
        if qu.equal(t, qc.h, 0.000001):
            print("passed fourierTest")
        else:
            print("failed fourierTest")
            print(" got T = ...")
            print(t)
    else:
        t = fourier(n)
        # Check the first row and column.
        const = pow(2, -n / 2) + 0j
        for j in range(2**n):
            if not qu.equal(t[0, j], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        for i in range(2**n):
            if not qu.equal(t[i, 0], const, 0.000001):
                print("failed fourierTest first part")
                print(" t = ")
                print(t)
                return
        print("passed fourierTest first part")
        # Check that T is unitary.
        tStar = numpy.conj(numpy.transpose(t))
        tStarT = numpy.matmul(tStar, t)
        id = numpy.identity(2**n, dtype=qc.one.dtype)
        if qu.equal(tStarT, id, 0.000001):
            print("passed fourierTest second part")
        else:
            print("failed fourierTest second part")
            print(" T^* T = ...")
            print(tStarT)

def fourierRecursiveTest(n):
    if qu.equal(fourierRecursive(n), fourier(n), .000001):
        print("passed fourierRecursiveTest")
    else:
        print("failed fourierRecursiveTest")

### RUNNING THE TESTS ###

def main():
    applicationTest()
    applicationTest()
    tensorTest()
    tensorTest()
    functionTest(2, 2)
    fourierTest(3)
    fourierRecursiveTest(3)

if __name__ == "__main__":
    main()
