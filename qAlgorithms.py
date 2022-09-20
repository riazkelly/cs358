
import random
import numpy
import math

import qConstants as qc
import qUtilities as qu
import qGates as qg
import qMeasurement as qm
import qBitStrings as qb

def grover(n, k, f):
    '''Assumes n >= 1, k >= 1. Assumes that k is small compared to 2^n.
    Implements the Grover core subroutine. The F parameter is an (n + 1)-qbit
    gate representing a function f : {0, 1}^n -> {0, 1} such that
    SUM_alpha f(alpha) = k. Returns a list or tuple of n classical one-qbit
    states (either |0> or |1>), such that the corresponding n-bit string delta
    usually satisfies f(delta) = 1.'''

    input = qg.tensor(qg.power(qc.ket0, n), qc.ket1)
    input = qg.application(qg.power(qc.h, n + 1), input)
    R = numpy.zeros((2 ** n, 2 ** n), dtype = complex)

    for i in range(len(R)):
        for j in range(len(R)):
            R[i][j] = 1 / (2 ** n)
    gateR = qg.tensor(R, qc.i)
    t = numpy.arcsin(1 / (2 ** (n/2)))
    l = int((math.pi / (4 * t)) - 1/2)

    while l > 0:
        input = qg.application(f, input)
        input = qg.application(gateR, input)
        l = l - 1
    delta = []

    while len(input) > 2:
        measure = qm.first(input)
        input = measure[1]
        delta.append(measure[0])

    return delta

def shor(n, f):
    '''Assumes n >= 1. Given an (n + n)-qbit gate F representing a function
    f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list or tuple
    of n classical one-qbit states (|0> or |1>) corresponding to the output of
    Shor’s quantum circuit.'''

    input = qg.power(qc.ket0, 2*n)
    gate1 = qg.tensor(qg.power(qc.h, n), qg.power(qc.i, n))
    input = qg.application(gate1, input)
    input = qg.application(f, input)

    while len(input) > (2 ** n):
        input = qm.last(input)[0]

    input = qg.application(qg.fourier(n), input)
    delta = []

    while len(input) > 1:
        measure = qm.first(input)
        input = measure[1]
        delta.append(measure[0])

    return delta

def simon(n, f):
    '''The inputs are an integer n >= 2 and an (n + (n - 1))-qbit gate F
    representing a function f: {0, 1}^n -> {0, 1}^(n - 1) hiding an n-bit
    string delta as in the Simon (1994) problem. Returns a list or tuple of n
    classical one-qbit states (each |0> or |1>) corresponding to a uniformly
    random bit string gamma that is perpendicular to delta.'''

    input = qg.power(qc.ket0, (n + (n - 1)))
    gate1 = qg.tensor(qg.power(qc.h, n), qg.power(qc.i, n - 1))
    gate2 = qg.power(qc.h, n)
    input = qg.application(gate1, input)
    input = qg.application(f, input)

    while len(input) > (2 ** n):
        input = qm.last(input)[0]
    input = qg.application(gate2, input)
    gamma = []

    while len(input) > 1:
        measure = qm.first(input)
        input = measure[1]
        gamma.append(measure[0])

    return gamma

def bennett():
    '''Runs one iteration of the core algorithm of Bennett (1992).
    Returns a tuple of three items --- |alpha>, |beta>, |gamma> --- each of which is either |0> or |1>.'''
    alpha = []
    beta = []
    gamma = []
    rand1 = random.random()

    if rand1 < .5:
        alpha = qc.ket0
    else:
        alpha = qc.ket1

    rand2 = random.random()
    if rand2 < .5:
        beta = qc.ket0
    else:
        beta = qc.ket1

    rand3 = random.random()
    if numpy.array_equal(alpha, qc.ket0) and numpy.array_equal(beta, qc.ket0):
        gamma = qc.ket0
    elif numpy.array_equal(alpha, qc.ket0) and numpy.array_equal(beta, qc.ket1):
        if rand3 < .5:
            gamma = qc.ket0
        else:
            gamma = qc.ket1

    elif numpy.array_equal(alpha, qc.ket1) and numpy.array_equal(beta, qc.ket1):
        gamma = qc.ket0
    else:
        if rand3 < .5:
            gamma = qc.ket0
        else:
            gamma = qc.ket1

    return (alpha, beta, gamma)

def deutsch(f):
    '''Implements the algorithm of Deutsch (1985).
    That is, given a two-qbit gate F representing a function f : {0, 1} -> {0, 1},
    returns |1> if f is constant, and |0> if f is not constant.'''

    input = qg.tensor(qc.ket1, qc.ket1)
    doubleH = qg.tensor(qc.h, qc.h)
    input = qg.application(doubleH, input)
    input = qg.application(f, input)
    input = qg.application(doubleH, input)
    output = qm.first(input)[0]

    return output

def bernsteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate F representing a function
    f : {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown delta
    in {0, 1}^n, returns the list or tuple of n classical one-qbit states (each
    |0> or |1>) corresponding to delta.'''

    input = qg.tensor(qg.power(qc.ket0, n), qc.ket1)
    H = qg.power(qc.h, n + 1)
    output = qg.application(H, input)
    output = qg.application(f, output)
    output = qg.application(H, output)
    delta = []
    while len(output) > 2:
        measure = qm.first(output)
        delta.append(measure[0])
        output = measure[1]
    return delta


### DEFINING SOME TESTS ###

def bennettTest(m):
    # Runs Bennett's core algorithm m times.
    trueSucc = 0
    trueFail = 0
    falseSucc = 0
    falseFail = 0
    for i in range(m):
        result = bennett()
        if qu.equal(result[2], qc.ket1, 0.000001):
            if qu.equal(result[0], result[1], 0.000001):
                falseSucc += 1
            else:
                trueSucc += 1
        else:
            if qu.equal(result[0], result[1], 0.000001):
                trueFail += 1
            else:
                falseFail += 1
    print("check bennettTest for false success frequency exactly 0")
    print("    false success frequency = ", str(falseSucc / m))
    print("check bennettTest for true success frequency about 0.25")
    print("    true success frequency = ", str(trueSucc / m))
    print("check bennettTest for false failure frequency about 0.25")
    print("    false failure frequency = ", str(falseFail / m))
    print("check bennettTest for true failure frequency about 0.5")
    print("    true failure frequency = ", str(trueFail / m))

def deutschTest():
    def fNot(x):
        return (1 - x[0],)
    resultNot = deutsch(qg.function(1, 1, fNot))
    if qu.equal(resultNot, qc.ket0, 0.000001):
        print("passed deutschTest first part")
    else:
        print("failed deutschTest first part")
        print("    result = " + str(resultNot))
    def fId(x):
        return x
    resultId = deutsch(qg.function(1, 1, fId))
    if qu.equal(resultId, qc.ket0, 0.000001):
        print("passed deutschTest second part")
    else:
        print("failed deutschTest second part")
        print("    result = " + str(resultId))
    def fZero(x):
        return (0,)
    resultZero = deutsch(qg.function(1, 1, fZero))
    if qu.equal(resultZero, qc.ket1, 0.000001):
        print("passed deutschTest third part")
    else:
        print("failed deutschTest third part")
        print("    result = " + str(resultZero))
    def fOne(x):
        return (1,)
    resultOne = deutsch(qg.function(1, 1, fOne))
    if qu.equal(resultOne, qc.ket1, 0.000001):
        print("passed deutschTest fourth part")
    else:
        print("failed deutschTest fourth part")
        print("    result = " + str(resultOne))

def bernsteinVaziraniTest(n):
    delta = qb.string(n, random.randrange(0, 2**n))
    def f(s):
        return (qb.dot(s, delta),)
    gate = qg.function(n, 1, f)
    qbits = bernsteinVazirani(n, gate)
    bits = tuple(map(qu.bitValue, qbits))
    diff = qb.addition(delta, bits)
    if diff == n * (0,):
        print("passed bernsteinVaziraniTest")
    else:
        print("failed bernsteinVaziraniTest")
        print(" delta = " + str(delta))

def simonTest(n):
    # Pick a non-zero delta uniformly randomly.
    delta = qb.string(n, random.randrange(1, 2**n))
    # Build a certain matrix M.
    k = 0
    while delta[k] == 0:
        k += 1
    m = numpy.identity(n, dtype=int)
    m[:, k] = delta
    mInv = m
    # This f is a linear map with kernel {0, delta}. So it’s a valid example.
    def f(s):
        full = numpy.dot(mInv, s) % 2
        full = tuple([full[i] for i in range(len(full))])
        return full[:k] + full[k + 1:]
    gate = qg.function(n, n - 1, f)
    '''End of Josh's Code'''

    goal_matrix = [] # matrix of linearly independent gammas

    while len(goal_matrix) < n - 1:
        test_matrix = []
        for element in goal_matrix:
            test_matrix.append(element)
        zero_gamma = () # a zero vector that is used to test if gamma is linearly independent

        output = simon(n, gate)
        gamma = ()
        for i in range(len(output)):
            zero_gamma = zero_gamma + (0,)
            gamma = gamma + qb.bitString(output[i])

        test_matrix.append(gamma)
        test_matrix = qb.reduction(test_matrix)
        goal_matrix.append(zero_gamma)

        if test_matrix != goal_matrix: # testing if the gamma is linearly independent
            goal_matrix = []
            for element in test_matrix:
                goal_matrix.append(element)
        else:
            goal_matrix.pop(len(goal_matrix) - 1)

    location = len(goal_matrix) # column that is linearly dependent

    for i in range(len(goal_matrix)):
        if goal_matrix[i][i] != 1:
            location = i
            break

    prediction = ()

    for i in range(len(goal_matrix)):
        if i == location:
            prediction = prediction + (1,)
        elif goal_matrix[i][location] == 1:
            prediction = prediction + (1,)
        else:
            prediction = prediction + (0,)

    if location == len(goal_matrix):
        prediction = prediction + (1,)
    else:
        prediction = prediction + (0,)

    '''Start of Josh's Code'''
    pass
    if delta == prediction:
        print("passed simonTest")
    else:
        print("failed simonTest")
        print(" delta = " + str(delta))
        print(" prediction = " + str(prediction))

def get_b(n, F):
    '''Helper function that interprets output of Shor into an integer'''
    output = shor(n, F)
    delta = ()
    for element in output:
        delta = delta + qb.bitString(element)
    b = qb.integer(delta)
    return b

def shorTest(n, m):
    '''Assumes n >= 4 and that 2^n >= m^2'''

    k = m
    while math.gcd(k, m) != 1:
        k = random.randint(2, m - 1)

    # Brute forcing p
    l = 1
    while qu.powerMod(k, l, m) != 1:
        l += 1
    brute_force_p = l

    # New Shor Algorithm
    def f(bitString):
        l = qb.integer(bitString)
        return qb.string(n, qu.powerMod(k, l, m))
    F = qg.function(n, n, f)
    p = 0
    b = 0
    while p == 0 :
        d = m
        d_prime = m

        while d >= m:
            while b == 0:
                b = get_b(n, F)
            x_not = b / (2 ** n)

            cANDd = qu.continuedFraction(n, m, x_not)
            c = cANDd[0]
            d = cANDd[1]

            while d >= m and abs((b / (2 ** n)) - (c / d)) <= (1 / (2 ** (n + 1))):
                cANDd = qu.continuedFraction(n, m, x_not)
                c = cANDd[0]
                d = cANDd[1]
        if (qu.powerMod(k, d, m)) == 1:
            p = d
            break

        while d_prime >= m:
            while b == 0:
                b = get_b(n, F)
            x_not = b / (2 ** n)

            cANDd = qu.continuedFraction(n, m, x_not)
            c_prime = cANDd[0]
            d_prime = cANDd[1]

            while d_prime >= m and abs((b / (2 ** n)) - (c_prime / d_prime)) <= (1 / (2 ** (n + 1))):
                cANDd = qu.continuedFraction(n, m, x_not)
                c_prime = cANDd[0]
                d_prime = cANDd[1]
        if (qu.powerMod(k, d_prime, m)) == 1:
            p = d_prime
            break

        lcm = (d * d_prime) / (math.gcd(d, d_prime))
        if qu.powerMod(k, lcm, m) == 1:
            p = lcm
            break

    if p == brute_force_p:
        print("Passed shortest... STG LFG")
    else:
        print("Failed shortest")
        print("p = " + str(brute_force_p))
        print("but shor said p = " + str(p))

def groverTest(n, k):
# Pick k distinct deltas uniformly randomly.
    deltas = []
    while len(deltas) < k:
        delta = qb.string(n, random.randrange(0, 2**n))
        if not delta in deltas:
            deltas.append(delta)
    # Prepare the F gate.
    def f(alpha):
        for delta in deltas:
            if alpha == delta:
                return (1,)
        return (0,)
    fGate = qg.function(n, 1, f)
    # Run Grover’s algorithm up to 10 times.
    qbits = grover(n, k, fGate)
    bits = tuple(map(qu.bitValue, qbits))
    j = 1
    while (not bits in deltas) and (j < 10):
        qbits = grover(n, k, fGate)
        bits = tuple(map(qu.bitValue, qbits))
        j += 1
    if bits in deltas:
        print("passed groverTest in " + str(j) + " tries")
    else:
        print("failed groverTest")
        print(" exceeded 10 tries")
        print(" prediction = " + str(bits))
        print(" deltas = " + str(deltas))

### RUNNING THE TESTS ###

def main():
    bennettTest(10000)
    deutschTest()
    bernsteinVaziraniTest(4)
    simonTest(4)
    shorTest(4, 5)
    groverTest(6, 3)

if __name__ == "__main__":
    main()
