import torch
import numpy as np

# code from https://github.com/GrishKate/accelerating_orthogonalization/blob/main/polynomials.py
np.set_printoptions(precision=20)


def get_polynomial(Ext):
    n = len(Ext) - 1
    M = np.zeros((n + 1, n + 1), dtype=np.float64)
    for i in range(n + 1):
        for j in range(n):
            M[i, j] = Ext[i] ** (2 * j + 1)
        M[i, n] = (-1) ** (i + 1)
    c = np.linalg.solve(M, np.ones(n + 1, dtype=np.float64))
    return c


def get_ext(c):
    n = len(c)
    coeffs = [(2 * j + 1) * c[j] for j in range(n)]
    rts = np.roots(coeffs[::-1])
    return np.sqrt(rts)


def remez_step(Ext):
    n = len(Ext) - 1
    c = get_polynomial(Ext)
    coeffs = [c[j // 2 - 1] if j % 2 == 0 else 0 for j in range(1, 2 * n + 1)]
    p = np.poly1d(coeffs[::-1])
    e = c[n].item()
    NewExt = np.concatenate(([Ext[0]], get_ext(c[:n]), [Ext[-1]]))
    f = lambda x: np.abs(p(x) - 1)
    newe = np.max(f(NewExt))
    return p, NewExt, e, newe


def remez(A, B, degree):
    # Remez algorithm for finding optimal polynomial approximating the unity function f==1
    # on the segment [A, B]
    n = (degree + 1) // 2
    Ext = np.linspace(B, A, n + 1, dtype=np.float64)
    p = np.poly1d([0])
    newe = 0
    for i in range(100):
        try:
            p, NewExt, e, newe = remez_step(Ext)
        except np.linalg.LinAlgError:
            return kovarik_formula(degree), 0  # if the segment converged to [A, B]=[1, 1]
        if newe < abs(e) + 1e-20:
            return p, newe
        Ext = np.array(NewExt, dtype=np.float64)
    return p, newe


def c_n_k(n, k):
    s = 1
    for i in range(n - k + 1, n + 1):
        s *= i
    for i in range(1, k + 1):
        s /= i
    return s


def kovarik_formula(degree):
    # generates polynomial of specified degree from the paper
    # Zdislav Kovarik, "SOME ITERATIVE METHODS FOR IMPROVING ORTHONORMALITY", 1970
    p = np.zeros(degree + 1)
    p[1] += 1
    a = 1
    for i in range(1, (degree + 1) // 2):
        for j in range(2 * (i - 1) + 1, 2 * i + 1):
            a *= j / 2
        a /= i ** 2
        sign = 1
        for k in range(0, i + 1):
            p[2 * k + 1] += a * sign * c_n_k(i, k)
            sign *= -1
    return np.poly1d(p[::-1])

def explicit3(A, B):
    # explicit formula for optimal 3-rd per polynomial on the segment [A, B]
    e = np.sqrt((A ** 2 + A * B + B ** 2) / 3)
    a = 2 / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    p = np.poly1d([-a, 0, a * (A ** 2 + A * B + B ** 2), 0])
    err = (2 * e ** 3 - A ** 2 * B - B ** 2 * A) / (2 * e ** 3 + A ** 2 * B + B ** 2 * A)
    return p, err


def find_left_bd(delta, B, degree):
    # find one optimal polynomial with high derivative at zero on the interval [0, B], which falls into [1-delta, 1+delta]
    # B is the right boundary of the interval
    # delta is the desired accuracy of approximation
    Al = 0.0
    Ar = B
    A = (Al + Ar) / 2
    p, f = remez(A, B, degree)
    while abs(delta - f) > 1e-15:
        if f < delta:
            Ar = (Ar + Al) / 2
        else:
            Al = (Al + Ar) / 2
        p, f = remez((Al + Ar) / 2, B, degree)
    return p, f, (Al + Ar) / 2


def delta_orthogonalization(n=1, degree=3, delta=0.3, B=1):
    # find composition of n polynomials of specified degree on the interval [0, B], which falls into [1-delta, 1+delta]
    # the derivative of composition at zero is maximized
    Al = 0.0
    Ar = B
    e = 100
    while abs(e - delta) > 1e-7:
        a, b = (Al + Ar) / 2, B
        lst = []
        for i in range(n):
            if degree == 3:
                Q, e = explicit3(a, b)
            else:
                Q, e = remez(a, b, degree)
            lst.append(Q)
            a, b = 1 - e, 1 + e
        if e < delta:
            Ar = (Ar + Al) / 2
        else:
            Al = (Al + Ar) / 2
    return lst, (Al + Ar) / 2


def cans_iteration(A, n, a, degree=3, preprocess=False, preprocess_iters=4, delta=0.99, **kwargs):
    A = A.clone()
    # CANS iteration for orthogonalization
    # a is the left boundary of the segment
    # n is the maximum number of iterations
    if A.shape[0] < A.shape[1]:
        A = A.T

    if degree == 3:
        A2 = A.T @ A
        A3 = A @ A2
        denom = torch.norm(A3, p='fro') ** (1/3) + 1e-7

        A2 /= denom ** 2
        A3 /= denom ** 3
    elif degree == 5:
        A2 = A.T @ A
        A3 = A @ A2
        A5 = A3 @ A2
        denom = torch.norm(A5, p='fro') ** (1/5) + 1e-7
        A2 /= denom ** 2
        A3 /= denom ** 3
        A5 /= denom ** 5
    else:
        raise NotImplementedError("Only degrees 3 and 5 are implemented")
    A = A / denom


    b = 1  # assume that matrix is normalized
    I = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    e = 10
    if preprocess:
        lst, _ = delta_orthogonalization(preprocess_iters, degree, delta)
        for i in range(preprocess_iters):
            if degree == 3:
                A = lst[i][1] * A + lst[i][3] * A3
            elif degree == 5:
                A = lst[i][1] * A + lst[i][3] * A3 + lst[i][5] * A5
            A2 = A.T @ A
            A3 = A @ A2
            if degree == 5:
                A5 = A3 @ A2
        a, b = 1 - delta, 1 + delta
    cnt = 0
    err = torch.norm(A2 - I) / torch.norm(I, p='fro')
    while cnt < n and (err > 1e-6):
        if cnt > 0:
            A3 = A @ A2
            if degree == 5:
                A5 = A3 @ A2
        if degree == 3:
            p, e = explicit3(a, b)
            a, b = 1 - e, 1 + e
            A = p[1] * A + p[3] * A3
        elif degree == 5:
            p, e = remez(a, b, degree)
            a, b = 1 - e, 1 + e
            A = p[1] * A + p[3] * A3 + p[5] * A5
            b *= 1.01  # for numerical stability
        A2 = A.T @ A
        err = torch.norm(A2 - I) / torch.norm(I, p='fro')
        cnt += 1
    return A
