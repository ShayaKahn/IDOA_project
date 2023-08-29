import numpy as np
cimport numpy as np

def f(double t, double[:] x, double[:] r, double[:] s, double[:, :] A, double delta):
    # RHS of the ODE
    cdef int i, p
    cdef int n = np.size(x)

    return np.array([r[i] * x[i] - s[i] * x[i] ** 2 + sum([A[i, p] * x[
           i] * x[p] for p in range(n) if p != i]) for i in range(n)])

def event(double t, double[:] x, double[:] r, double[:] s, double[:, :] A, double delta):
    # Event function for the ODE solver
    cdef double[:] fx = f(t, x, r, s, A, delta)
    cdef double max_fx = np.max(np.abs(fx))
    return max_fx - delta