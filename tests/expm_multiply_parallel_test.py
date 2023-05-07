from __future__ import print_function, division

import sys, os

# qspin_path = os.path.join(os.getcwd(), "../")
# sys.path.insert(0, qspin_path)

# set number of OpenMP threads to run in parallel
# uncomment this line if omp error occurs on OSX for python 3
os.environ["KMP_DUPLICATE_LIB_OK"] = "true"
os.environ["OMP_PROC_BIND"] = "true"
# set number of OpenMP threads to run in parallel
os.environ["OMP_NUM_THREADS"] = "16"
# set number of MKL threads to run in parallel
os.environ["MKL_NUM_THREADS"] = "16"

# print(os.environ["OMP_NUM_THREADS"])
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import random, eye
import numpy as np
import time


def test_imag_time(L=22, seed=0):
    np.random.seed(seed)

    basis = spin_basis_1d(L, m=0, kblock=0, pblock=1, zblock=1)
    print("basis size: {}".format(basis.Ns))

    J = [[1.0, i, (i + 1) % L] for i in range(L)]
    static = [["xx", J], ["yy", J], ["zz", J]]
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)

    (E,), psi_gs = H.eigsh(k=1, which="SA")

    psi_gs = psi_gs.ravel()

    A = -(H.tocsr() - E * eye(H.Ns, format="csr", dtype=np.float64))

    U = expm_multiply_parallel(A)

    v1 = np.random.uniform(-1, 1, size=H.Ns)
    v1 /= np.linalg.norm(v1)

    v2 = v1.copy()

    ntest = 100
    t1_tot = 0
    t2_tot = 0
    for i in range(ntest):
        t2 = time.time()
        v2 = U.dot(v2)
        v2 /= np.linalg.norm(v2)
        t2 = time.time() - t2
        t2_tot += t2

        t1 = time.time()
        v1 = expm_multiply(A, v1)
        v1 /= np.linalg.norm(v1)
        t1 = time.time() - t1
        t1_tot += t1

        if np.abs(H.expt_value(v2) - E) < 1e-15:
            break  #

        i += 1

    print("imag time test time: {}s".format(t1_tot / ntest))
    print("imag time parallel test time: {}s".format(t2_tot / ntest))
    np.testing.assert_allclose(
        v1,
        v2,
        rtol=0,
        atol=1e-15,
        err_msg="imaginary time test failed, seed {:d}".format(seed),
    )


def test_ramdom_matrix(N=5000, ntest=10, seed=0):
    np.random.seed(seed)
    
    i = 0
    t1_tot = 0
    t2_tot = 0
    while i < ntest:
        print("testing random matrix {}".format(i + 1))
        A = random(N, N, density=np.log(N) / N) + 1j * random(
            N, N, density=np.log(N) / N
        )
        A = A.tocsr()

        v = np.random.normal(0, 1, size=N) + 1j * np.random.normal(0, 1, size=N)
        v /= np.linalg.norm(v)

        t1 = time.time()
        v1 = expm_multiply(A, v)
        t2 = time.time()
        v2 = expm_multiply_parallel(A).dot(v)
        t3 = time.time()
        t1_tot += t2 - t1
        t2_tot += t3 - t2

        np.testing.assert_allclose(
            v1,
            v2,
            rtol=0,
            atol=1e-15,
            err_msg="random matrix test failed, seed {:d}".format(seed),
        )
        i += 1

    print("random matrix test time: {}s".format(t1_tot / ntest))
    print("random matrix parallel test time: {}s".format(t2_tot / ntest))

def test_ramdom_int_matrix(N=5000, ntest=10, seed=0):
    np.random.seed(seed)
    i = 0
    t1_tot = 0
    t2_tot = 0
    while i < ntest:
        print("testing random integer matrix {}".format(i + 1))
        data_rvs = lambda n: np.random.randint(-100, 100, size=n, dtype=np.int8)
        A = random(N, N, density=np.log(N) / N, data_rvs=data_rvs, dtype=np.int8)
        A = A.tocsr()

        v = np.random.normal(0, 1, size=N) + 1j * np.random.normal(0, 1, size=N)
        v /= np.linalg.norm(v)

        t1 = time.time()
        v1 = expm_multiply(-0.01j * A, v)
        t2 = time.time()
        v2 = expm_multiply_parallel(A, a=-0.01j, dtype=np.complex128).dot(v)
        t3 = time.time()

        np.testing.assert_allclose(
            v1,
            v2,
            rtol=0,
            atol=1e-15,
            err_msg="random matrix test failed, seed {:d}".format(seed),
        )
        i += 1

    print("random int matrix test time: {}s".format(t1_tot / ntest))
    print("random int matrix parallel test time: {}s".format(t2_tot / ntest))

test_imag_time()
test_ramdom_matrix()
test_ramdom_int_matrix()
print("expm_multiply_parallel tests passed!")
