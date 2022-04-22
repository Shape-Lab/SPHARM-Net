"""
July 2021

Seungbo Ha, mj0829@unist.ac.kr
Ilwoo Lyu, ilwoolyu@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Ulsan National Institute of Science and Technology
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree as KDTree
from scipy.sparse import coo_matrix


class TriangleSearch:
    def __init__(self, v, f):
        """
        Fast closest triangle search. This module is particularly useful for spherical tessellation.
        The module only works on spherical tessellation.

        Parameters
        __________
        v : 2D array, shape = [n_vertex, 3]
            3D coordinates of the unit sphere.
        f : 2D array, shape = [n_face, 3]
            Triangles of the unit sphere.
        """

        self.v = v
        self.f = f

        centroid = self.v[self.f[:, 0], :] + self.v[self.f[:, 1], :] + self.v[self.f[:, 2], :]
        centroid /= np.linalg.norm(centroid, axis=1, keepdims=True)

        self.MD = KDTree(centroid)

        a = self.v[self.f[:, 0], :]
        b = self.v[self.f[:, 1], :]
        c = self.v[self.f[:, 2], :]
        normal = np.cross(a - b, a - c)
        # Exception: when normal == [0, 0, 0]
        zero_normal = (normal == 0).all(axis=1)
        normal[zero_normal] = a[zero_normal]
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)

        self.face_normal = normal

        self.r = np.multiply(self.v[self.f[:, 0], :], self.face_normal).sum(1)

    def barycentric(self, v, f, p):
        """
        Barycentric coefficients of p inside triangles f.
        """

        a = v[f[:, 0], :]
        b = v[f[:, 1], :]
        c = v[f[:, 2], :]

        n = np.cross(b - a, c - a)
        abc = np.linalg.norm(n, axis=1, keepdims=True)
        n /= abc
        u = (np.cross(b - p, c - p) * n).sum(-1, keepdims=True) / abc
        v = (np.cross(c - p, a - p) * n).sum(-1, keepdims=True) / abc
        w = 1 - u - v

        return np.hstack([u, v, w])

    def query(self, query_v, tol=-1e-6):
        """
        Find triangle that contains query_v and then compute their barycentric coefficients.
        """

        q = query_v / np.linalg.norm(query_v, axis=1, keepdims=True)

        q_num = q.shape[0]
        qID = np.arange(q_num)
        which_tri = np.zeros(q_num, dtype=np.int)
        bary_list = np.zeros((q_num, 3))

        k = 1
        while qID.size > 0:
            query = q[qID, :]
            _, kk = self.MD.query(query, k)

            if k > 1:
                kk = kk[:, k - 1]  # k-th closest triangle

            q_proj = np.multiply(query.transpose(1, 0), self.r[kk] / np.multiply(query, self.face_normal[kk]).sum(1))
            q_proj = q_proj.transpose(1, 0)
            b = self.barycentric(self.v, self.f[kk], q_proj)
            valid = (b >= tol).sum(1) == 3
            which_tri[qID[valid]] = kk[valid]
            bary_list[qID[valid]] = b[valid]
            qID = np.delete(qID, valid)

            k += 1

        return which_tri, bary_list


def vertex_area(v, f):
    """
    Vertex-wise area. The area is approximated by a third of neighborhood triangle areas.

    Parameters
    __________
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the unit sphere.
    f : 2D array, shape = [n_face, 3]
        Triangles of the unit sphere.

    Returns
    _______
    area : 1D array, shape = [n_vertex]
        Area per vertex.
    """

    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    a = u[f[:, 0], :]
    b = u[f[:, 1], :]
    c = u[f[:, 2], :]

    normal = np.cross(a - b, a - c)
    area = np.linalg.norm(normal, axis=1, keepdims=True)
    area = coo_matrix(
        (np.repeat(area, 3), (np.repeat(np.arange(f.shape[0]), 3), f.flatten())), shape=(f.shape[0], v.shape[0])
    )
    area = np.squeeze(np.asarray(coo_matrix.sum(area, axis=0))) / 6

    return area


def legendre(n, x, tol, tstart):
    """
    The Schmidt semi-normalized associated polynomials.

    Parameters
    __________
    n : int
        degree of spherical harmonics.
    x : 1D array, shape = [n_vertex]
        List of cosine values.
    tol : float
        sqrt(tiny).
    tstart: float
        eps.

    Returns
    _______
    Y : 2D array, shape = [n + 1, n_vertex]
        Schmidt semi-normalized associated polynomials.

    Notes
    _____
    MATLAB: https://www.mathworks.com/help/matlab/ref/legendre.html
    """

    Y = np.zeros(shape=(n + 1, len(x)))

    if n == 0:
        Y[0] = 1
    elif n == 1:
        Y[0] = x
        Y[1] = np.sqrt(1.0 - x * x)
    else:
        factor = np.sqrt(1.0 - x * x)
        rootn = [np.sqrt(i) for i in range(2 * n + 1)]
        pole = np.where(factor == 0)[0]
        factor[pole] = 1
        twocot = -2 * x / factor
        sn = np.power(-factor, n)

        Y[n, :] = np.sqrt(np.prod(1.0 - 1.0 / (2 * np.arange(1, n + 1)))) * sn
        Y[n - 1, :] = Y[n, :] * twocot * n / rootn[2 * n]
        for m in range(n - 2, -1, -1):
            Y[m] = (Y[m + 1] * twocot * (m + 1) - Y[m + 2] * rootn[n + m + 2] * rootn[n - m - 1]) / (
                rootn[n + m + 1] * rootn[n - m]
            )

        idx = np.where(np.absolute(sn) < tol)[0]
        v = 9.2 - np.log(tol) / (n * factor[idx])
        w = 1 / np.log(v)
        m1 = 1 + n * factor[idx] * v * w * (1.0058 + w * (3.819 - w * 12.173))
        m1 = np.minimum(n, np.floor(m1)).astype("int")

        Y[:, idx] = 0
        m1_unique = np.unique(m1)
        for mm1 in m1_unique:
            k = np.where(m1 == mm1)[0]
            col = idx[k]

            Y[mm1 - 1, col[x[col] < 0]] = np.sign((n + 1) % 2 - 0.5) * tstart
            Y[mm1 - 1, col[x[col] >= 0]] = np.sign(mm1 % 2 - 0.5) * tstart
            for m in range(mm1 - 2, -1, -1):
                Y[m, col] = (
                    Y[m + 1, col] * twocot[col] * (m + 1) - Y[m + 2, col] * rootn[n + m + 2] * rootn[n - m - 1]
                ) / (rootn[n + m + 1] * rootn[n - m])
            sumsq = tol + np.sum(Y[: mm1 - 2, col] * Y[: mm1 - 2, col], axis=0)
            Y[:mm1, col] /= np.sqrt(2 * sumsq - Y[0, col] * Y[0, col])

        Y[1:, pole] = 0
        Y[0, pole] = np.power(x[pole], n)
        Y[1:None] *= rootn[2]
        Y[1:None:2] *= -1

    return Y


def spharm_real(x, l, threads=None, lbase=0):
    """
    A set of real spherical harmonic bases using the Schmidt semi-normalized associated polynomials.
    The spherical harmonics will be generated from lbase to l.

    Parameters
    __________
    x : 2D array, shape = [n_vertex, 3]
        Array of 3D coordinates of the unit sphere.
    l : int
        Degree of spherical harmonics.
    lbase : int
        Base degree of spherical harmonics.
    threads : int
        Non-negative number of threads for parallel computing powered by joblib.

    Returns
    _______
    Y : 2D array, shape = [(l - lbase + 1) ** 2, n_vertex]
        Real spherical harmonic bases.
    """

    def cart2sph(x):
        azimuth = np.arctan2(x[:, 1], x[:, 0])
        elevation = np.pi / 2 - np.arctan2(x[:, 2], np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))

        return azimuth, elevation

    def basis(Y, theta, lfrom, lto, lbase, c, s, tol, tstart):
        c2 = np.cos(-theta)
        for l in range(lfrom, lto, 1):
            center = (l + 1) * (l + 1) - l - 1 - lbase * lbase
            Y[center : center + l + 1] = legendre(l, c2, tol, tstart) * np.sqrt((2 * l + 1) / (4 * np.pi))
        if lfrom == 0:
            lfrom = 1
        for l in range(lfrom, lto, 1):
            center = (l + 1) * (l + 1) - l - 1 - lbase * lbase
            Y[center - 1 : center - l - 1 : -1] = Y[center + 1 : center + l + 1] * s[0:l]
            Y[center + 1 : center + l + 1] *= c[0:l]

    if lbase < 0:
        lbase = 0

    phi, theta = cart2sph(x)

    size = (l + 1) * (l + 1) - lbase * lbase
    Y = np.zeros(shape=(size, len(x)))

    m = np.arange(1, l + 1)
    deg = m[:, None] * phi[None, :]
    c = np.cos(deg)
    s = np.sin(deg)

    tol = np.sqrt(np.finfo(x.dtype).tiny)
    tstart = np.finfo(x.dtype).eps

    if threads is not None and threads > 1:
        Parallel(n_jobs=threads, require="sharedmem")(
            delayed(basis)(Y, theta, n, n + 1, lbase, c, s, tol, tstart) for n in range(lbase, l + 1, 1)
        )
    else:
        basis(Y, theta, lbase, l + 1, lbase, c, s, tol, tstart)

    return Y
