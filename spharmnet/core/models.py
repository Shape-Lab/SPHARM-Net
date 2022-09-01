"""
July 2021

Ilwoo Lyu, ilwoolyu@unist.ac.kr
Seungbo Ha, mj0829@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Ulsan National Institute of Science and Technology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SHConv, SHT, ISHT
from ..lib.sphere import vertex_area, spharm_real
from ..lib.io import read_mesh


class SHConvBlock(nn.Module):
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear=None, fullband=True, bn=True):
        """
        SHConvBlock [1].
        The SHConvBlock performs SHT on a spherical signal followed by rotation-equivariant spectral convolution and ISHT.
        The module behaves like SHConv-FB by setting fullband=True, which supports constrained full-bandwidth convolution.
        The module otherwise allows only spectral convolution, which assumes spectral pooling.

        [1] Ha, Seungbo, and Ilwoo Lyu.
            "SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation."
            IEEE Transactions on Medical Imaging (2022).

        Parameters
        __________
        Y : 2D array, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.
        Y_inv : 2D array, shape = [n_vertex, (L+1)**2]
            Matrix form of harmonic basis.
        area : 1D array
            Area per vertex.
        in_ch : int
            # of input channels.
        out_ch : int
            # of output channels.
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        nonlinear : None or torch.nn.functional
            Non-linear activation. If not set, nn.Identity will be used.
        fullband : bool
            Full-bandwidth convolution.
        bn : bool
            batch normalization before non-linear activation.

        Notes
        _____
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, out_ch, n_vertex]
        """

        super().__init__()

        self.shconv = nn.Sequential(SHT(L, Y_inv, area), SHConv(in_ch, out_ch, L, interval), ISHT(Y))
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=not bn) if fullband else lambda _: 0
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False) if bn else nn.Identity()
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

    def forward(self, x):
        x = self.shconv(x) + self.impulse(x)
        x = self.bn(x)
        x = self.nonlinear(x)

        return x


class SPHARM_Net(nn.Module):
    def __init__(self, sphere, device, in_ch=3, n_class=32, C=128, L=80, D=3, interval=5, threads=1, verbose=False):
        """
        SPHARM-Net [1].
        In the encoding (decoding) phase, each block halves (doubles) the harmonic bandwidth L while doubling (halving)
        the number of channels C. The final block aggregates the learned information to infer parcellation labels.
        The SHConv block performs SHT on a spherical signal followed by rotation-equivariant spectral convolution and ISHT.
        The SHConv-FB block adds a scaled spherical signal to the SHConv block to support constrained full-bandwidth convolution.

        [1] Ha, Seungbo, and Ilwoo Lyu.
            "SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation."
            IEEE Transactions on Medical Imaging (2022).

        Parameters
        __________
        sphere : str
            Sphere mesh file. VTK (ASCII) < v4.0 or FreeSurfer format.
        device : torch.device
            Device indicator.
        in_ch : int
            # of input geometric features.
        n_class : int
            # of labels, i.e., the output layer size.
        C : int
            # of channels in the entry layer (see the paper for details).
        L : int
            Spectral bandwidth that supports individual component learning (see the paper for details).
        D : int
            Depth of encoding/decoding levels (see the paper for details).
        interval : int
            Interval of anchor points (see the paper for details).
        threads : int
            # of CPU threads for basis reconstruction. Useful if the unit sphere has dense tesselation.

        Notes
        _____
        In channel shape  : [batch, in_ch, n_vertex]
        Out channel shape : [batch, n_class, n_vertex]
        """

        super().__init__()

        L_in = L
        self.down = []
        self.up = []

        ch_inc = 2
        out_ch = C

        v, f = read_mesh(sphere)
        v = v.astype(float)
        area = vertex_area(v, f)
        Y = spharm_real(v, L, threads)

        area = torch.from_numpy(area).to(device=device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(device=device, dtype=torch.float32)

        Y_inv = Y.T
        nonlinear = F.relu

        # encoding
        for i in range(D):
            L = L_in // 2**i
            if verbose:
                print("Down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.down.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, i == 0))
            in_ch = out_ch
            out_ch *= ch_inc

        L //= 2
        out_ch //= ch_inc
        in_ch = out_ch
        if verbose:
            print("Bottom\t| C:{} -> {}\t| L:{}".format(in_ch, out_ch, L))
        self.down.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, False))

        # decoding
        for i in range(D - 1):
            L = L_in // 2 ** (D - 1 - i)
            in_ch = out_ch * 2
            out_ch //= ch_inc
            if verbose:
                print("Up {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.up.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, False))

        in_ch = out_ch * 2
        L *= 2
        if verbose:
            print("Up {}\t| C:{} -> {}\t| L:{}".format(D, in_ch, out_ch, L))
        self.up.append(SHConvBlock(Y, Y_inv, area, in_ch, out_ch, L, interval, nonlinear, True))
        if verbose:
            print("Final\t| C:{} -> {}\t| L:{}".format(out_ch, n_class, L))
        self.final = SHConvBlock(Y, Y_inv, area, out_ch, n_class, L_in, interval, None, True)

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_ = [self.down[0](x)]
        for i in range(len(self.down) - 1):
            x_.append(self.down[i + 1](x_[-1]))
        x = x_[-1]
        for i in range(len(self.up)):
            x = torch.cat([x, x_[-2 - i]], dim=1)
            x = self.up[i](x)
        x = self.final(x)

        return x
