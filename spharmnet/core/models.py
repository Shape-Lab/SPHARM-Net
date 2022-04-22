"""
July 2021

Seungbo Ha, mj0829@unist.ac.kr
Ilwoo Lyu, ilwoolyu@unist.ac.kr

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


class Down(nn.Module):
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True):
        super().__init__()

        self.shconv = nn.Sequential(
            SHT(L, Y_inv, area),
            SHConv(in_ch, out_ch, L, interval),
            ISHT(Y),
        )
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=False) if fullband else None
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        x1 = self.shconv(x)
        if self.impulse is not None:
            x2 = self.impulse(x)
            x = x1 + x2
        else:
            x = x1
        x = self.bn(x)
        x = F.relu(x)

        return x


class Up(nn.Module):
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval, fullband=True):
        super().__init__()

        self.shconv = nn.Sequential(
            SHT(L, Y_inv, area),
            SHConv(in_ch, out_ch, L, interval),
            ISHT(Y),
        )
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=False) if fullband else None
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x, x_append):
        x = torch.cat([x, x_append], dim=1)
        x1 = self.shconv(x)
        if self.impulse is not None:
            x2 = self.impulse(x)
            x = x1 + x2
        else:
            x = x1
        x = self.bn(x)
        x = F.relu(x)

        return x


class Final(nn.Module):
    def __init__(self, Y, Y_inv, area, in_ch, out_ch, L, interval):
        super().__init__()

        self.shconv = nn.Sequential(
            SHT(L, Y_inv, area),
            SHConv(in_ch, out_ch, L, interval),
            ISHT(Y),
        )
        self.impulse = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch, momentum=0.1, affine=True, track_running_stats=False)

    def forward(self, x):
        x1 = self.shconv(x)
        x2 = self.impulse(x)
        x = x1 + x2
        x = self.bn(x)

        return x


class SPHARM_Net(nn.Module):
    def __init__(self, sphere, device, in_ch=3, n_class=32, C=128, L=80, D=3, interval=5, threads=1, verbose=False):
        """
        SPHARM-Net.
        In the encoding (decoding) phase, each block halves (doubles) the harmonic bandwidth L while doubling (halving)
        the number of channels C. The final block aggregates the learned information to infer parcellation labels.
        The SHConv block performs SHT on a spherical signal followed by rotation-equivariant spectral convolution and ISHT.
        The SHConv-FB block adds a scaled spherical signal to the SHConv block to support constrained full-bandwidth convolution.

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
        threads: int
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

        # encoding
        for i in range(D):
            L = L_in // 2 ** i
            if verbose:
                print("Down {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, i == 0))
            in_ch = out_ch
            out_ch *= ch_inc

        L //= 2
        out_ch //= ch_inc
        in_ch = out_ch
        if verbose:
            print("Bottom\t| C:{} -> {}\t| L:{}".format(in_ch, out_ch, L))
        self.down.append(Down(Y, Y_inv, area, in_ch, out_ch, L, interval, False))

        # decoding
        for i in range(D - 1):
            L = L_in // 2 ** (D - 1 - i)
            in_ch = out_ch * 2
            out_ch //= ch_inc
            if verbose:
                print("Up {}\t| C:{} -> {}\t| L:{}".format(i + 1, in_ch, out_ch, L))
            self.up.append(Up(Y, Y_inv, area, in_ch, out_ch, L, interval, False))

        in_ch = out_ch * 2
        L *= 2
        if verbose:
            print("Up {}\t| C:{} -> {}\t| L:{}".format(D, in_ch, out_ch, L))
        self.up.append(Up(Y, Y_inv, area, in_ch, out_ch, L, interval, True))
        if verbose:
            print("Final\t| C:{} -> {}\t| L:{}".format(out_ch, n_class, L))
        self.final = Final(Y, Y_inv, area, out_ch, n_class, L_in, interval)

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        x_down = [self.down[0](x)]
        for i in range(len(self.down) - 1):
            x_down.append(self.down[i + 1](x_down[-1]))
        x_up = x_down[-1]
        for i in range(len(self.up)):
            x_up = self.up[i](x_up, x_down[-2 - i])
        x_up = self.final(x_up)

        return x_up
