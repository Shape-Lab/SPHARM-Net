"""
July 2021

Ilwoo Lyu, ilwoolyu@unist.ac.kr
Seungbo Ha, mj0829@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Ulsan National Institute of Science and Technology
"""

import math
import torch
import torch.nn as nn


class SHConv(nn.Module):
    def __init__(self, in_channels, out_channels, L, interval):
        """
        The spectral convolutional filter has L+1 coefficients.
        Among the L+1 points, we set anchor points for every interval of "interval".
        Those anchors are linearly interpolated to fill the blank positions.

        Parameters
        __________
        in_channels : int
            # of input channels in this layer.
        out_channels : int
            # of output channels in this layer.
        L : int
            Bandwidth of input channels. An individual harmonic coefficient is learned in this bandwidth.
        interval : int
            Interval of anchor points. Harmonic coefficients are learned at every "interval".
            The intermediate coefficients between the anchor points are linearly interpolated.

        Notes
        _____
        Input shape  : [batch, in_channels, (L+1)**2]
        Output shape : [batch, out_channels, (L+1)**2]
        """

        super().__init__()

        ncpt = int(math.ceil(L / interval)) + 1
        interval2 = 1 if interval == 1 else L - (ncpt - 2) * interval

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, ncpt, 1))
        self.l0 = nn.Parameter(
            torch.arange(0, 1, 1.0 / interval).repeat(1, ncpt - 2).view((ncpt - 2, interval)), requires_grad=False
        )
        self.l1 = nn.Parameter(torch.arange(0, 1 + 1e-8, 1.0 / interval2).view((1, interval2 + 1)), requires_grad=False)
        self.repeats = nn.Parameter(torch.tensor([(2 * l + 1) for l in range(L + 1)]), requires_grad=False)

        stdv = 1.0 / math.sqrt(in_channels * (L + 1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        w1 = (
            torch.mul((1 - self.l0), self.weight[:, :, :-2, :]) + torch.mul(self.l0, self.weight[:, :, 1:-1, :])
        ).flatten(-2)
        w2 = (
            torch.mul((1 - self.l1), self.weight[:, :, -2:-1, :]) + torch.mul(self.l1, self.weight[:, :, -1:, :])
        ).flatten(-2)
        w = torch.repeat_interleave(torch.cat([w1, w2], dim=2), self.repeats, dim=2)

        x = torch.mul(w.unsqueeze(0), x.unsqueeze(2)).sum(1)

        return x


class SHT(nn.Module):
    def __init__(self, L, Y_inv, area):
        """
        Spherical harmonic transform (SHT).
        Spherical signals are transformed into the spectral components.

        Parameters
        __________
        L : int
            Bandwidth of SHT. This should match L in SpectralConv.
        Y_inv : 2D array, shape = [n_vertex, (L+1)**2]
            Matrix form of harmonic basis.
        area : 1D array
            Area per vertex.

        Notes
        _____
        Input shape  : [batch, n_ch, n_vertex]
        Output shape : [batch, n_ch, (L+1)**2]
        """

        super().__init__()

        self.Y_inv = Y_inv[:, : (L + 1) ** 2]
        self.area = area

    def forward(self, x):
        x = torch.mul(self.area, x)
        x = torch.matmul(x, self.Y_inv)

        return x


class ISHT(nn.Module):
    def __init__(self, Y):
        """
        Inverse spherical harmonic transform (ISHT).
        Spherical signals are reconstructed from the spectral components.

        Parameters
        __________
        Y : 2D array, shape = [(L+1)**2, n_vertex]
            Matrix form of harmonic basis.

        Notes
        _____
        Input shape  : [batch, n_ch, (L+1)**2]
        Output shape : [batch, n_ch, n_vertex]
        """

        super().__init__()

        self.Y = Y

    def forward(self, x):
        x = torch.matmul(x, self.Y[: x.shape[-1], :])

        return x
