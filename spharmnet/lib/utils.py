"""
July 2021

Ilwoo Lyu, ilwoolyu@unist.ac.kr
Seungbo Ha, mj0829@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Ulsan National Institute of Science and Technology
"""

import os
import numpy as np
import random
import re
import itertools
from collections import Iterable, defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy import stats
from scipy.sparse import coo_matrix, csgraph

from .io import read_dat


class SphericalDataset(Dataset):
    def __init__(self, data_dir, partition, fold, num_vert, classes, in_ch, seed, aug, n_splits, hemi, data_norm=True):
        """
        Loader for SPHARM-Net. This module subdivides the input dataset for cross-validation.

        Parameters
        __________
        data_dir : str
            Path to data directory. Data naming convention should be met.
            Geometry: data_dir/features/{subj}.{lh}.aug0.{feat}.dat.
            Label: data_dir/labels/{subj}.{lh}.aug0.label.dat.
        partition : str
            Partition = ['train', 'val', 'test']
        fold : int
            Cross-validation fold.
        num_vert : int
            # of vertices of the reference sphere mesh used for re-tessellation.
            The samples in the dataset are assumed to follow the same tessellation.
        classes : 1D int array
            List of labels. Their numbers are not necessarily continuous.
        in_ch : 1D str array
            input geometric features.
        seed : int
            Seed for data shuffling. This shuffling is deterministic.
        aug : int
            Not used unless training samples are augmented.
        n_splits : int
            A total of cross-validation folds.
        hemi : str
            Hemisphere = ['lh', 'rh']. Both hemispheres can be trained together.
        data_norm : bool, optional
            Z-score+prctile data normalization.
        """

        assert partition in ["train", "test", "val"]
        self.num_vert = num_vert
        self.partition = partition
        self.data_norm = data_norm

        feat_dir = os.path.join(data_dir, "features")
        feat_files = os.listdir(feat_dir)
        feat_files = [f for f in feat_files if f.split(".")[1] in hemi]
        feat_files = [f for ch in in_ch for f in feat_files if ".".join(f.split(".")[3:-1]) == ch]

        label_dir = os.path.join(data_dir, "labels")
        label_files = os.listdir(label_dir)
        label_files = [f for f in label_files if f.split(".")[1] in hemi]

        feat_dict = dict()
        for f in feat_files:
            temp = f.split(".")[0:2]  # ['subj_name', 'lh']
            subj = ".".join(temp)  # 'subj_name.lh'
            if subj not in feat_dict:
                feat_dict[subj] = dict()
            key = "aug" + re.sub("[^0-9]", "", f.split(".")[2])  # aug0, aug1, ...
            f_path = os.path.join(feat_dir, f)
            feat_dict[subj].setdefault(key, []).append(f_path)

        label_dict = dict()
        for f in label_files:
            temp = f.split(".")[0:2]  # ['subj_name', 'lh']
            subj = ".".join(temp)  # 'subj_name.lh'
            if subj not in label_dict:
                label_dict[subj] = dict()
            key = "aug" + re.sub("[^0-9]", "", f.split(".")[2])
            f_path = os.path.join(label_dir, f)
            label_dict[subj][key] = label_dict[subj].setdefault(key, f_path)

        subj_list = label_dict.keys()
        subj_list = sorted(subj_list)
        random.seed(seed)
        random.shuffle(subj_list)

        train_subj, val_subj, test_subj = self.kfold(subj_list, n_splits, fold)

        # final list
        self.feat_list = []
        self.label_list = []
        self.name_list = []

        if partition == "train":
            for subj in train_subj:
                for i in range(0, aug + 1):
                    self.feat_list.append(feat_dict[subj]["aug" + str(i)])
                    self.label_list.append(label_dict[subj]["aug" + str(i)])
                    self.name_list.append(subj)

        if partition == "val":
            for subj in val_subj:
                self.feat_list.append(feat_dict[subj]["aug0"])
                self.label_list.append(label_dict[subj]["aug0"])
                self.name_list.append(subj)

        if partition == "test":
            for subj in test_subj:
                self.feat_list.append(feat_dict[subj]["aug0"])
                self.label_list.append(label_dict[subj]["aug0"])
                self.name_list.append(subj)

        # label dictionary
        self.lut, _ = squeeze_label(classes)

    def __len__(self):
        return len(self.feat_list)

    def __getitem__(self, idx):
        # load files
        data = np.array([])
        for f in self.feat_list[idx]:
            temp = read_dat(f, self.num_vert)
            data = np.append(data, temp)

        data = np.reshape(data, (-1, self.num_vert)).astype(np.float32)
        label = read_dat(self.label_list[idx], self.num_vert)
        label = np.asarray(label).astype(int)

        if self.data_norm:
            data = normalize_data(data)

        label = [self.lut[l] for l in label]
        label = np.asarray(label)

        return data, label, self.name_list[idx]

    def kfold(self, subj, n_splits=5, fold=1):
        total_subj = len(subj)
        fold_size = total_subj // n_splits
        fold_residual = total_subj - fold_size * n_splits

        fold_size = [fold_size + 1 if i < fold_residual else fold_size for i in range(n_splits)]
        fold_idx = [0] + list(itertools.accumulate(fold_size))

        id_base = n_splits
        id_val = (id_base + fold - 1) % n_splits
        id_test = (id_base + fold) % n_splits

        val = subj[fold_idx[id_val] : fold_idx[id_val] + fold_size[id_val]]
        test = subj[fold_idx[id_test] : fold_idx[id_test] + fold_size[id_test]]
        if id_val > id_test:
            train = subj[fold_idx[id_test] + fold_size[id_test] : fold_idx[id_val]]
        else:
            train = subj[0 : fold_idx[id_val]] + subj[fold_idx[id_test] + fold_size[id_test] : None]

        return train, val, test

    def subj_list(self):
        return self.name_list


class Logger(object):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        try:
            return len(self.read())
        except BaseException as e:
            print("An exception occurred: {}".format(e))

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]

        line = ""
        if isinstance(values, dict):
            for key, val in values.items():
                line += "{} : {}\n".format(key, val)
        else:
            for v in values:
                if isinstance(v, int):
                    line += "{:<10}".format(v)
                elif isinstance(v, float):
                    line += "{:<15.6f}".format(v)
                elif isinstance(v, str):
                    line += "{}".format(v)
                else:
                    raise Exception("Not supported type.")
        with open(self.path, "a") as f:
            f.write(line[:-1] + "\n")


def eval_dice(input, target, n_class, area=None):
    """
    Dice socre.

    Parameters
    __________
    input : torch.tensor, shape = [batch, n_vertex]
        Model inference.
    target : torch.tensor, shape = [batch, n_vertex]
        True labels.
    n_class : int
        # of classes.
    area : torch.tensor, shape = [n_vertex]
        Vertex-wise area.

    Returns
    _______
    dice : torch.tensor, shape = [batch, n_class]
        Batch-wise Dice score.
    """

    if area is None:
        area = 1
    num_batch = input.shape[0]
    batch_numer = torch.zeros(num_batch, n_class)
    batch_denom = torch.zeros(num_batch, n_class)

    for i in range(n_class):
        batch_numer[:, i] = torch.mul(area, ((input == i) & (target == i)).int()).sum(dim=1)
        batch_denom[:, i] = torch.mul(area, (target == i).int()).sum(dim=1) + torch.mul(
            area, (input == i).int()
        ).sum(dim=1)

    return 2 * batch_numer / batch_denom


def eval_accuracy(input, target):
    """
    Accuracy.

    Parameters
    __________
    input : torch.tensor, shape = [batch, n_vertex]
        Model inference.
    target : torch.tensor, shape = [batch, n_vertex]
        True labels.

    Returns
    _______
    n_correct : int
        # of correct vertices.
    n_vert : int
        # of vertices.
    """

    n_correct = (input == target).sum().item()
    n_vert = len(target.flatten(0))

    return n_correct, n_vert


def squeeze_label(data):
    """
    Label mapping to squeeze. Input labels are not necessarily continuous,
    which can consume more space without squeeze.

    Parameters
    __________
    data : 1D int array
        Input labels.

    Returns
    _______
    lut_old2new : dict (int -> int)
        Mapping to squeezed indices.
    lut_new2old : dict (int -> int)
        Mapping to original indices.
    """

    lut_old2new = defaultdict(lambda: 0)
    lut_new2old = dict()
    for i, label in enumerate(data):
        lut_old2new[label] = i
        lut_new2old[i] = label

    return lut_old2new, lut_new2old


def normalize_data(data):
    """
    Data normalization.

    Parameters
    __________
    data : 2D array, shape = [n_channel, n_vert]
        Input data.

    Returns
    _______
    data : 2D array, shape = [n_channel, n_vert]
        Normalized data.
    """

    data = stats.zscore(data, axis=-1)
    data[data < -3] = -3 - (1 - np.exp(3 + data[data < -3]))
    data[data > 3] = 3 + (1 - np.exp(3 - data[data > 3]))
    data /= np.std(data, axis=-1)[:, None]
    data[data < -3] = -3 - (1 - np.exp(3 + data[data < -3]))
    data[data > 3] = 3 + (1 - np.exp(3 - data[data > 3]))

    return data


def refine_label(data, f):
    """
    Label refinement.
    This function assumes that each ROI has a single connected component.
    Do NOT use this function if any ROI has more than one connected component.

    Parameters
    __________
    data : 2D array, shape = [n_label, n_vert]
        Model inference.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.

    Returns
    _______
    data : 2D array, shape = [n_label, n_vert]
        Refined model inference.
    """

    n_label, n_vert = data.shape
    v1_ = np.hstack((f[:, 0], f[:, 1], f[:, 2]))
    v2_ = np.hstack((f[:, 1], f[:, 2], f[:, 0]))
    n_comp = 0
    label = np.argmax(data, 0)
    label_old = np.zeros(data.shape[-1])

    while n_comp != n_label and (label_old != label).any():
        label_old = label

        idx = label[v1_] == label[v2_]
        v1 = v1_[idx]
        v2 = v2_[idx]

        m = coo_matrix((np.ones(v1.shape[0]), (v1, v2)), shape=(n_vert, n_vert))
        n_comp, comp = csgraph.connected_components(m, directed=False, return_labels=True)

        comp_size = [len(label[comp == i]) for i in range(n_comp)]
        comp_ordered = np.argsort(comp_size)[::-1]

        for i in range(n_label, n_comp):
            idx = comp == comp_ordered[i]
            data[label[idx], idx] = np.finfo(data.dtype).min

        label = np.argmax(data, 0)

    return data
