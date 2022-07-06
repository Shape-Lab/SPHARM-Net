"""
July 2021

Seungbo Ha, mj0829@unist.ac.kr
Ilwoo Lyu, ilwoolyu@unist.ac.kr

3D Shape Analysis Lab
Department of Computer Science and Engineering
Ulsan National Institute of Science and Technology
"""

import numpy as np
import os


def read_feature(fname):
    """
    Read FreeSurfer's geometry.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    feat : 1D array
        Vertex-wise geometry.
    fnum : int
        # of faces.

    Notes
    _____
        https://github.com/fieldtrip/fieldtrip/tree/master/external/freesurfer
        https://github.com/fieldtrip/fieldtrip/blob/master/external/freesurfer/read_curv.m
        They read binary files in big-endian.
    """

    with open(fname, "rb") as f:
        h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
        vnum = (h0 << 16) + (h1 << 8) + h2

        if vnum == 0xFFFFFF:
            vnum, fnum, vals_per_vertex = np.fromfile(f, dtype=np.dtype(">i4"), count=3)
            feat = np.fromfile(f, dtype=np.dtype(">f4"), count=vnum)
        else:
            f0, f1, f2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            fnum = (f0 << 16) + (f1 << 8) + f2
            feat = np.fromfile(f, dtype=np.dtype(">i2"), count=vnum) / 100

    return feat, fnum


def read_surf(fname):
    """
    Read FreeSurfer's surface.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    vertex_coords : 2D array, shape = [n_vertex, 3]
        Vertex coordinates.
    faces : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    TRIANGLE_FILE_MAGIC_NUMBER = 0xFFFFFE
    QUAD_FILE_MAGIC_NUMBER = 0xFFFFFF
    NEW_QUAD_FILE_MAGIC_NUMBER = 0xFFFFFD

    with open(fname, "rb") as f:
        h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
        magic = (h0 << 16) + (h1 << 8) + h2

        if (magic == QUAD_FILE_MAGIC_NUMBER) | (magic == NEW_QUAD_FILE_MAGIC_NUMBER):
            # need to be verified
            h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            vnum = (h0 << 16) + (h1 << 8) + h2

            h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            fnum = (h0 << 16) + (h1 << 8) + h2

            vertex_coords = np.fromfile(f, dtype=np.dtype(">i2"), count=3 * vnum) / 100
            vertex_coords = vertex_coords.reshape(-1, 3)
            arr = np.fromfile(f, dtype=np.dtype("B"), count=9 * fnum)
            faces = (arr[0::3] << 16) + (arr[0::3] << 8) + arr[0::3]
            faces = faces.reshape(-1, 3)

            return vertex_coords, faces

        elif magic == TRIANGLE_FILE_MAGIC_NUMBER:
            f.readline()
            f.readline().strip()

            vnum, fnum = np.fromfile(f, dtype=np.dtype(">i4"), count=2)
            vertex_coords = np.fromfile(f, dtype=np.dtype(">f4"), count=3 * vnum)
            faces = np.fromfile(f, dtype=np.dtype(">i4"), count=3 * fnum)

            vertex_coords = vertex_coords.reshape(vnum, 3)
            faces = faces.reshape(fnum, 3)

            return vertex_coords, faces

        else:
            raise Exception("SurfReaderError: unknown format!")


def read_annotation(fname):
    """
    Read FreeSurfer's annot.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    vertices : 1D array
        Vertex coordinates.
    label : 1D array
        Vertex-wise label.
    """

    with open(fname, "rb") as f:
        A = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]
        temp = np.fromfile(f, dtype=np.dtype(">i4"), count=2 * A)
        vertices = temp[0::2]
        label = temp[1::2]

        _bool = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]

        if _bool:
            numEntries = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]

            if numEntries > 0:  # need to be verified
                temp_len = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]
                temp = np.fromfile(f, dtype=np.dtype(">i1"), count=temp_len)
                # orig_tab = ''.join([chr(item) for item in temp[:-1]])
                # print("Original colortable file : ", orig_tab)
                structure_ls = []
                structureID_ls = []
                for _ in range(numEntries):
                    temp_len = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]
                    temp = np.fromfile(f, dtype=np.dtype(">i1"), count=temp_len)
                    struct_name = "".join([chr(item) for item in temp[:-1]])
                    structure_ls.append(struct_name)
                    row = np.fromfile(f, dtype=np.dtype(">i4"), count=4)
                    structureID = row[0] + row[1] * 2 ** 8 + row[2] * 2 ** 16
                    structureID_ls.append(structureID)

            else:
                version = -numEntries
                if version != 2:
                    raise Exception("AnnotReaderError: version != 2", version)

                numEntries = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]
                temp_len = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]
                temp = np.fromfile(f, dtype=np.dtype(">i1"), count=temp_len)
                # orig_tab = ''.join([chr(item) for item in temp[:-1]])
                # print("Original colortable file : ", orig_tab)
                numEntriesToRead = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]

                structure_ls = []
                structureID_ls = []
                for _ in range(numEntriesToRead):
                    structure = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0] + 1
                    if structure < 0:
                        raise Exception("AnnotReaderError: entry index < 0")

                    temp_len = np.fromfile(f, dtype=np.dtype(">i4"), count=1)[0]
                    temp = np.fromfile(f, dtype=np.dtype(">i1"), count=temp_len)
                    struct_name = "".join([chr(item) for item in temp[:-1]])
                    structure_ls.append(struct_name)
                    row = np.fromfile(f, dtype=np.dtype(">i4"), count=4)
                    structureID = row[0] + row[1] * 2 ** 8 + row[2] * 2 ** 16
                    structureID_ls.append(structureID)

    return vertices, label, structure_ls, structureID_ls


def read_vtk(fname):
    """
    Read a vtk file (ASCII version and VTK < v4.0).

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    fid = open(fname, "r")
    lines = fid.readlines()
    fid.close()

    v = []
    f = []
    b = ([lines.index(i) for i in lines if i.startswith("POINTS")])[0] + 1
    nVert = int(lines[b - 1].split()[1])
    while len(v) < nVert * 3:
        line = lines[b]
        row = [float(n) for n in line.split()]
        v += row
        b += 1

    b = ([lines.index(i) for i in lines if i.startswith("POLYGONS")])[0] + 1
    nFaces = int(lines[b - 1].split()[1])
    for i in range(b, b + nFaces):
        line = lines[i]
        row = [int(n) for n in line.split()]
        row = row[1:]
        f.append(row)

    v = np.reshape(v, (nVert, 3))
    f = np.array(f)

    return v, f


def write_property(fname, v, f, prop):
    """
    Write vtk file with vertex-wise meta data.

    Parameters
    __________
    fname : str
        Output file name.
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    prop : dict, shape = ("attribute name", vertex-wise data)
        Attribute can be any 1D scalar array with size of (n_vertex, 1).
    """

    fid = open(fname, "w")
    len_v = v.shape[0]
    len_f = f.shape[0]

    fid.write("# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n")
    fid.write(f"POINTS {len_v} float\n")
    for row in v:
        fid.write(f"{row[0]} {row[1]} {row[2]}\n")
    fid.write(f"POLYGONS {len_f} {len_f * 4}\n")
    for row in f:
        fid.write(f"3 {row[0]} {row[1]} {row[2]}\n")
    fid.write(f"POINT_DATA {len_v}\n")
    fid.write(f"FIELD ScalarData {len(prop)}\n")
    for key in prop.keys():
        fid.write(f"{key} 1 {len_v} float\n")
        val = prop[key]
        for num in val:
            fid.write(f"{num}\n")
    fid.close()


def read_mesh(fname):
    """
    Read a mesh file.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    _, ext = os.path.splitext(fname.lower())

    if ext == ".vtk":
        v, f = read_vtk(fname)
    else:
        v, f = read_surf(fname)

    return v, f
