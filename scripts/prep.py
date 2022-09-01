"""
Example script for data preparation.

If you use this code, please cite the following paper.

    Seungbo Ha and Ilwoo Lyu
    SPHARM-Net: Spherical Harmonics-based Convolution for Cortical Parcellation.
    IEEE Transactions on Medical Imaging. 2022

Copyright 2022 Ilwoo Lyu

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import os
import argparse
import numpy as np
from joblib import Parallel, delayed

from spharmnet.lib.sphere import TriangleSearch
from spharmnet.lib.io import read_feat, read_mesh, read_annot


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sphere",
        type=str,
        default="./sphere/ico6.vtk",
        help="Reference sphere mesh for re-tessellation (vtk or FreeSurfer format)",
    )
    parser.add_argument("--data-dir", type=str, help="Path to FreeSurfer home (default: $SUBJECTS_DIR)")
    parser.add_argument("--feat-dir", type=str, default="surf", help="Path to geometry for parcellation")
    parser.add_argument("--label-dir", type=str, default="label", help="Path to target labels")
    parser.add_argument("--native-sphere-dir", type=str, default="surf", help="Path to native sphere")
    parser.add_argument("--out-dir", type=str, default="./dataset", help="Path to re-tessellated data (output)")
    parser.add_argument("--native-sphere", type=str, default="sphere", help="Native sphere mesh (sphere, sphere.reg, etc.)")
    parser.add_argument("--hemi", type=str, nargs="+", choices=["lh", "rh"], help="Hemisphere for data generation", required=True)
    parser.add_argument("--in-ch", type=str, default=["curv", "sulc", "inflated.H"], nargs="+", help="List of geometry")
    parser.add_argument("--annot", type=str, default="aparc", help="Manual labels (e.g. aparc for ?h.aparc.annot)")
    parser.add_argument("--threads", type=int, default=1, help="# of CPU threads for parallel data generation")
    args = parser.parse_args()

    return args


def gen_data(data_dir, out_dir, feat_dir, label_dir, native_sphere_dir, subj_name, hemi, native_sphere, in_ch, ico_v, annot_file):
    print("Processing {}...".format(subj_name))

    feat_dir = os.path.join(data_dir, subj_name, feat_dir)
    native_sphere_dir = os.path.join(data_dir, subj_name, native_sphere_dir)
    label_dir = os.path.join(data_dir, subj_name, label_dir)

    feat_out_dir = os.path.join(out_dir, "features")
    label_out_dir = os.path.join(out_dir, "labels")
    csv_out_dir = os.path.join(out_dir, "label_csv")

    for this_hemi in hemi:
        native_v, native_f = read_mesh(os.path.join(native_sphere_dir, this_hemi + "." + native_sphere))
        tree = TriangleSearch(native_v, native_f)
        triangle_idx, bary_coeff = tree.query(ico_v)

        # Generating features
        for feat_name in in_ch:
            feat_path = os.path.join(feat_dir, this_hemi + "." + feat_name)
            feat = read_feat(feat_path)
            feat_remesh = np.multiply(feat[native_f[triangle_idx]], bary_coeff).sum(-1)

            with open(
                os.path.join(feat_out_dir, "{}.{}.aug0.{}.dat".format(subj_name, this_hemi, feat_name)), "wb"
            ) as f:
                f.write(feat_remesh)

        # Generating labels
        num_vert = native_v.shape[0]
        label_arr = np.zeros(num_vert, dtype=np.int16)

        annot = os.path.join(label_dir, this_hemi + "." + annot_file + ".annot")
        vertices, label, sturcture_ls, structureID_ls = read_annot(annot)

        label = [structureID_ls.index(l) if l in structureID_ls else 0 for l in label]
        label_arr[vertices] = label

        label_remesh = label_arr[native_f[triangle_idx, np.argmax(bary_coeff, axis=1)]]
        with open(os.path.join(label_out_dir, "{}.{}.aug0.label.dat".format(subj_name, this_hemi)), "wb") as f:
            f.write(label_remesh)

        # write csv file
        with open(os.path.join(csv_out_dir, "{}.{}.csv".format(subj_name, this_hemi)), "w") as f:
            f.write("label,ID\n")

            for id, roi in enumerate(sturcture_ls):
                f.write("{},{}\n".format(roi, id))


def main(args):
    feat_out_dir = os.path.join(args.out_dir, "features")
    label_out_dir = os.path.join(args.out_dir, "labels")
    csv_out_dir = os.path.join(args.out_dir, "label_csv")

    if not os.path.exists(feat_out_dir):
        os.makedirs(feat_out_dir)
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)
    if not os.path.exists(csv_out_dir):
        os.makedirs(csv_out_dir)

    data_dir = os.environ.get("SUBJECTS_DIR") if args.data_dir is None else args.data_dir
    print("Subject dir: {}".format(data_dir))

    subj_name_ls = sorted(next(os.walk(data_dir))[1])

    ico_v, _ = read_mesh(args.sphere)

    Parallel(n_jobs=args.threads)(
        delayed(gen_data)(
            data_dir=data_dir,
            out_dir=args.out_dir,
            feat_dir=args.feat_dir,
            label_dir=args.label_dir,
            native_sphere_dir=args.native_sphere_dir,
            subj_name=subj_name,
            hemi=args.hemi,
            native_sphere=args.native_sphere,
            in_ch=args.in_ch,
            ico_v=ico_v,
            annot_file=args.annot,
        )
        for subj_name in subj_name_ls
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
