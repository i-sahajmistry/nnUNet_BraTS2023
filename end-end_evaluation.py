import json
import os
from glob import glob
from subprocess import call

import nibabel
import numpy as np
from joblib import Parallel, delayed

from data_preprocessing.preprocessor import Preprocessor
from utils.utils import get_task_code
from scipy.ndimage import label

import argparse


parser = argparse.ArgumentParser()

# Preparing
parser.add_argument("--data", type=str, help = "Validation Data Path")
parser.add_argument("--results", type=str, help = "Output Path")
parser.add_argument("--ckpt-path", type=str, help = "Checkpoint Path")

# Processing
parser.add_argument(
    "--exec_mode",
    type=str,
    default="test",
    choices=["training", "val", "test"],
    help="Mode for data preprocessing",
)
parser.add_argument("--ohe", action="store_true", help="Add one-hot-encoding for foreground voxels (voxels > 0)")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--task", type=str, default="00", help="Number of task to be run. MSD uses numbers 00-10")
parser.add_argument("--dim", type=int, default=3, choices=[2, 3], help="Data dimension to prepare")
parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs for data preprocessing")

args = parser.parse_args()
args.ohe = True


def load_nifty(directory, example_id, suffix):
    return nibabel.load(os.path.join(directory, example_id + "-" + suffix + ".nii.gz"))


def load_channels(d, example_id):
    return [load_nifty(d, example_id, suffix) for suffix in ["t2f", "t1n", "t1c", "t2w"]]


def get_data(nifty, dtype="int16"):
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)


def prepare_nifty(d):
    example_id = d.split("/")[-1]
    print(example_id)
    t2f, t1n, t1c, t2w = load_channels(d, example_id)
    affine, header = t2f.affine, t2f.header
    vol = np.stack([get_data(t2f), get_data(t1n), get_data(t1c), get_data(t2w)], axis=-1)
    vol = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
    nibabel.save(vol, os.path.join(d, example_id + ".nii.gz"))

    if os.path.exists(os.path.join(d, example_id + "-seg.nii.gz")):
        seg = load_nifty(d, example_id, "seg")
        affine, header = seg.affine, seg.header
        vol = get_data(seg, "unit8")
        vol[vol == 4] = 3
        seg = nibabel.nifti1.Nifti1Image(vol, affine, header=header)
        nibabel.save(seg, os.path.join(d, example_id + "-seg.nii.gz"))


def prepare_dirs(data, train):
    img_path, lbl_path = os.path.join(data, "images"), os.path.join(data, "labels")
    call(f"mkdir -p {img_path}", shell=True)
    if train:
        call(f"mkdir -p {lbl_path}", shell=True)
    dirs = glob(os.path.join(data, "BraTS*"))
    for d in dirs:
        if "-" in d.split("/")[-1]:
            files = glob(os.path.join(d, "*.nii.gz"))
            for f in files:
                if "t2f" in f or "t1n" in f or "t1c" in f or "t2w" in f:
                    continue
                if "-seg" in f:
                    call(f"cp {f} {lbl_path}", shell=True)
                else:
                    call(f"mv {f} {img_path}", shell=True)


def prepare_dataset_json(data, train):
    images, labels = glob(os.path.join(data, "images", "*")), glob(os.path.join(data, "labels", "*"))
    images = sorted([img.replace(data + "/", "") for img in images])
    labels = sorted([lbl.replace(data + "/", "") for lbl in labels])

    modality = {"0": "T2F", "1": "T1N", "2": "T1C", "3": "T2W"}
    labels_dict = {"0": "background", "1": "nicrosis", "2": "edima", "3": "enhancing tumour"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)


def run_parallel(func, args):
    return Parallel(n_jobs=os.cpu_count())(delayed(func)(arg) for arg in args)


def prepare_dataset(data, train):
    print(f"Preparing BraTS21 dataset from: {data}")
    # for item in sorted(glob(os.path.join(data, "BraTS*"))):
    #     prepare_nifty(item)
    run_parallel(prepare_nifty, sorted(glob(os.path.join(data, "BraTS*"))))
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)


def to_lbl(pred):
    enh = pred[2]
    c1, c2, c3 = pred[0] > 0.5, pred[1] > 0.5, pred[2] > 0.5
    pred = (c1 > 0).astype(np.uint8)
    pred[(c2 == False) * (c1 == True)] = 2
    pred[(c3 == True) * (c1 == True)] = 3

    components, n = label(pred == 3)
    for et_idx in range(1, n + 1):
        _, counts = np.unique(pred[components == et_idx], return_counts=True)
        if 1 < counts[0] and counts[0] < 8 and np.mean(enh[components == et_idx]) < 0.9:
            pred[components == et_idx] = 1

    et = pred == 3
    if 0 < et.sum() and et.sum() < 73 and np.mean(enh[et]) < 0.9:
        pred[et] = 1

    pred = np.transpose(pred, (2, 1, 0)).astype(np.uint8)
    return pred


def prepare_preditions(e, args):
    fname = e[0].split("/")[-1].split(".")[0]
    preds = [np.load(f) for f in e]
    p = to_lbl(np.mean(preds, 0))

    img = nibabel.load(os.path.join(args.data, f"images/{fname}.nii.gz"))
    nibabel.save(
        nibabel.Nifti1Image(p, img.affine, header=img.header),
        os.path.join(args.results, f"final_preds/{fname}.nii.gz"),
    )


if __name__ == '__main__':

    print("Preparing Dataset!")
    prepare_dataset(args.data, False)

    print()

    print("Pre-Processing Dataset!")
    Preprocessor(args).run()
    task_code = get_task_code(args)

    data_path = os.path.join(args.results, get_task_code(args))
    call(f"python main.py --gpus 1 --amp --save_preds --exec_mode predict --brats --brats22_model --results {args.results} --data {data_path}/test --ckpt_path {args.ckpt_path} --tta --depth 5 --filters 64 96 128 192 256 384", shell=True)    

    os.makedirs(os.path.join(args.results, "final_preds"), exist_ok=True)
    preds = sorted(glob(os.path.join(args.results, "predictions*")))
    examples = list(zip(*[sorted(glob(f"{p}/*.npy")) for p in preds]))
    print("Preparing final predictions")
    for e in examples:
        prepare_preditions(e, args)
    print("Finished!")

    print()

    print("Removing temporary files!")
    rm_path = os.path.join(args.results, get_task_code(args))
    call(f"rm -rf {rm_path}", shell=True)

    rm_path = os.path.join(args.results, "params.json")
    call(f"rm -rf {rm_path}", shell=True)

    rm_path = os.path.join(args.data, "images")
    call(f"rm -rf {rm_path}", shell=True)

    rm_path = os.path.join(args.data, "dataset.json")
    call(f"rm -rf {rm_path}", shell=True)

    rm_path = os.path.join(args.results, "dataset.json")
    call(f"rm -rf {rm_path}", shell=True)

    preds = sorted(glob(os.path.join(args.results, "predictions*")))
    for p in preds:
        call(f"rm -rf {p}", shell=True)

    print("Temporary files removed!")

    final_preds = os.path.join(args.results, "final_preds")
    call(f"mv {final_preds}/* {args.results}", shell=True)

    call(f"rm -rf {final_preds}", shell=True)


''' 
python end-end_evaluation.py --ckpt_path="./results/saved_models/folds=0-epoch=39-dice=84.53.ckpt" \
--data="./data/men_ValidationData" --results="./results/predictions_folds=0-epoch=39-dice=84_53"
'''