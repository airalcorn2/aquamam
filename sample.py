# I have to import this first or I get weird library errors.
import healpy as hp

import numpy as np
import pickle
import sys
import torch

from aquamam import AQuaMaM, get_labels
from configs import configs
from datasets import load_dataloaders
from ipdf import IPDF
from scipy.spatial.transform import Rotation
from train import get_R_grid


def get_toy_dicts(dataset, device, model):
    cat2labels_idxs = {}
    cat2rot_counts = {}
    cat2Rs = {}
    for (cat, quats) in dataset.cat2rots.items():
        cat2Rs[cat] = Rotation.from_quat(quats).as_matrix()
        labels = (
            get_labels(torch.Tensor(quats).to(device), model.bins)[:, :3].cpu().numpy()
        )
        labels = labels[np.lexsort(np.rot90(labels))]
        labels_idxs = {}
        rot_counts = {}
        for row in labels:
            labels_idx = len(labels_idxs)
            labels_idxs[tuple(row)] = labels_idx
            rot_counts[labels_idx] = 0

        cat2labels_idxs[cat] = labels_idxs
        cat2rot_counts[cat] = rot_counts

    return (cat2Rs, cat2labels_idxs, cat2rot_counts)


def sample_aquamam_toy():
    (cat2Rs, cat2labels_idxs, cat2rot_counts) = get_toy_dicts(
        test_loader.dataset, device, model
    )
    cat2incorrect_labels = {cat: {} for cat in cat2Rs}
    cat2best_R_dists = {}
    with torch.no_grad():
        for (imgs, _) in test_loader:
            (tokens, vals) = model.sample(imgs.to(device))
            quats = vals.cpu().numpy()
            Rs = Rotation.from_quat(quats).as_matrix()

            for (cat, cat_Rs) in cat2Rs.items():
                pred_cat_Rs = Rs[imgs == cat]
                R_diffs = np.einsum(
                    "bij,cjk->bcik", pred_cat_Rs, cat_Rs.transpose(0, 2, 1)
                )
                traces = np.trace(R_diffs, axis1=2, axis2=3)
                best_R_dists = np.arccos((traces - 1) / 2).min(axis=1)
                cat2best_R_dists.setdefault(cat, []).append(best_R_dists)

            imgs = imgs.cpu().numpy()
            tokens = tokens.cpu().numpy()
            for (idx, cat) in enumerate(imgs):
                labels = tuple(tokens[idx])
                try:
                    labels_idx = cat2labels_idxs[cat][labels]
                    cat2rot_counts[cat][labels_idx] += 1

                except KeyError:
                    cat2incorrect_labels[cat][labels] = (
                        cat2incorrect_labels[cat].get(labels, 0) + 1
                    )

    for (cat, best_R_dists) in cat2best_R_dists.items():
        cat2best_R_dists[cat] = np.concatenate(best_R_dists)

    dicts = {
        "cat2labels_idxs": cat2labels_idxs,
        "cat2incorrect_labels": cat2incorrect_labels,
        "cat2best_R_dists": cat2best_R_dists,
        "cat2rot_counts": cat2rot_counts,
    }
    pickle.dump(dicts, open(f"{which_model}_{which_dataset}.pydict", "wb"))


def sample_ipdf_toy():
    cat2Rs = test_loader.dataset.cat2rots
    cat2best_R_dists = {}
    R_grid = get_R_grid(config["number_queries"]).reshape(1, -1, 9).to(device)
    cat2rot_counts = {cat: {} for cat in cat2Rs}
    with torch.no_grad():
        for (imgs, _) in test_loader:
            pred_cat_R = model.sample(imgs.to(device), R_grid)[0].reshape(3, 3)
            cat = imgs[0].item()
            cat_Rs = cat2Rs[cat]
            pred_cat_R = pred_cat_R.cpu().numpy()
            str_R = str(pred_cat_R)
            cat2rot_counts[cat][str_R] = cat2rot_counts[cat].get(str_R, 0) + 1
            R_diffs = pred_cat_R @ cat_Rs.transpose(0, 2, 1)
            traces = np.trace(R_diffs, axis1=1, axis2=2)
            best_R_dist = np.arccos((traces - 1) / 2).min()
            cat2best_R_dists.setdefault(cat, []).append(best_R_dist)

    for (cat, best_R_dists) in cat2best_R_dists.items():
        cat2best_R_dists[cat] = np.array(best_R_dists)

    dicts = {"cat2best_R_dists": cat2best_R_dists, "cat2rot_counts": cat2rot_counts}
    pickle.dump(dicts, open(f"{which_model}_{which_dataset}.pydict", "wb"))


if __name__ == "__main__":
    which_model = sys.argv[1]
    which_dataset = sys.argv[2]
    config = configs[which_model][which_dataset]
    params_f = f"{which_model}_{which_dataset}.pth"
    device = "cuda:0"
    model_details = {"model": which_model.split("_")[0]}
    if which_model == "aquamam":
        model = AQuaMaM(**config["model_args"]).to(device)

    else:
        model = IPDF(**config["model_args"]).to(device)
        model_details["neg_samples"] = 1

    if which_dataset == "toy":
        model_details["max_pow"] = config["model_args"]["toy_args"]["max_pow"]

    model.load_state_dict(torch.load(params_f))
    model.eval()

    if which_dataset == "toy":
        (test_loader, _, _) = load_dataloaders(
            which_dataset,
            model_details,
            config["test_batch_size"],
            config["num_workers"],
        )

    else:
        (_, _, test_loader) = load_dataloaders(
            which_dataset,
            model_details,
            config["test_batch_size"],
            config["num_workers"],
        )

    if which_dataset == "toy":
        if which_model == "aquamam":
            sample_aquamam_toy()
        else:
            sample_ipdf_toy()
