# I have to import this first or I get weird library errors.
import healpy as hp

import numpy as np
import sys
import time
import torch

from aquamam import AQuaMaM
from configs import configs
from datasets import load_dataloaders
from ipdf import IPDF
from sample import get_toy_dicts
from scipy.spatial.transform import Rotation
from train import get_R_grid


def predict_aquamam_toy():
    (cat2Rs, _, _) = get_toy_dicts(test_loader.dataset, device, model)
    cat2best_R_dists = {}
    with torch.no_grad():
        for (imgs, qs) in test_loader:
            vals = model.beam_search(imgs.to(device), config["beam_k"]).cpu().numpy()
            Rs = Rotation.from_quat(vals).as_matrix()

            for (cat, cat_Rs) in cat2Rs.items():
                pred_cat_Rs = Rs[imgs == cat]
                R_diffs = np.einsum(
                    "bij,cjk->bcik", pred_cat_Rs, cat_Rs.transpose(0, 2, 1)
                )
                traces = np.trace(R_diffs, axis1=2, axis2=3)
                best_R_dists = np.arccos((traces - 1) / 2).min(axis=1)
                cat2best_R_dists.setdefault(cat, []).append(best_R_dists)

    for (cat, best_R_dists) in cat2best_R_dists.items():
        cat2best_R_dists[cat] = np.concatenate(best_R_dists)
        print(f"{cat}: {180 * cat2best_R_dists[cat].mean()}°")


def predict_aquamam():
    R_dists = []
    start = time.time()
    with torch.no_grad():
        for (imgs, qs) in test_loader:
            true_Rs = Rotation.from_quat(qs).as_matrix()
            vals = model.beam_search(imgs.to(device), config["beam_k"]).cpu().numpy()
            pred_Rs = Rotation.from_quat(vals).as_matrix()

            R_diffs = pred_Rs @ true_Rs.transpose(0, 2, 1)
            traces = np.trace(R_diffs, axis1=1, axis2=2)
            R_dists.append(np.arccos((traces - 1) / 2))

    print(f"time: {time.time() - start:.2f} seconds")
    R_dists = np.concatenate(R_dists)
    print(f"Average distance: {180 * R_dists.mean() / np.pi}°")


def predict_ipdf():
    R_grid = get_R_grid(config["number_queries"]).to(device).reshape(1, -1, 9)
    R_dists = []
    start = time.time()
    with torch.no_grad():
        for (imgs, Rs_fake_Rs) in test_loader:
            top_R_idx = model.get_scores(imgs.to(device), R_grid.to(device))[0].argmax()
            true_R = Rs_fake_Rs[0, 0].reshape(3, 3)
            pred_R = R_grid[0, top_R_idx].reshape(3, 3)
            R_diff = pred_R @ true_R.float().T.to(device)
            trace = torch.trace(R_diff)
            R_dists.append(torch.arccos((trace - 1) / 2).item())

    print(f"time: {time.time() - start:.2f} seconds")
    R_dists = np.array(R_dists)
    print(f"Average distance: {180 * R_dists.mean() / np.pi}°")


if __name__ == "__main__":
    which_model = sys.argv[1]
    which_dataset = sys.argv[2]
    config = configs[which_model][which_dataset]
    params_f = f"{which_model}_{which_dataset}.pth"
    device = "cuda:0"
    model_details = {"model": which_model.split("_")[0]}
    if which_model == "aquamam":
        model = AQuaMaM(**config["model_args"]).to(device)
        config["test_batch_size"] = config["test_batch_size"] // config["beam_k"]

    else:
        model = IPDF(**config["model_args"]).to(device)
        model_details["neg_samples"] = 1

    if which_dataset == "toy":
        model_details["max_pow"] = config["model_args"]["toy_args"]["max_pow"]

    model.load_state_dict(torch.load(params_f))
    model.eval()

    (_, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["test_batch_size"], config["num_workers"]
    )

    if which_model.startswith("aquamam"):
        if which_dataset == "toy":
            predict_aquamam_toy()
        else:
            predict_aquamam()

    else:
        predict_ipdf()
