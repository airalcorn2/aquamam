# I have to import this first or I get weird library errors.
import healpy as hp

import numpy as np
import pickle
import sys
import time
import torch

from aquamam import AQuaMaM, get_exact_qy_densities, get_exact_qz_densities, get_labels
from configs import configs
from datasets import load_dataloaders
from ipdf import IPDF
from torch import nn
from train import get_R_grid


def evaluate_aquamam():
    criterion = nn.CrossEntropyLoss(reduction="sum")
    nll = 0.0
    start = time.time()
    classification_nll = 0
    with torch.no_grad():
        for (imgs, qs) in test_loader:
            qs = qs.to(device)
            preds = model(imgs.to(device), qs)

            q_labels = get_labels(qs, model.bins)[:, :3]
            loss = criterion(preds.permute(0, 2, 1), q_labels)
            classification_nll += loss.item()
            loss -= torch.log(
                get_exact_qy_densities(qs, model.bins, model.bin_bottoms)
            ).sum()
            loss -= torch.log(
                get_exact_qz_densities(qs, model.bins, model.bin_bottoms)
            ).sum()
            loss -= torch.log(qs[:, 3]).sum()
            nll += loss.item()

    print(f"time: {time.time() - start:.2f} seconds")
    print(f"classification_nll: {classification_nll / len(test_loader.dataset)}")

    n_bins = config["model_args"]["n_bins"]
    nll /= len(test_loader.dataset)
    ll = np.log(n_bins) - np.log(2) - nll

    print(f"ll: {ll}")


def evaluate_ipdf():
    R_grid = get_R_grid(config["number_queries"]).to(device)
    nll = 0.0
    start = time.time()
    with torch.no_grad():
        for (imgs, Rs_fake_Rs) in test_loader:
            # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L154.
            R = Rs_fake_Rs[0, 0].reshape(3, 3).float().to(device)
            R_delta = R_grid[0].T @ R
            R_grid_new = (R_grid @ R_delta).reshape(1, -1, 9)
            prob = model(imgs.to(device), R_grid_new.to(device))[0]
            loss = -torch.log(prob)
            nll += loss.item()

    print(f"time: {time.time() - start:.2f} seconds")
    print(f"ll: {-nll / len(test_loader.dataset)}")


def get_ipdf_toy_scores():
    R_grid = get_R_grid(config["number_queries"]).to(device)
    img2scores = {}
    with torch.no_grad():
        for (imgs, Rs_fake_Rs) in test_loader:
            # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L154.
            R = Rs_fake_Rs[0, 0].reshape(3, 3).float().to(device)
            img = imgs[0].item()
            if img not in img2scores:
                img2scores[img] = {}

            R_np = str(R.cpu().numpy())
            if R_np in img2scores[img]:
                continue

            R_delta = R_grid[0].T @ R
            R_grid_new = (R_grid @ R_delta).reshape(1, -1, 9)
            score = model.get_scores(imgs.to(device), R_grid_new.to(device))[:, 0]
            img2scores[img][R_np] = score.item()

    pickle.dump(img2scores, open("img2scores.pydict", "wb"))


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

    (_, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["test_batch_size"], config["num_workers"]
    )

    if which_model == "ipdf":
        evaluate_ipdf()

    else:
        evaluate_aquamam()
