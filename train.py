import healpy as hp
import numpy as np
import sys
import torch

from aquamam import AQuaMaM, get_labels
from configs import configs
from datasets import load_dataloaders
from ipdf import IPDF
from scipy.spatial.transform import Rotation
from torch import nn, optim


def train_aquamam():
    criterion = nn.CrossEntropyLoss(reduction="sum")
    best_valid_loss = float("inf")
    no_improvement = 0
    lr_drops = 0
    for epoch in range(config["epochs"]):
        print(f"epoch: {epoch}")
        model.train()
        for (idx, (imgs, qs)) in enumerate(train_loader):
            qs = qs.to(device)
            preds = model(imgs.to(device), qs)
            q_labels = get_labels(qs, model.bins)[:, :3]
            loss = criterion(preds.permute(0, 2, 1), q_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 500 == 0:
                print(f"batch_loss: {loss.item() / len(imgs)}", flush=True)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for (imgs, qs) in valid_loader:
                qs = qs.to(device)
                preds = model(imgs.to(device), qs)
                q_labels = get_labels(qs, model.bins)[:, :3]
                loss = criterion(preds.permute(0, 2, 1), q_labels)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader.dataset)
        print(f"valid_loss: {valid_loss}\n")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            no_improvement = 0
            lr_drops = 0
            torch.save(model.state_dict(), params_f)

        else:
            no_improvement += 1
            if no_improvement == config["patience"]:
                lr_drops += 1
                if lr_drops == 2:
                    break

                no_improvement = 0
                print("Reducing learning rate.")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.5


def generate_healpix_grid(recursion_level=None, size=None):
    # See: # https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L380.
    # I replaced TensorFlow functions with functions from SciPy and NumPy.
    assert not (recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size / 72.0) / np.log(8.0))), 0)

    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    polars = np.arccos(s2_points[:, 2])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)

    R1s = Rotation.from_euler("X", azimuths).as_matrix()
    R2s = Rotation.from_euler("Z", polars).as_matrix()
    R3s = Rotation.from_euler("X", tilts).as_matrix()

    Rs = np.einsum("bij,tjk->tbik", R1s @ R2s, R3s).reshape(-1, 3, 3)
    return Rs


def get_R_grid(number_queries):
    # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L272.
    grid_sizes = 72 * 8 ** np.arange(7)
    size = grid_sizes[np.argmin(np.abs(np.log(number_queries) - np.log(grid_sizes)))]
    R_grid = generate_healpix_grid(size=size)
    return torch.Tensor(R_grid)


def train_ipdf():
    epoch = 0
    step = 0
    iterations = config["iterations"]
    warmup_steps = config["warmup_steps"]
    lr = config["lr"]
    best_valid_loss = float("inf")
    R_grid = get_R_grid(config["number_queries"]).to(device)
    while True:
        if step > iterations:
            break

        print(f"epoch: {epoch}")
        model.train()
        train_loss = 0
        for (imgs, Rs_fake_Rs) in train_loader:
            probs = model(imgs.to(device), Rs_fake_Rs.float().to(device))
            loss = -torch.log(probs).mean()
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step > iterations:
                break

            # See: https://github.com/google-research/google-research/blob/207f63767d55f8e1c2bdeb5907723e5412a231e1/implicit_pdf/train.py#L160.
            warmup_factor = min(step, warmup_steps) / warmup_steps
            decay_step = max(step - warmup_steps, 0) / (iterations - warmup_steps)
            new_lr = lr * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2
            for g in optimizer.param_groups:
                g["lr"] = new_lr

        train_loss /= len(train_loader)
        print(f"train_loss: {train_loss}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for (imgs, Rs_fake_Rs) in valid_loader:
                if which_dataset == "toy":
                    # See: https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L154.
                    R = Rs_fake_Rs[0, 0].reshape(3, 3).float().to(device)
                    R_delta = R_grid[0].T @ R
                    R_grid_new = (R_grid @ R_delta).reshape(1, -1, 9)
                    prob = model(imgs.to(device), R_grid_new.to(device))[0]
                    loss = -torch.log(prob)

                else:
                    probs = model(imgs.to(device), Rs_fake_Rs.float().to(device))
                    loss = -torch.log(probs).mean()

                valid_loss += loss.item()

        valid_loss /= len(valid_loader.dataset)
        print(f"valid_loss: {valid_loss}\n")
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), params_f)

        epoch += 1
        torch.cuda.empty_cache()


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
        model_details["neg_samples"] = config["neg_samples"]

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    if which_dataset == "toy":
        model_details["max_pow"] = config["model_args"]["toy_args"]["max_pow"]

    (train_loader, valid_loader, _) = load_dataloaders(
        which_dataset, model_details, config["batch_size"], config["num_workers"]
    )

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    if which_model == "ipdf":
        train_ipdf()

    else:
        train_aquamam()
