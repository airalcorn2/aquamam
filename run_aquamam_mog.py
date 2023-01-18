import matplotlib.pyplot as plt
import numpy as np
import time
import torch

from aquamam_mog import AQuaMaMMoG, get_full_lls, get_pre_lls
from configs import configs
from datasets import load_dataloaders
from scipy.spatial.transform import Rotation
from torch import optim


def train_aquamam_mog():
    best_valid_loss = float("inf")
    no_improvement = 0
    lr_drops = 0
    for epoch in range(config["epochs"]):
        print(f"epoch: {epoch}")
        model.train()
        for (idx, (imgs, qs)) in enumerate(train_loader):
            qs = qs.to(device)
            preds = model(imgs.to(device), qs)
            loss = -get_pre_lls(preds, qs).sum()
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
                loss = -get_pre_lls(preds, qs).sum()
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


def evaluate_aquamam_mog():
    ll = 0.0
    start = time.time()
    with torch.no_grad():
        for (imgs, qs) in test_loader:
            qs = qs.to(device)
            preds = model(imgs.to(device), qs)
            ll += get_full_lls(preds, qs).sum().item()

    print(f"time: {time.time() - start:.2f} seconds")
    print(f"ll: {ll / len(test_loader.dataset)}")


def sample_aquamam_mog_toy():
    cat2Rs = {}
    for (cat, quats) in test_loader.dataset.cat2rots.items():
        cat2Rs[cat] = Rotation.from_quat(quats).as_matrix()

    cat2best_R_dists = {}
    all_imgs = []
    rotvecs = []
    with torch.no_grad():
        for (imgs, _) in test_loader:
            all_imgs.append(imgs.numpy())
            vals = model.sample(imgs.to(device))
            quats = vals.cpu().numpy()
            rs = Rotation.from_quat(quats)
            Rs = rs.as_matrix()
            rotvecs.append(rs.as_rotvec())

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
        print(f"{cat}: {cat2best_R_dists[cat].mean()}")

    np.save("mog_rotvecs.npy", np.concatenate(rotvecs))
    np.save("mog_imgs.npy", np.concatenate(all_imgs))


def plot_rotvecs():
    imgs = np.load("mog_imgs.npy")
    rotvecs = np.load("mog_rotvecs.npy")[imgs == 0][:1000]
    (xs, zs, ys) = np.split(rotvecs, 3, 1)

    plt.rcParams.update({"font.size": 20})
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(xs, ys, zs, s=10, alpha=0.3)
    ax.scatter(0.36354304, -2.52608405, 1.75160622, c="red", s=50, alpha=1)
    # See: https://stackoverflow.com/a/72928548/1316276.
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.set_xlabel("$e_{x}$")
    ax.set_ylabel("$e_{z}$")
    ax.set_zlabel("$e_{y}$")
    plt.show()


if __name__ == "__main__":
    which_model = "aquamam_mog"
    which_dataset = "toy"
    config = configs[which_model][which_dataset]
    params_f = f"{which_model}_{which_dataset}.pth"
    device = "cuda:0"

    model_details = {"model": which_model.split("_")[0]}
    model = AQuaMaMMoG(**config["model_args"]).to(device)

    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params}")

    model_details["max_pow"] = config["model_args"]["toy_args"]["max_pow"]
    (train_loader, valid_loader, _) = load_dataloaders(
        which_dataset, model_details, config["batch_size"], config["num_workers"]
    )

    optimizer = optim.Adam(model.parameters(), config["lr"])
    train_aquamam_mog()

    model.load_state_dict(torch.load(params_f))
    model.eval()

    (_, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["test_batch_size"], config["num_workers"]
    )
    evaluate_aquamam_mog()

    (test_loader, _, _) = load_dataloaders(
        which_dataset,
        model_details,
        config["test_batch_size"],
        config["num_workers"],
    )
    sample_aquamam_mog_toy()
