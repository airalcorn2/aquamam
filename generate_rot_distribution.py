import numpy as np
import torch

from aquamam import AQuaMaM, get_exact_qy_densities, get_exact_qz_densities, get_labels
from configs import configs
from datasets import load_dataloaders
from scipy.spatial.transform import Rotation

POS_SAMPLES = 499
NEG_SAMPLES = 500


def generate_cube_dist():
    which_model = "aquamam"
    which_dataset = "cube"
    config = configs[which_model][which_dataset]

    device = "cuda:0"
    model = AQuaMaM(**config["model_args"]).to(device)
    params_f = f"{which_model}_{which_dataset}.pth"
    model.load_state_dict(torch.load(params_f))
    model.eval()

    model_details = {"model": "aquamam"}
    (_, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["test_batch_size"], config["num_workers"]
    )

    test_idxs = [19, 101]
    cube_dists = []
    for test_idx in test_idxs:
        (img, true_q) = test_loader.dataset[test_idx]

        with torch.no_grad():
            (_, samp_qs) = model.sample(img[None].repeat(POS_SAMPLES, 1, 1, 1).to(device))

        pos_qs = torch.cat([torch.Tensor(true_q[None]).to(device), samp_qs])

        neg_qs = Rotation.random(NEG_SAMPLES).as_quat()
        neg_qs[neg_qs[:, -1] < 0] = -neg_qs[neg_qs[:, -1] < 0]
        neg_qs = torch.Tensor(neg_qs).to(device)
        qs = torch.cat([pos_qs, neg_qs])

        with torch.no_grad():
            preds = model(img[None].repeat(len(qs), 1, 1, 1).to(device), qs)

        q_labels = get_labels(qs, model.bins)[:, :3]
        log_ps = torch.zeros(len(preds)).to(device)
        n_samples = 1 + POS_SAMPLES + NEG_SAMPLES
        for i in range(3):
            qc_log_ps = torch.log_softmax(preds[:, i], dim=1)
            log_ps += qc_log_ps[torch.arange(n_samples), q_labels[:, i]]

        log_ps += torch.log(get_exact_qy_densities(qs, model.bins, model.bin_bottoms))
        log_ps += torch.log(get_exact_qz_densities(qs, model.bins, model.bin_bottoms))
        log_ps += torch.log(qs[:, 3])
        log_ps += np.log(config["model_args"]["n_bins"]) - np.log(2)

        vals = torch.cat([qs, log_ps[:, None]], dim=1)
        cube_dists.append(vals.cpu().numpy())

    np.save(f"cube_dists.npy", np.stack(cube_dists))


def generate_cylinder_dist():
    which_model = "aquamam"
    which_dataset = "cylinder"
    config = configs[which_model][which_dataset]

    device = "cuda:0"
    model = AQuaMaM(**config["model_args"]).to(device)
    params_f = f"{which_model}_{which_dataset}.pth"
    model.load_state_dict(torch.load(params_f))
    model.eval()

    model_details = {"model": "aquamam"}
    (_, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["test_batch_size"], config["num_workers"]
    )

    test_idx = 8
    (img, true_q) = test_loader.dataset[test_idx]

    with torch.no_grad():
        (_, samp_qs) = model.sample(img[None].repeat(POS_SAMPLES, 1, 1, 1).to(device))

    pos_qs = torch.cat([torch.Tensor(true_q[None]).to(device), samp_qs])

    neg_qs = Rotation.random(NEG_SAMPLES).as_quat()
    neg_qs[neg_qs[:, -1] < 0] = -neg_qs[neg_qs[:, -1] < 0]
    neg_qs = torch.Tensor(neg_qs).to(device)
    qs = torch.cat([pos_qs, neg_qs])

    with torch.no_grad():
        preds = model(img[None].repeat(len(qs), 1, 1, 1).to(device), qs)

    q_labels = get_labels(qs, model.bins)[:, :3]
    log_ps = torch.zeros(len(preds)).to(device)
    n_samples = 1 + POS_SAMPLES + NEG_SAMPLES
    for i in range(3):
        qc_log_ps = torch.log_softmax(preds[:, i], dim=1)
        log_ps += qc_log_ps[torch.arange(n_samples), q_labels[:, i]]

    log_ps += torch.log(get_exact_qy_densities(qs, model.bins, model.bin_bottoms))
    log_ps += torch.log(get_exact_qz_densities(qs, model.bins, model.bin_bottoms))
    log_ps += torch.log(qs[:, 3])
    log_ps += np.log(config["model_args"]["n_bins"]) - np.log(2)

    vals = torch.cat([qs, log_ps[:, None]], dim=1)
    np.save(f"cylinder_dist.npy", vals.cpu().numpy())


def sample_cube():
    which_model = "aquamam"
    which_dataset = "cube"
    config = configs[which_model][which_dataset]

    device = "cuda:0"
    model = AQuaMaM(**config["model_args"]).to(device)
    params_f = f"{which_model}_{which_dataset}.pth"
    model.load_state_dict(torch.load(params_f))
    model.eval()

    model_details = {"model": "aquamam"}
    (_, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["test_batch_size"], config["num_workers"]
    )

    n_samples = 1 + POS_SAMPLES + NEG_SAMPLES
    test_idx = 19
    (img, true_q) = test_loader.dataset[test_idx]

    with torch.no_grad():
        imgs = img[None].repeat(n_samples, 1, 1, 1).to(device)
        (_, qs) = model.sample(imgs)
        preds = model(imgs, qs)

        q_labels = get_labels(qs, model.bins)[:, :3]
        log_ps = torch.zeros(len(preds)).to(device)
        for i in range(3):
            qc_log_ps = torch.log_softmax(preds[:, i], dim=1)
            log_ps += qc_log_ps[torch.arange(n_samples), q_labels[:, i]]

        log_ps += torch.log(get_exact_qy_densities(qs, model.bins, model.bin_bottoms))
        log_ps += torch.log(get_exact_qz_densities(qs, model.bins, model.bin_bottoms))
        log_ps += torch.log(qs[:, 3])
        log_ps += np.log(config["model_args"]["n_bins"]) - np.log(2)

        rotvecs = Rotation.from_quat(qs.cpu().numpy()).as_rotvec()

    np.save(f"cube_rotvecs.npy", rotvecs)


if __name__ == "__main__":
    generate_cube_dist()
    generate_cylinder_dist()
