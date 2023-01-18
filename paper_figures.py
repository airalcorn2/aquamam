import matplotlib.pyplot as plt
import numpy as np
import pickle
import pprint
import seaborn as sns
import torch
import torch.nn.functional as F
import torch_bingham

from datasets import load_dataloaders
from configs import configs, max_pow
from generate_rot_distribution import POS_SAMPLES
from PIL import ImageDraw, ImageFont
from renderer import Renderer
from renderer_settings import *
from scipy.spatial.transform import Rotation
from torch import nn, optim


def count_unique_labels():
    which_model = "aquamam"
    which_dataset = "cube"
    config = configs[which_model][which_dataset]
    model_details = {"model": which_model}
    (train_loader, _, test_loader) = load_dataloaders(
        which_dataset, model_details, config["batch_size"], config["num_workers"]
    )
    n_bins = configs[which_model][which_dataset]["model_args"]["n_bins"]
    bins = np.linspace(-1, 1, n_bins + 1)[1:]

    label_sets = {}
    counts = np.zeros((n_bins, n_bins))
    for (which_dataset, loader) in [("train", train_loader), ("test", test_loader)]:
        df = loader.dataset.df
        quats = np.stack(df["quat"].values)
        labels = np.digitize(quats, bins, right=True)[:, :3]
        label_set = set()
        for tokens in labels:
            label_set.add(tuple(tokens))
            counts[tokens[0], tokens[1]] += 1

        label_sets[which_dataset] = label_set

    same = label_sets["train"] & label_sets["test"]
    print(len(label_sets["train"]))
    print(len(label_sets["test"]))
    print(len(same))


def plot_die_rot_dist():
    renderer = Renderer(
        camera_distance=8,
        angle_of_view=ANGLE_OF_VIEW / 2,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    obj_mtl_path = "cube_obj/cube"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
    # Render object in default pose.
    renderer.render(0.0, 0.0, 0.0).show()
    R = Rotation.from_euler("XY", [np.pi / 4, np.pi / 4]).as_matrix()
    renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
    renderer.render(0.0, 0.0, 0.0).show()
    R = Rotation.from_euler("XY", [-np.pi / 4, -np.pi / 4]).as_matrix()
    renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
    renderer.render(0.0, 0.0, 0.0).show()

    render_idxs = [{0, 180, 191, 479}, {0}]
    cube_dists = np.load("cube_dists.npy")
    plt.rcParams.update({"font.size": 20})
    max_log_p = 15
    color_shift = 50
    alpha_shift = 125
    font = ImageFont.truetype("arial.ttf", 30)
    render_all = False
    for (cube_idx, cube_dist) in enumerate(cube_dists):
        # Note: I had to use the pull request here: https://github.com/matplotlib/matplotlib/pull/23085
        # to get this to look right.
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        rotvecs = Rotation.from_quat(cube_dist[:, :4]).as_rotvec()
        (xs, zs, ys) = np.split(rotvecs, 3, 1)

        log_ps = cube_dist[:, -1]
        print(log_ps[1 + POS_SAMPLES :].max())
        colors = (log_ps + color_shift) / (color_shift + max_log_p)
        colors = colors.clip(0.0, 1)
        alphas = (log_ps + alpha_shift) / (alpha_shift + max_log_p)
        alphas = alphas.clip(0.0, 1) ** 3

        top_log_ps = (-log_ps).argsort()

        plot = ax.scatter(
            xs,
            ys,
            zs,
            depthshade=False,
            c=colors,
            s=100,
            alpha=alphas,
            vmin=0,
            vmax=1.0,
            cmap="coolwarm",
        )
        # See: https://stackoverflow.com/a/72928548/1316276.
        limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
        ax.set_box_aspect(np.ptp(limits, axis=1))

        if render_all:
            cube_render_idxs = set(np.arange(1 + POS_SAMPLES))
        else:
            cube_render_idxs = render_idxs[cube_idx]

        for render_idx in cube_render_idxs:
            R = Rotation.from_rotvec(rotvecs[render_idx]).as_matrix()
            renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
            image = renderer.render(0.0, 0.0, 0.0)
            draw = ImageDraw.Draw(image)
            label = f"{render_idx}: {log_ps[render_idx]:.4f}"
            draw.text((0, 0), label, font=font, fill=(255, 255, 255))
            image.show()

            # for idx in cube_render_idxs:
            ax.text(xs[render_idx, 0], ys[render_idx, 0], zs[render_idx, 0], render_idx)

        cbar = fig.colorbar(plot, label=r"$\ln(p(\mathbf{q}))$")
        n_ticks = 51
        ticks = np.arange(n_ticks) / (n_ticks - 1)
        labels = np.round(np.linspace(-color_shift, max_log_p, n_ticks), 2)
        skips = 5
        cbar.set_ticks(ticks[::skips], labels=labels[::skips])
        ax.set_xlabel("$e_{x}$")
        ax.set_ylabel("$e_{z}$")
        ax.set_zlabel("$e_{y}$")
        plt.show()


def plot_cylinder_rot_dist():
    renderer = Renderer(
        camera_distance=8,
        angle_of_view=ANGLE_OF_VIEW / 2,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    obj_mtl_path = "cylinder_obj/cylinder"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
    # Render object in default pose.
    renderer.render(0.0, 0.0, 0.0).show()
    R = Rotation.from_euler("XY", [np.pi / 4, np.pi / 4]).as_matrix()
    renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
    renderer.render(0.0, 0.0, 0.0).show()
    R = Rotation.from_euler("XY", [-np.pi / 4, -np.pi / 4]).as_matrix()
    renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
    renderer.render(0.0, 0.0, 0.0).show()

    cylinder_render_idxs = {0, 3, 21, 122}
    cylinder_dist = np.load("cylinder_dist.npy")
    plt.rcParams.update({"font.size": 20})
    max_log_p = 15
    color_shift = 50
    alpha_shift = 125
    font = ImageFont.truetype("arial.ttf", 30)

    # Note: I had to use the pull request here: https://github.com/matplotlib/matplotlib/pull/23085
    # to get this to look right.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    rotvecs = Rotation.from_quat(cylinder_dist[:, :4]).as_rotvec()
    (xs, zs, ys) = np.split(rotvecs, 3, 1)

    log_ps = cylinder_dist[:, -1]
    print(log_ps[POS_SAMPLES + 1 :].max())
    colors = (log_ps + color_shift) / (color_shift + max_log_p)
    colors = colors.clip(0.0, 1)
    alphas = (log_ps + alpha_shift) / (alpha_shift + max_log_p)
    alphas = alphas.clip(0.0, 1) ** 3

    plot = ax.scatter(
        xs,
        ys,
        zs,
        depthshade=False,
        c=colors,
        s=100,
        alpha=alphas,
        vmin=0,
        vmax=1.0,
        cmap="coolwarm",
    )
    # See: https://stackoverflow.com/a/72928548/1316276.
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    for render_idx in cylinder_render_idxs:
        R = Rotation.from_rotvec(rotvecs[render_idx]).as_matrix()
        renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
        image = renderer.render(0.0, 0.0, 0.0)
        draw = ImageDraw.Draw(image)
        label = f"{render_idx}: {log_ps[render_idx]:.4f}"
        draw.text((0, 0), label, font=font, fill=(255, 255, 255))
        image.show()

        ax.text(xs[render_idx, 0], ys[render_idx, 0], zs[render_idx, 0], render_idx)

    cbar = fig.colorbar(plot, label=r"$\ln(p(\mathbf{q}))$")
    n_ticks = 51
    ticks = np.arange(n_ticks) / (n_ticks - 1)
    labels = np.round(np.linspace(-color_shift, max_log_p, n_ticks), 2)
    skips = 5
    cbar.set_ticks(ticks[::skips], labels=labels[::skips])
    ax.set_xlabel("$e_{x}$")
    ax.set_ylabel("$e_{z}$")
    ax.set_zlabel("$e_{y}$")
    plt.show()


def plot_cube_samples():
    rotvecs = np.load("cube_rotvecs.npy")
    (xs, zs, ys) = np.split(rotvecs, 3, 1)

    # Note: I had to use the pull request here: https://github.com/matplotlib/matplotlib/pull/23085
    # to get this to look right.
    plt.rcParams.update({"font.size": 20})
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    plot = ax.scatter(xs, ys, zs, depthshade=False)

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)
    # See: https://stackoverflow.com/a/72928548/1316276.
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    ax.set_xlabel("$e_{x}$")
    ax.set_ylabel("$e_{z}$")
    ax.set_zlabel("$e_{y}$")

    plt.show()


def convert_labels_to_arr(labels):
    n_bins = configs["aquamam"]["toy"]["model_args"]["n_bins"]
    quat = 2 * ((np.array(labels) + 0.5) / n_bins - 0.5)
    qw = np.sqrt(1 - np.sum(quat**2))
    qw = 0 if np.isnan(qw) else qw
    quat = np.array(list(quat) + [qw])
    quat /= np.linalg.norm(quat)
    return Rotation.from_quat(quat).as_matrix()


def convert_matrix_str_to_arr(matrix_str):
    matrix_str = matrix_str.replace("[", "").replace("]", "")
    return np.array(matrix_str.split(), dtype=float).reshape(3, 3)


def print_toy_statistics():
    dicts = pickle.load(open("aquamam_toy.pydict", "rb"))
    print("Incorrect: AQuaMaM")
    for (cat, incorrect_labels) in dicts["cat2incorrect_labels"].items():
        incorrect = sum(incorrect_labels.values())
        total = len(dicts["cat2best_R_dists"][cat])
        incorrect_per = 100 * incorrect / total
        print(f"{cat}: {incorrect}/{total} ({incorrect_per:.3f}%)")

    print()
    print("Average Error: AQuaMaM")
    all_R_dists = []
    cat_best_R_dists = list(dicts["cat2best_R_dists"].items())
    cat_best_R_dists.sort(key=lambda x: x[0])
    for (cat, best_R_dists) in cat_best_R_dists:
        # Convert to degrees.
        best_R_dists *= 180 / np.pi
        print(f"{cat}: {best_R_dists.mean():.3f}° ({best_R_dists.std():.3f})")
        all_R_dists.append(best_R_dists)

    all_R_dists = np.concatenate(all_R_dists)
    print(f"All: {all_R_dists.mean():.3f}° ({all_R_dists.std():.3f})")

    print()
    bad_cat = 5
    pprint.pprint(dicts["cat2labels_idxs"][bad_cat])
    print()
    incorrect_label_counts = list(dicts["cat2incorrect_labels"][bad_cat].items())
    incorrect_label_counts.sort(key=lambda x: x[1], reverse=True)
    pprint.pprint(incorrect_label_counts[:3])

    print()
    threshold = 5 / 180 * np.pi
    threshold_30 = 30 / 180 * np.pi
    cat2rot_counts = dicts["cat2rot_counts"]
    (_, _, test_loader) = load_dataloaders(
        "toy", {"model": "aquamam", "max_pow": 5}, 1, 1
    )
    cat2rots = test_loader.dataset.cat2rots
    cat2incorrect_at_30 = {cat: 0 for cat in cat2rot_counts}
    for (cat, incorrect_counts) in dicts["cat2incorrect_labels"].items():
        Rs = Rotation.from_quat(cat2rots[cat]).as_matrix()
        Rs = np.stack(Rs)
        for (labels, count) in incorrect_counts.items():
            R = convert_labels_to_arr(labels)
            R_diffs = R @ Rs.transpose(0, 2, 1)
            traces = np.trace(R_diffs, axis1=1, axis2=2)
            dists = np.arccos((traces - 1) / 2)
            closest_R = dists.argmin()
            if dists[closest_R] < threshold:
                cat2rot_counts[cat][closest_R] += count

            elif count > 5:
                Rs = np.concatenate([Rs, R[None]])
                cat2rot_counts[cat][len(cat2rot_counts[cat])] = count

            if dists[closest_R] > threshold_30:
                cat2incorrect_at_30[cat] += count

    pprint.pprint(cat2incorrect_at_30)
    both_cat2rot_counts = {"aquamam": cat2rot_counts}

    dicts = pickle.load(open("ipdf_toy.pydict", "rb"))
    print("Average Error: IPDF")
    cat_best_R_dists = list(dicts["cat2best_R_dists"].items())
    cat_best_R_dists.sort(key=lambda x: x[0])
    all_R_dists = []
    for (cat, best_R_dists) in cat_best_R_dists:
        # Convert to degrees.
        best_R_dists *= 180 / np.pi
        print(f"{cat}: {best_R_dists.mean():.3f}° ({best_R_dists.std():.3f})")
        all_R_dists.append(best_R_dists)

    all_R_dists = np.concatenate(all_R_dists)
    print(f"All: {all_R_dists.mean():.3f}° ({all_R_dists.std():.3f})")

    print()
    threshold = 5 / 180 * np.pi
    cat2rot_counts = {}
    for (cat, rot_counts) in dicts["cat2rot_counts"].items():
        Rs = None
        new_rots = []
        new_rot_counts = {}
        for rot in rot_counts.keys():
            R = convert_matrix_str_to_arr(rot)
            if Rs is None:
                Rs = R[None]
                new_rots.append(rot)
                new_rot_counts[rot] = rot_counts[rot]

            else:
                R_diffs = R @ Rs.transpose(0, 2, 1)
                traces = np.trace(R_diffs, axis1=1, axis2=2)
                dists = np.arccos((traces - 1) / 2)
                closest_R = dists.argmin()
                if dists[closest_R] < threshold:
                    new_rot_counts[new_rots[closest_R]] += rot_counts[rot]

                else:
                    Rs = np.concatenate([Rs, R[None]])
                    new_rots.append(rot)
                    new_rot_counts[rot] = rot_counts[rot]

        total_rots = sum(rot_counts.values())
        print(
            f"{cat}/{1 / 2**cat}: {total_rots}/{len(rot_counts)}/{len(new_rot_counts)}"
        )
        rot_counts = list(new_rot_counts.items())
        rot_counts.sort(key=lambda x: x[1], reverse=True)
        cat2rot_counts[cat] = {i: c for (i, (r, c)) in enumerate(rot_counts)}
        # rot_counts = [(k, v, v / total_rots) for (k, v) in rot_counts]
        # print(rot_counts[:5])

    both_cat2rot_counts["ipdf"] = cat2rot_counts

    img2scores = pickle.load(open("img2scores.pydict", "rb"))
    exp = np.exp(list(img2scores[5].values()))
    ps = exp / exp.sum()

    plt.rcParams.update({"font.size": 20})
    (rows, cols) = (2, 3)
    figsize = (11, 7)
    for model in ["AQuaMaM", "IPDF"]:
        (fig, axs) = plt.subplots(rows, cols, figsize=figsize)
        cat2rot_counts = both_cat2rot_counts[model.lower()]
        for (cat, rot_counts) in cat2rot_counts.items():
            for rot in range(2**cat):
                rot_counts.setdefault(rot, 0)

        for (idx, (cat, rot_counts)) in enumerate(cat2rot_counts.items()):
            (row_idx, col_idx) = (idx // cols, idx % cols)
            rots = [str(rot) for rot in rot_counts.keys()]
            counts = np.array(list(rot_counts.values()))
            props = counts / counts.sum()
            axs[row_idx, col_idx].bar(rots[: 2**cat], props[: 2**cat])
            label = "Uniform" if idx == 0 else None
            axs[row_idx, col_idx].axhline(
                1 / 2**cat, color="red", linestyle="--", label=label
            )
            axs[row_idx, col_idx].set_title(f"Category {cat}")
            if idx >= 4:
                xticks = axs[row_idx, col_idx].xaxis.get_major_ticks()
                for idx in range(len(xticks)):
                    if idx % 5 != 0:
                        xticks[idx].label1.set_visible(False)

        fig.suptitle(model)
        fig.supxlabel("Rotation")
        fig.supylabel("Proportion")
        fig.legend(loc="lower right")

        plt.subplots_adjust(
            left=0.12, right=0.98, top=0.86, bottom=0.15, wspace=0.41, hspace=0.37
        )
        plt.show()


def get_epoch_time_functions(details):
    total_time = details["total_time"]
    total_epochs = len(details["nlls"])

    def epoch2time(epoch):
        return epoch / total_epochs * total_time

    def time2epoch(time):
        return time / total_time * total_epochs

    return (epoch2time, time2epoch)


def plot_toy_losses():
    plt.rcParams.update({"font.size": 20})
    model_details = {
        "aquamam": {
            "min_nll": sum([np.log(2**i) for i in range(max_pow + 1)])
            / (max_pow + 1),
            "y_label": "Classification NLL",
            "name": "AQuaMaM",
        },
        "ipdf_train": {
            "min_nll": -np.log(4096 / np.pi**2),
            "y_label": "NLL",
            "name": "IPDF (Train)",
        },
        "ipdf_valid": {
            "min_nll": -np.log(2359296 / np.pi**2),
            "y_label": "NLL",
            "name": "IPDF (Valid)",
        },
    }

    for model in model_details:
        which_model = model.split("_")[0]
        with open(f"{which_model}_toy.log") as f:
            nlls = []
            for line in f:
                if line.startswith("valid_loss:"):
                    if model in {"aquamam", "ipdf_valid"}:
                        nlls.append(float(line.split(": ")[1]))

                elif line.startswith("train_loss") and (model == "ipdf_train"):
                    nlls.append(float(line.split(": ")[1]))

                elif line.startswith("real"):
                    time_str = line.split()[1][:-1]
                    (m, s) = (float(t) for t in time_str.split("m"))
                    model_details[model]["total_time"] = 60 * m + s

        model_details[model]["nlls"] = np.array(nlls)
        model_details[model]["top"] = model_details[model]["min_nll"] + 0.53
        model_details[model]["bottom"] = model_details[model]["min_nll"] - 0.05

    model2funcs = {}
    for (model, details) in model_details.items():
        model2funcs[model] = get_epoch_time_functions(details)

    (fig, axs) = plt.subplots(1, 3)
    for (idx, (model, details)) in enumerate(model_details.items()):
        nlls = details["nlls"]
        total_epochs = len(nlls)
        axs[idx].plot(
            np.arange(total_epochs), nlls, label=f"{details['name']} Evaluation"
        )
        min_nll = details["min_nll"]
        axs[idx].axhline(
            min_nll, color="red", linestyle="--", label="Theoretical Minimum"
        )
        axs[idx].set_xlabel("Epoch")
        axs[idx].set_ylim(top=details["top"], bottom=details["bottom"])
        axs[idx].set_ylabel(details["y_label"])
        (epoch2time, time2epoch) = model2funcs[model]
        secax = axs[idx].secondary_xaxis("top", functions=(epoch2time, time2epoch))
        secax.set_xlabel("Time (s)")
        axs[idx].legend()

    for (model, details) in model_details.items():
        details = model_details[model]
        total_time = details["total_time"]
        nlls = details["nlls"] - details["min_nll"]
        total_epochs = len(nlls)
        times = np.arange(total_epochs) / total_epochs * total_time
        axs[2].plot(times, nlls, label=details["name"])

    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("NLL - Theoretical Minimum NLL")
    axs[2].set_ylim(bottom=-0.05, top=0.53)
    axs[2].legend()
    plt.show()

    for model in ["ipdf_valid", "ipdf_train", "aquamam"]:
        details = model_details[model]
        total_time = details["total_time"]
        nlls = details["nlls"] - details["min_nll"]
        total_epochs = len(nlls)
        times = np.arange(total_epochs) / total_epochs * total_time
        if model == "ipdf_train":
            (times, nlls) = (times[:-1], nlls[:-1])

        plt.plot(times, nlls, label=details["name"])

    plt.xlabel("Time (s)")
    plt.ylabel("NLL - Theoretical Minimum NLL")
    plt.ylim(bottom=-0.05, top=0.53)
    plt.legend()
    plt.show()


def plot_likelihoods():
    plt.rcParams.update({"font.size": 20})

    grid_sizes = 2 ** np.arange(31)
    ipdf_max_ll = np.log(2359296 / np.pi**2)
    gpt_tokens = 50257
    aquamam_max_ll = 3 * (np.log(gpt_tokens) - np.log(2))
    n_bins = grid_sizes

    (fig, axs) = plt.subplots(2, 1)

    lls = 3 * np.log(n_bins) - 2 * np.log(2) - np.log(np.pi)
    axs[0].plot(n_bins, lls)

    axs[0].axhline(aquamam_max_ll, color="green", linestyle="--")
    axs[0].axhline(ipdf_max_ll, color="red", linestyle="--")

    lls = np.log(grid_sizes / np.pi**2)
    axs[0].plot(grid_sizes, lls)

    axs[0].set_xlim(-1000, gpt_tokens)
    axs[0].set_ylim(0, 30)

    lls = 3 * np.log(n_bins) - 2 * np.log(2) - np.log(np.pi)
    axs[1].plot(n_bins, lls, label="AQuaMaM Bins")

    axs[1].axhline(
        aquamam_max_ll, color="green", linestyle="--", label="AQuaMaM Max-LL @ GPT-3"
    )
    axs[1].axhline(
        ipdf_max_ll, color="red", linestyle="--", label="Murphy et al. (2021) Max-LL"
    )

    lls = np.log(grid_sizes / np.pi**2)
    axs[1].plot(grid_sizes, lls, label="IPDF Grid Size")

    fig.supxlabel("N")
    fig.supylabel("Max-LL")
    fig.legend(loc="upper right", prop={"size": 12})

    plt.show()


def unimodal_example():
    # Render object in default pose.
    renderer = Renderer(
        camera_distance=8,
        angle_of_view=ANGLE_OF_VIEW / 2,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    obj_mtl_path = "cube_obj/cube"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
    renderer.render(0.0, 0.0, 0.0).show()

    true_angles = [("Y", np.pi / 4), ("X", np.pi / 4)]
    true_quats = []
    for (axis, angle) in true_angles:
        r = Rotation.from_euler(axis, angle)
        R = r.as_matrix()
        renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
        renderer.render(0.0, 0.0, 0.0).show()
        true_quats.append(r.as_quat())

    true_quats = torch.Tensor(np.stack(true_quats))

    pre_quat = nn.Parameter(torch.randn(4))
    pre_dZ = nn.Parameter(torch.randn(3))

    optimizer = optim.Adam([pre_quat, pre_dZ], lr=1e-1)

    epochs = 1001
    for epoch in range(epochs):
        quat = F.normalize(pre_quat, dim=0)
        quats = quat.unsqueeze(0).repeat(2, 1)

        # See: https://github.com/Multimodal3DVision/torch_bingham/blob/master/cam_reloc/CamPoseNet.py#L88.
        dZ = F.softplus(pre_dZ)
        Z = torch.cumsum(dZ, dim=0)
        lambdas = -1 * Z.clamp(1e-12, 900)
        lambdas = lambdas.unsqueeze(0).repeat(2, 1)

        # See: https://github.com/Multimodal3DVision/torch_bingham/blob/master/cam_reloc/Losses.py.
        loss = -torch_bingham.bingham_prob(quats, lambdas, true_quats).mean()
        if epoch % 100 == 0:
            print(epoch)
            print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    quat = F.normalize(pre_quat, dim=0).detach()
    pred_R = Rotation.from_quat(quat).as_matrix()
    renderer.prog["R_obj"].write(pred_R.T.astype("f4").tobytes())
    renderer.render(0.0, 0.0, 0.0).show()


def alt_unimodal_example():
    # Render object in default pose.
    renderer = Renderer(
        camera_distance=8,
        angle_of_view=ANGLE_OF_VIEW / 2,
        dir_light=DIR_LIGHT,
        dif_int=DIF_INT,
        amb_int=AMB_INT,
        default_width=WINDOW_SIZE,
        default_height=WINDOW_SIZE,
        cull_faces=CULL_FACES,
    )
    obj_mtl_path = "cube_obj/cube"
    renderer.set_up_obj(f"{obj_mtl_path}.obj", f"{obj_mtl_path}.mtl")
    renderer.render(0.0, 0.0, 0.0).show()

    true_angles = [("Z", np.pi / 4), ("Z", -np.pi / 4)]
    true_quats = []
    for (axis, angle) in true_angles:
        r = Rotation.from_euler(axis, angle)
        R = r.as_matrix()
        renderer.prog["R_obj"].write(R.T.astype("f4").tobytes())
        renderer.render(0.0, 0.0, 0.0).show()
        true_quats.append(r.as_quat())

    true_quats = torch.Tensor(np.stack(true_quats))

    pre_quat = nn.Parameter(torch.randn(4))
    pre_dZ = nn.Parameter(torch.randn(3))

    optimizer = optim.Adam([pre_quat, pre_dZ], lr=1e-1)

    epochs = 1001
    for epoch in range(epochs):
        quat = F.normalize(pre_quat, dim=0)
        quats = quat.unsqueeze(0).repeat(2, 1)

        # See: https://github.com/Multimodal3DVision/torch_bingham/blob/master/cam_reloc/CamPoseNet.py#L88.
        dZ = F.softplus(pre_dZ)
        Z = torch.cumsum(dZ, dim=0)
        lambdas = -1 * Z.clamp(1e-12, 900)
        lambdas = lambdas.unsqueeze(0).repeat(2, 1)

        # See: https://github.com/Multimodal3DVision/torch_bingham/blob/master/cam_reloc/Losses.py.
        loss = -torch_bingham.bingham_prob(quats, lambdas, true_quats).mean()
        if epoch % 100 == 0:
            print(epoch)
            print(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    quat = F.normalize(pre_quat, dim=0).detach()
    pred_R = Rotation.from_quat(quat).as_matrix()
    renderer.prog["R_obj"].write(pred_R.T.astype("f4").tobytes())
    renderer.render(0.0, 0.0, 0.0).show()


def plot_dilution_factors():
    plt.rcParams.update({"font.size": 20})
    N = 1000000
    quats = Rotation.random(N).as_quat()
    # q and -q specify the same rotation, so I force real part to be
    # non-negative. See: https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html.
    quats[quats[:, -1] < 0] = -quats[quats[:, -1] < 0]
    sns.ecdfplot(-np.log(quats[:, -1]))
    plt.xlabel("$-\ln(q_{w})$")
    plt.show()


def get_quaternion_dists():
    for N in [500, 50257]:
        qx = 1 - 2 / N
        q1 = np.array([qx, 0, 0, np.sqrt(1 - qx**2)])
        q2 = np.array([1, 0, 0, 0])
        print(180 * np.arccos(2 * (q1 @ q2) ** 2 - 1) / np.pi)

        qx = qy = qz = -1 / N
        qw = np.sqrt(1 - qx**2 - qy**2 - qz**2)
        q1 = np.array([qx, qy, qz, qw])
        q2 = np.array([-qx, -qy, -qz, qw])
        print(180 * np.arccos(2 * (q1 @ q2) ** 2 - 1) / np.pi)
