import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.spatial.transform import Rotation

plt.rcParams.update({"font.size": 20})


def one_dimension():
    xs = np.linspace(-1, 1, 500)
    ys = np.sqrt(1 - xs**2)
    skip = 50
    samp_xs = xs[skip::skip]
    samp_ys = ys[skip::skip]
    plt.plot(xs, ys)
    plt.hlines(0, -1, 1)
    plt.vlines(samp_xs, 0, samp_ys)
    plt.plot(samp_xs, samp_ys, "o", label="$S^{1}$ Manifold Sample")
    plt.plot(samp_xs, np.zeros_like(samp_xs), "o", label="$\mathbb{R}^{1}$ Sample")
    plt.axis("equal")
    plt.axis("off")
    plt.legend(loc="upper right")
    plt.show()

    N = 1000000
    thetas = np.random.uniform(0, np.pi, size=N)
    (fig, axs) = plt.subplots(2, 2)
    axs[0, 0].hist(thetas)
    axs[0, 0].set_title(r"Samples of $\theta$")
    axs[0, 0].set_xlabel(r"$\theta$")
    axs[0, 0].set_ylabel("Count")

    xs = np.cos(thetas)
    axs[0, 1].hist(xs)
    axs[0, 1].set_title(r"Samples of $x = \cos(\theta)$")
    axs[0, 1].set_xlabel(r"$x$")
    axs[0, 1].set_ylabel("Count")

    n_bins = 500
    bins = np.linspace(-1, 1, n_bins + 1)[1:]
    labels = np.digitize(xs, bins, right=True)
    (unique_labels, counts) = np.unique(labels, return_counts=True)
    ps = counts / counts.sum()
    entropy = -np.sum(ps * np.log(ps))
    print(entropy)

    # See: https://nathancarter.github.io/how2data/site/how-to-plot-discrete-probability-distributions-in-python-using-scipy/.
    axs[1, 1].plot(unique_labels, ps, "o")
    axs[1, 1].vlines(unique_labels, 0, ps)
    axs[1, 1].set_ylim(bottom=0)
    axs[1, 1].set_title("Bin Probabilities for $x$")
    axs[1, 1].set_xlabel("$x$")
    axs[1, 1].set_ylabel("$p$")

    new_thetas = np.linspace(0, np.pi, 100)
    new_xs = np.cos(new_thetas)

    fuxs = np.ones(len(new_thetas))
    fuys = -new_xs / np.sqrt(1 - new_xs**2)

    # See: https://en.wikipedia.org/wiki/Cross_product#Matrix_notation.
    es = np.eye(2)
    form_dets = es[0] * fuys[:, None] - es[1] * fuxs[:, None]

    # Same as: lengths = np.linalg.norm(fus, axis=1).
    lengths = np.linalg.norm(form_dets, axis=1)
    # Equals np.abs(-1 * np.sin(new_thetas)).
    cov_factors = 1 / lengths

    new_bins = np.digitize(new_xs, bins, right=True)
    bin_width = 2 / n_bins
    new_ps = ps[new_bins] * 1 / bin_width * cov_factors

    axs[1, 0].plot(new_thetas, new_ps, "o")
    axs[1, 0].vlines(new_thetas, 0, new_ps)
    axs[1, 0].hlines(1 / np.pi, 0, np.pi, color="red", label=r"$\frac{1}{\pi}$")
    axs[1, 0].set_ylim(bottom=0)
    axs[1, 0].set_title(
        r"$p_{\theta}(\theta) = p_{\mathrm{Bin}(x)}(\cos(\theta)) \frac{1}{w} |-\sin(\theta)|$"
    )
    axs[1, 0].set_xlabel(r"$\theta$")
    axs[1, 0].set_ylabel(r"$p_{\theta}(\theta)$")
    axs[1, 0].legend(loc="upper center")
    plt.show()


def fill_in_areas(n_bins, row, col, half, areas, area):
    # Top right quadrant.
    row = n_bins - row - 1
    col = col + half
    areas[row, col] = area
    # Top left quadrant.
    col = -col - 1
    areas[row, col] = area
    # Bottom left quadrant.
    row = -row - 1
    areas[row, col] = area
    # Bottom right quadrant.
    col = -col - 1
    areas[row, col] = area


def integral(a, b, c, eps):
    # See: https://math.stackexchange.com/a/1924936/614328.
    B = 0.5 * (b * np.sqrt(1 - b**2) + np.arcsin(b)) - c * b
    A = 0.5 * (a * np.sqrt(1 - a**2) + np.arcsin(a)) - c * a
    # Occasionally get a negative zero, which I'm assuming is a numerical issue.
    return max(B - A, eps)


def two_dimensions():
    N = 1000000
    # See: https://stats.stackexchange.com/questions/7977/how-to-generate-uniformly-distributed-points-on-the-surface-of-the-3-d-unit-sphe.
    points = np.random.normal(size=(N, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points = points[points[:, 2] > 0]

    samps = 1000
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*np.hsplit(points[:samps], 3))
    # See: https://stackoverflow.com/a/72928548/1316276.
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))
    plt.show()

    n_bins = 200
    bins = np.linspace(-1, 1, n_bins + 1)[1:]
    xy_labels = np.digitize(points[:, :2], bins, right=True)
    (unique_labels, counts) = np.unique(xy_labels, return_counts=True, axis=0)
    all_counts = np.zeros((n_bins, n_bins))
    for ((col, row), count) in zip(unique_labels, counts):
        all_counts[row, col] = count

    plt.imshow(all_counts, interpolation="none", extent=[-1, 1, -1, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Count")
    plt.show()

    ps = all_counts / all_counts.sum()
    entropy = -np.sum(ps[ps > 0] * np.log(ps[ps > 0]))
    print(entropy)

    bin_width = 2 / n_bins

    half = n_bins // 2
    bin_tops = bins[half:][::-1][:, None].repeat(half, axis=1)
    bin_bottoms = bins[half - 1 : -1][::-1][:, None].repeat(half, axis=1)
    bin_lefts = bins[half - 1 : -1][None].repeat(half, axis=0)
    bin_rights = bins[half:][None].repeat(half, axis=0)

    cand_xs = np.sqrt(1 - bin_tops**2)
    fully_contained = bin_rights <= cand_xs
    area = bin_width**2
    areas = np.zeros((n_bins, n_bins))
    for (row, col) in np.argwhere(fully_contained):
        fill_in_areas(n_bins, row, col, half, areas, area)

    intersecting = (bin_lefts < cand_xs) & (cand_xs < bin_rights)
    eps = bin_width**2
    for (row, col) in np.argwhere(intersecting):
        # Rectangular area.
        cand_x = cand_xs[row, col]
        rec_area = (cand_x - bin_lefts[row, col]) * bin_width
        # Circular area.
        (a, b, c) = (cand_x, bin_rights[row, col], bin_bottoms[row, col])
        cir_area = integral(a, b, c, eps)
        area = rec_area + cir_area
        fill_in_areas(n_bins, row, col, half, areas, area)

    cand_ys = np.sqrt(1 - bin_lefts**2)
    intersecting = (bin_bottoms < cand_ys) & (cand_ys <= bin_tops)
    for (row, col) in np.argwhere(intersecting):
        # Circular area.
        (a, b, c) = (bin_lefts[row, col], bin_rights[row, col], bin_bottoms[row, col])
        area = integral(a, b, c, eps)
        fill_in_areas(n_bins, row, col, half, areas, area)

    num = 1000
    new_xys = points[:num, :2]
    (xs, ys) = np.hsplit(new_xys, 2)
    (xs, ys) = (xs.flatten(), ys.flatten())

    # See "Dealing with 3D Manifolds" here:
    # https://www.cs.cornell.edu/courses/cs6630/2015fa/notes/pdf-transform.pdf.
    ones = np.ones_like(xs)
    zeros = np.zeros_like(xs)
    zs = np.sqrt(1 - xs**2 - ys**2)

    fus = np.stack([ones, zeros, -xs / zs]).T
    fvs = np.stack([zeros, ones, -ys / zs]).T
    fuvs = np.concatenate([fus, fvs], axis=1).reshape(-1, 2, 3)

    # See: https://en.wikipedia.org/wiki/Cross_product#Matrix_notation.
    es = np.eye(3)
    i_dets = np.linalg.det(fuvs[:, :, [1, 2]])[:, None]
    j_dets = np.linalg.det(fuvs[:, :, [0, 2]])[:, None]
    k_dets = np.linalg.det(fuvs[:, :, [0, 1]])[:, None]
    form_dets = es[0] * i_dets - es[1] * j_dets + es[2] * k_dets

    # Same as: areas = np.linalg.norm(np.cross(fus, fvs), axis=1).
    surf_areas = np.linalg.norm(form_dets, axis=1)
    cov_factors = 1 / surf_areas

    new_bins = np.digitize(new_xys, bins, right=True)
    bin_areas = areas[new_bins[:, 0], new_bins[:, 1]]
    new_ps = ps[new_bins[:, 0], new_bins[:, 1]] * 1 / bin_areas * cov_factors
    plt.scatter(np.arange(len(new_ps)), new_ps)
    bin_areas = bin_width**2
    new_ps = ps[new_bins[:, 0], new_bins[:, 1]] * 1 / bin_areas * cov_factors
    true_dens = 1 / (2 * np.pi)
    print(true_dens)
    plt.hlines(true_dens, 0, len(new_ps), color="red", label=r"$\frac{1}{2\pi}$")
    mean = np.mean(new_ps)
    print(mean)
    plt.hlines(mean, 0, len(new_ps), color="orange", label=r"$\mu$")
    plt.xlabel("Sample")
    plt.ylabel("$p(x, y, z)$")
    plt.legend()
    plt.show()


def three_dimensions():
    N = 1000000
    quats = Rotation.random(N).as_quat()
    # q and -q specify the same rotation, so I force real part to be
    # non-negative. See: https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html.
    quats[quats[:, -1] < 0] = -quats[quats[:, -1] < 0]

    n_bins = 30
    bins = np.linspace(-1, 1, n_bins + 1)[1:]
    xyz_labels = np.digitize(quats[:, :3], bins, right=True)
    (unique_labels, counts) = np.unique(xyz_labels, return_counts=True, axis=0)
    all_counts = np.zeros((n_bins, n_bins, n_bins))
    for ((col, row, dep), count) in zip(unique_labels, counts):
        all_counts[row, col, dep] = count

    ps = all_counts / all_counts.sum()
    entropy = -np.sum(ps[ps > 0] * np.log(ps[ps > 0]))
    print(entropy)

    num = 1000
    xyzs = quats[:num, :3]
    (xs, ys, zs) = np.hsplit(xyzs, 3)
    (xs, ys, zs) = (xs.flatten(), ys.flatten(), zs.flatten())

    ones = np.ones_like(xs)
    zeros = np.zeros_like(xs)
    ws = np.sqrt(1 - xs**2 - ys**2 - zs**2)

    fus = np.stack([ones, zeros, zeros, -xs / ws]).T
    fvs = np.stack([zeros, ones, zeros, -ys / ws]).T
    fts = np.stack([zeros, zeros, ones, -zs / ws]).T
    fuvts = np.concatenate([fus, fvs, fts], axis=1).reshape(-1, 3, 4)

    # See: https://en.wikipedia.org/wiki/Cross_product#Matrix_notation.
    i_dets = np.linalg.det(fuvts[:, :, [1, 2, 3]])[:, None]
    j_dets = np.linalg.det(fuvts[:, :, [0, 2, 3]])[:, None]
    k_dets = np.linalg.det(fuvts[:, :, [0, 1, 3]])[:, None]
    l_dets = np.linalg.det(fuvts[:, :, [0, 1, 2]])[:, None]
    es = np.eye(4)
    form_dets = es[0] * i_dets - es[1] * j_dets + es[2] * k_dets + es[3] * l_dets

    vols = np.linalg.norm(form_dets, axis=1)
    cov_factors = 1 / vols
    sns.kdeplot(np.log(cov_factors), bw_adjust=0.5)
    plt.show()

    new_bins = np.digitize(xyzs, bins, right=True)
    bin_width = 2 / n_bins
    new_ps = ps[new_bins[:, 0], new_bins[:, 1], new_bins[:, 2]] * 1 / bin_width**3
    new_ps = new_ps * cov_factors
    print(np.mean(new_ps))
    true_dens = 1 / np.pi**2
    print(true_dens)
    plt.scatter(np.arange(len(new_ps)), new_ps)
    plt.hlines(true_dens, 0, len(new_ps), color="red", label=r"$\frac{1}{\pi^{2}}$")
    plt.legend()
    plt.show()


def main():
    one_dimension()
    two_dimensions()
    three_dimensions()


if __name__ == "__main__":
    main()
