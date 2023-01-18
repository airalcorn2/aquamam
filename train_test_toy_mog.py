import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

LNSQRT2PI = np.log(np.sqrt(2 * np.pi))


# Not used in this script. Only for review purposes.
def sample_aquamam_qcs(qs, x, which_qc):
    n_comps = x.shape[-1] // 3
    probs = torch.softmax(x[:, :n_comps], dim=1)
    comps = torch.multinomial(probs, 1).flatten()

    idxs = torch.arange(len(comps))
    mus = x[:, n_comps : 2 * n_comps][idxs, comps]
    sds = torch.exp(x[:, 2 * n_comps :][idxs, comps])
    ss = torch.normal(mus, sds)

    ubs = torch.sqrt(1 - torch.sum(qs[:, :which_qc] ** 2, dim=1))
    qcs = -ubs + 2 * ubs * torch.sigmoid(ss)
    return qcs


# Not used in this script. Only for review purposes.
def get_aquamam_mog_training_lls(x, qs):
    n_comps = x.shape[-1] // 3
    lls = 0
    for i in range(3):
        ubs = torch.sqrt(1 - torch.sum(qs[:, :i] ** 2, dim=1))

        qcs = qs[:, i]
        ss = torch.log(qcs + ubs) - torch.log(ubs - qcs)
        ss = ss.unsqueeze(1).repeat(1, n_comps)

        log_pis = torch.log_softmax(x[:, i, :n_comps], dim=1)

        mus = x[:, i, n_comps : 2 * n_comps]
        log_stds = x[:, i, 2 * n_comps :]
        log_ps = -log_stds - LNSQRT2PI - 0.5 * ((ss - mus) / torch.exp(log_stds)) ** 2

        lls += torch.logsumexp(log_pis + log_ps, dim=1)

    return lls


# Not used in this script. Only for review purposes.
def get_aquamam_mog_full_lls(x, qs):
    lls = get_aquamam_mog_training_lls(x, qs) + torch.log(qs[:, 3])
    for i in range(3):
        ubs = torch.sqrt(1 - torch.sum(qs[:, :i] ** 2, dim=1))
        qcs = qs[:, i]
        lls += torch.log(torch.abs(2 * ubs / (ubs**2 - qcs**2)))

    return lls


def get_lls(params, xs):
    n_comps = params.shape[1] // 3

    log_pis = torch.log_softmax(params[:, :n_comps], dim=1)

    mus = params[:, n_comps : 2 * n_comps]
    log_stds = params[:, 2 * n_comps :]
    xs = xs.unsqueeze(1).repeat(1, n_comps)
    log_ps = -log_stds - LNSQRT2PI - 0.5 * ((xs - mus) / torch.exp(log_stds)) ** 2

    lls = torch.logsumexp(log_pis + log_ps, dim=1)

    return lls


def train_mog():
    best_valid_loss = float("inf")
    patience = 5
    no_improvement = 0
    lr_drops = 0
    epochs = 10000
    for epoch in range(epochs):
        print(f"epoch: {epoch}")
        model.train()
        for (idx, xs) in enumerate(train_loader):
            xs = xs.to(device)
            preds = model(xs)
            loss = -get_lls(preds, xs).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 500 == 0:
                print(f"batch_loss: {loss.item() / len(xs)}", flush=True)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for xs in valid_loader:
                xs = xs.to(device)
                preds = model(xs)
                loss = -get_lls(preds, xs).sum()
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
            if no_improvement == patience:
                lr_drops += 1
                if lr_drops == 2:
                    break

                no_improvement = 0
                print("Reducing learning rate.")
                for g in optimizer.param_groups:
                    g["lr"] *= 0.5


def plot_data():
    mog_data = np.load("mog_data.npy")
    sns.kdeplot(np.array(mog_data))

    mog_samps = np.load("mog_samps.npy")
    sns.kdeplot(np.array(mog_samps))
    plt.show()


class MoGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MoG(nn.Module):
    def __init__(self, n_comps):
        super().__init__()
        self.params = nn.Parameter(torch.randn(3 * n_comps))

    def forward(self, xs):
        return self.params.unsqueeze(0).repeat(len(xs), 1)


if __name__ == "__main__":
    mus = [-2, 4, 5, 15]
    sds = [1.0, 0.3, 0.1, 2.0]
    (train_n, valid_n) = (500000, 10000)
    train_samps_per_mode = train_n // len(mus)
    valid_samps_per_mode = valid_n // len(mus)
    train_X = []
    valid_X = []
    for (idx, mu) in enumerate(mus):
        train_X.append(np.random.normal(mu, sds[idx], train_samps_per_mode))
        valid_X.append(np.random.normal(mu, sds[idx], train_samps_per_mode))

    train_dataset = MoGDataset(np.concatenate(train_X))
    valid_dataset = MoGDataset(np.concatenate(valid_X))
    batch_size = 128
    num_workers = 2
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=batch_size, num_workers=num_workers
    )

    n_comps = 256
    device = "cuda:0"
    model = MoG(n_comps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    params_f = "mog.pth"
    train_mog()

    np.save("mog_data.npy", train_dataset.data)

    model.load_state_dict(torch.load(params_f))
    model.eval()

    with torch.no_grad():
        params = model.params.unsqueeze(0).repeat(valid_n, 1)

        probs = torch.softmax(params[:, :n_comps], dim=1)
        comps = torch.multinomial(probs, 1).flatten()

        idxs = torch.arange(len(comps))
        mus = params[:, n_comps : 2 * n_comps][idxs, comps]
        sds = torch.exp(params[:, 2 * n_comps :][idxs, comps])
        samps = torch.normal(mus, sds)

    np.save("mog_samps.npy", samps.cpu().numpy())
