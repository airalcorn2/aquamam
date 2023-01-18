import numpy as np
import pandas as pd

from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NORMS = {
    "ipdf": transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "aquamam": transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}
SEED = 2010


def load_dataloaders(dataset, model_details, batch_size, num_workers):
    assert dataset in {"toy", "cylinder", "cube"}

    if dataset == "toy":
        train_dataset = ToyDataset("train", model_details)
        valid_dataset = ToyDataset("test", model_details)
        test_dataset = valid_dataset
        valid_batch_size = 1 if model_details["model"] == "ipdf" else batch_size

    else:
        data_dir = dataset
        train_dataset = SolidDataset(data_dir, "train", model_details)
        valid_dataset = SolidDataset(data_dir, "valid", model_details)
        test_dataset = SolidDataset(data_dir, "test", model_details)
        valid_batch_size = batch_size

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, batch_size=valid_batch_size, num_workers=num_workers
    )
    test_batch_size = 1 if model_details["model"] == "ipdf" else batch_size
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, num_workers=num_workers
    )
    return (train_loader, valid_loader, test_loader)


def get_ipdf_sample(neg_samples, R, img):
    R_fake_Rs = np.concatenate([R[None], Rotation.random(neg_samples).as_matrix()])
    return (img, R_fake_Rs.reshape(-1, 9))


def get_aquamam_sample(quat, img):
    return (img, quat)


class ToyDataset(Dataset):
    def __init__(self, dataset, model_details):
        self.model = model = model_details["model"]
        assert model in {"ipdf", "aquamam"}
        if model == "ipdf":
            assert model_details["neg_samples"] > 0
            self.neg_samples = model_details["neg_samples"]

        max_pow = model_details["max_pow"]
        n_modes = [2**i for i in range(max_pow + 1)]
        max_modes = n_modes[-1]
        cat2rots = {}
        idx2cat_idx = []
        np.random.seed(SEED)
        for (cat, modes) in enumerate(n_modes):
            r = Rotation.random(modes)

            if model == "ipdf":
                cat2rots[cat] = r.as_matrix()

            else:
                quats = r.as_quat()
                # q and -q specify the same rotation, so I force the real part to be
                # non-negative. See: https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html.
                quats[quats[:, -1] < 0] = -quats[quats[:, -1] < 0]
                cat2rots[cat] = quats.astype("float32")

            cats = max_modes * [cat]
            rep = max_modes // modes
            idxs = rep * list(range(modes))
            cat_idxs = list(zip(cats, idxs))
            idx2cat_idx.extend(cat_idxs)

        self.is_train = dataset == "train"
        self.cat2rots = cat2rots
        self.cats = list(cat2rots)
        self.idx2cat_idx = idx2cat_idx

    def __len__(self):
        return 40000 if self.is_train else len(self.idx2cat_idx)

    def __getitem__(self, idx):
        if self.is_train:
            cat = np.random.choice(self.cats)
            rots = self.cat2rots[cat]
            idx = np.random.randint(len(rots))

        else:
            (cat, idx) = self.idx2cat_idx[idx]
            rots = self.cat2rots[cat]

        if self.model == "ipdf":
            return get_ipdf_sample(self.neg_samples, rots[idx], cat)

        else:
            return get_aquamam_sample(rots[idx], cat)


class SolidDataset(Dataset):
    def __init__(self, data_dir, dataset, model_details):
        self.model = model = model_details["model"]
        assert model in {"ipdf", "aquamam"}
        if model == "ipdf":
            assert model_details["neg_samples"] > 0
            self.neg_samples = model_details["neg_samples"]

        self.data_dir = data_dir
        self.dataset = dataset
        self.df = pd.read_csv(f"{data_dir}/metadata_{dataset}.csv")

        r = Rotation.from_euler("YXZ", self.df[["yaw", "pitch", "roll"]])
        self.preprocess = transforms.Compose([transforms.ToTensor(), NORMS[model]])

        if model == "ipdf":
            self.df["R"] = list(r.as_matrix())

        else:
            quats = r.as_quat()
            # q and -q specify the same rotation, so I force the real part to be
            # non-negative. See: https://marc-b-reynolds.github.io/quaternions/2017/11/10/AveRandomRot.html.
            quats[quats[:, -1] < 0] = -quats[quats[:, -1] < 0]
            self.df["quat"] = list(quats.astype("float32"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.data_dir}/images/{self.dataset}/{row['img_f']}")
        img = self.preprocess(img)

        if self.model == "ipdf":
            return get_ipdf_sample(self.neg_samples, row["R"], img)

        else:
            return get_aquamam_sample(row["quat"], img)
