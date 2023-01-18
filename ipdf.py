import torch

from torch import nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

resnets = {
    "resnet50": (resnet50, ResNet50_Weights),
    "resnet101": (resnet101, ResNet101_Weights),
}


class IPDF(nn.Module):
    def __init__(self, toy_args, resnet, L, n_hidden_nodes, mlp_layers):
        super().__init__()
        self.is_toy = toy_args["is_toy"]
        if self.is_toy:
            n_cats = toy_args["max_pow"] + 1
            visual_embedding_size = toy_args["visual_embedding_size"]
            self.cnn = nn.Embedding(n_cats, visual_embedding_size)

        else:
            # See: https://github.com/google-research/google-research/tree/master/implicit_pdf#reproducing-symsol-results
            # and Section S8.
            (m, w) = resnets[resnet]
            self.cnn = m(weights=w.IMAGENET1K_V1)
            visual_embedding_size = self.cnn.layer4[2].bn3.num_features

        self.img_linear = nn.Linear(visual_embedding_size, n_hidden_nodes)

        self.L = L
        R_feats = 2 * self.L * 9
        self.R_linear = nn.Linear(R_feats, n_hidden_nodes)
        mlp = [nn.ReLU()]
        for layer in range(mlp_layers - 1):
            mlp.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
            mlp.append(nn.ReLU())

        mlp.append(nn.Linear(n_hidden_nodes, 1))
        self.mlp = nn.Sequential(*mlp)

    def get_img_feats(self, imgs):
        if self.is_toy:
            x = self.cnn(imgs)

        else:
            x = self.cnn.conv1(imgs)
            x = self.cnn.bn1(x)
            x = self.cnn.relu(x)
            x = self.cnn.maxpool(x)

            x = self.cnn.layer1(x)
            x = self.cnn.layer2(x)
            x = self.cnn.layer3(x)
            x = self.cnn.layer4(x)

            x = self.cnn.avgpool(x)
            x = torch.flatten(x, 1)

        return x

    def get_scores(self, imgs, Rs):
        x = self.get_img_feats(imgs)

        Rs_encoded = []
        for l_pos in range(self.L):
            Rs_encoded.append(torch.sin(2**l_pos * torch.pi * Rs))
            Rs_encoded.append(torch.cos(2**l_pos * torch.pi * Rs))

        Rs_encoded = torch.cat(Rs_encoded, dim=-1)

        # See Equation (9) in Section S8 and:
        # https://github.com/google-research/google-research/blob/4d906a25489bb7859a88d982a6c5e68dd890139b/implicit_pdf/models.py#L120-L126.
        x = self.img_linear(x).unsqueeze(1) + self.R_linear(Rs_encoded)
        x = self.mlp(x).squeeze(2)

        return x

    def sample(self, imgs, Rs):
        probs = torch.softmax(self.get_scores(imgs, Rs), 1)
        R_idxs = torch.multinomial(probs, 1).flatten()
        return Rs[torch.arange(len(Rs)), R_idxs]

    def forward(self, imgs, Rs_fake_Rs):
        # See: https://github.com/google-research/google-research/blob/207f63767d55f8e1c2bdeb5907723e5412a231e1/implicit_pdf/models.py#L188
        # and Equation (2) in the paper.
        V = torch.pi**2 / Rs_fake_Rs.shape[1]
        probs = 1 / V * torch.softmax(self.get_scores(imgs, Rs_fake_Rs), 1)[:, 0]
        return probs
