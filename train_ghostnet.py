from torch.optim import Adam
from torch.utils.data import DataLoader
from Data import *
from GhostNet import ghostnet
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.34, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def train():
    loss_fun = FocalLoss()
    train_dataset = MaskFaceDataset('train.txt', train=True)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=128,
    )
    model = ghostnet( num_classes=1)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=5e-7)
    model = model.to(device)
    for epoch in range(120):
        for imgs,labels in train_dataloader:
            imgs = imgs.to(device)
            label = labels.to(device).float()
            out = model(imgs)
            out = F.sigmoid(out)
            loss = loss_fun(out, label.unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
    torch.save(model.state_dict(), "ghostnet_last.pth")
if __name__ == '__main__':
    train()
