import torch
import tqdm
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from Data import *
from model import *
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as ap
from GhostNetBase import ghostnet
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

class FocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
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


def fun():
    test_dataset = MaskFaceDataset('val.txt', train=False)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        shuffle=False,
        batch_size=1,
    )
    f = open("failcase.txt", 'w')
    model = ghostnet(num_classes=1).to(device)
    ckpt = torch.load("ghostnet_last.pth")
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    count = 0
    labels_ = np.array([],dtype=np.float32)
    outs_ = np.array([],dtype=np.float32)
    with torch.no_grad():
        for imgs,labels, img_path in tqdm.tqdm(test_dataloader):
            imgs = imgs.to(device)
            labels = labels.numpy()
            out = model(imgs)
            out = F.sigmoid(out)
            out = out.cpu().numpy().squeeze()
            pred = out
            pred = round(pred.item(),4)
            out = 1 if out > 0.5 else 0

            if out != labels[0]:
                f.write(img_path[0] + f" label:{labels[0]}, pred:{pred}" + "\n")
    f.close()

    # print('AP Score:', ap(labels_, outs_))
if __name__ == '__main__':
    fun()
