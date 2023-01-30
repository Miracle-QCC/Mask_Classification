import torch
from GhostNet import ghostnet
model = ghostnet( num_classes=1)
ckpt = torch.load('ghostnet_last.pth')
model.load_state_dict(ckpt)
model.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 3, 112, 112, requires_grad=True)

torch.onnx.export(model, x, 'ghostnet_112x112.onnx', input_names=input_names, output_names=output_names, verbose='True')