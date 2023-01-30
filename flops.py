from thop import profile

net = MobileNetV3_Small()
torch.save(net.state_dict(),"x.pth")
x = torch.randn(1, 3, 112,112)
flops, params = profile(net, (x,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))