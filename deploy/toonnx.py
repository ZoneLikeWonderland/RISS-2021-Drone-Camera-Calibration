import torch
import sys
sys.path.append("..")

net = torch.load(r"E:\RISS\runs\Jul15_13-50-01_DESKTOP-HN2581Fback\best_test_error.pth", "cpu")

image = torch.randn(1, 3, 480, 640)

result = net(image)

print(result.shape)


torch.onnx.export(net, image, "card.onnx", output_names={"output"}, verbose=True, opset_version=11)
