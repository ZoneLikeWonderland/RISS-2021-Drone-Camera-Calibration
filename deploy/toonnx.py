import torch
net = torch.load("best_test_error.pth", "cpu")

image = torch.randn(1, 3,  480, 640)

result = net(image)

print(result.shape)


torch.onnx.export(net, image, "card.onnx", output_names={"output"}, verbose=True, opset_version=11)
