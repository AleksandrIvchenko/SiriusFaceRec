import torch

from cface.models import ArcFaceExtractor


if __name__ == '__main__':
    device = torch.device('cpu')
    net = ArcFaceExtractor()
    output_pt = 'FeatureExtractor.pt'
    inputs = torch.randn(1, 3, 128, 128).to(device)
    traced_script_module = torch.jit.trace(net, inputs)
    traced_script_module.save(output_pt)

    print("Exported to PT")
