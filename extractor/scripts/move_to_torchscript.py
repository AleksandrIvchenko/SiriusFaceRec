import torch

from cface.models import ArcFaceExtractor


if __name__ == '__main__':
    device = torch.device('cpu')
    model = ArcFaceExtractor(n_classes=10178)
    ckpt = torch.load('models/arcface0.4-e10.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    output_pt = 'FeatureExtractor.pt'
    inputs = torch.randn(1, 3, 128, 128).to(device)
    traced_script_module = torch.jit.trace(model, inputs)
    traced_script_module.save(output_pt)

    print("Exported to PT")
