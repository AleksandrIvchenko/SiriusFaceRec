import torch

from cface.models import ArcFaceExtractor


if __name__ == '__main__':
    device = torch.device('cpu')
    model = ArcFaceExtractor(n_classes=10178, head_mode='arcface')
    ckpt = torch.load('models/arcface1.5-e12.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    output_pt = 'FeatureExtractor.pt'
    inputs = torch.randn(1, 3, 128, 128).to(device)
    traced_script_module = torch.jit.trace(model, inputs)
    traced_script_module.save(output_pt)

    print("Exported to PT")
