import torch
import torch.nn as nn
import torchvision.models as tvm


def _get_weights(name, pretrained):
    if not pretrained:
        return None
    try:
        if name == 'efficientnet_b0':
            return tvm.EfficientNet_B0_Weights.DEFAULT
        if name == 'efficientnet_b2':
            return tvm.EfficientNet_B2_Weights.DEFAULT
        if name == 'efficientnet_v2_s':
            return tvm.EfficientNet_V2_S_Weights.DEFAULT
        if name == 'convnext_tiny':
            return tvm.ConvNeXt_Tiny_Weights.DEFAULT
        if name == 'resnet50':
            return tvm.ResNet50_Weights.DEFAULT
    except AttributeError:
        # Older torchvision fallback; model constructors may accept pretrained=True.
        return 'OLD_TORCHVISION'
    return None


def create_model(model_name, num_classes, pretrained=True):
    weights = _get_weights(model_name, pretrained)
    old = weights == 'OLD_TORCHVISION'

    if model_name == 'efficientnet_b0':
        model = tvm.efficientnet_b0(pretrained=pretrained) if old else tvm.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == 'efficientnet_b2':
        model = tvm.efficientnet_b2(pretrained=pretrained) if old else tvm.efficientnet_b2(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == 'efficientnet_v2_s':
        model = tvm.efficientnet_v2_s(pretrained=pretrained) if old else tvm.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == 'convnext_tiny':
        model = tvm.convnext_tiny(pretrained=pretrained) if old else tvm.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    if model_name == 'resnet50':
        model = tvm.resnet50(pretrained=pretrained) if old else tvm.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError('Unsupported model_name: %s' % model_name)


def set_backbone_trainable(model, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    # Re-enable classifier parameters.
    if hasattr(model, 'classifier'):
        for p in model.classifier.parameters():
            p.requires_grad = True
    if hasattr(model, 'fc'):
        for p in model.fc.parameters():
            p.requires_grad = True


def load_flexible_checkpoint(model, checkpoint_path, device='cpu'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    current = model.state_dict()
    loaded = {}
    skipped = []
    for k, v in state.items():
        if k in current and current[k].shape == v.shape:
            loaded[k] = v
        else:
            skipped.append(k)
    current.update(loaded)
    model.load_state_dict(current)
    return {'loaded': len(loaded), 'skipped': len(skipped), 'skipped_keys': skipped[:20]}
