import copy

def freeze_model(model, conv_func):
    freezed_model = copy.deepcopy(model)
    for name, module in freezed_model.named_modules():
        if isinstance(module, conv_func):
            module.alpha.requires_grad = False
    return freezed_model