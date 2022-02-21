import copy

def track_ch(name, model):
    def hook(module, input, output):
        model.alive_ch[name] = output[1].clone().detach().cpu().numpy().sum()
    return hook

def track_complexity(name, model):
    def hook(module, input, output):
        model.size_dict[name] = output[2]
        model.ops_dict[name] = output[3]
    return hook

def register_hook(model, conv_func, *args):
    registered_model = copy.deepcopy(model)
    for name, module in registered_model.named_modules():
        if isinstance(module, conv_func):
            for arg in args:
                module.register_forward_hook(arg(name, registered_model))
    return registered_model
