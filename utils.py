import copy

class AverageBase:
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None

    def __str__(self):
        return str(round(self.value, 4))

    def __repr__(self):
        return self.value

    def __format__(self, fmt):
        return self.value.__format__(fmt)

    def __float__(self):
        return self.value

class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    def __init__(self, alpha=0.5):
        super().__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value

def freeze_model(model, conv_func):
    freezed_model = copy.deepcopy(model)
    for name, module in freezed_model.named_modules():
        if isinstance(module, conv_func):
            module.alpha.requires_grad = False
    return freezed_model