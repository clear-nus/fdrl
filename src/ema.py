import torch.nn as nn

# modified from https://github.com/ermongroup/ncsnv2/blob/master/models/ema.py
class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.state_dict().items():
            self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.state_dict().items():
            self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.state_dict().items():
            param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            device_list = inner_module.config['device']
            device = "cuda:" + str(device_list[0])
            module_copy = type(inner_module)(inner_module.config).to(device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy, device_ids = device_list)
        else:
            device = module.config['device']
            module_copy = type(module)(module.config).to(device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

    def to(self, device):
        for name, param in self.shadow.items():
            self.shadow[name] = param.to(device)