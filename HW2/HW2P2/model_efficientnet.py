from efficientnet_pytorch import EfficientNet
from utils.base import *


class EfficientNetB7(Model):
    def __init__(self, params):
        super(EfficientNetB7, self).__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b7', num_classes=params.output_channels,
                                          image_size=(params.size, params.size))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class EfficientNetB2(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b2', num_classes=params.output_channels,
                                          image_size=(params.size, params.size))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class EfficientNetB4(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b4', num_classes=params.output_channels,
                                          image_size=(params.size, params.size))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class EfficientNetB0(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0', num_classes=params.output_channels,
                                          image_size=(params.size, params.size))

    def forward(self, x: torch.Tensor):
        return self.net(x)
