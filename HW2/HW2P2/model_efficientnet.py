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


class PretrainedB7(Model):
    def __init__(self, params):
        super(PretrainedB7, self).__init__(params)
        self.net = EfficientNet.from_pretrained('efficientnet-b7',
                                                num_classes=params.output_channels)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PretrainedB2(Model):
    def __init__(self, params):
        super(PretrainedB2, self).__init__(params)
        self.net = EfficientNet.from_pretrained('efficientnet-b2',
                                                num_classes=params.output_channels)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PretrainedB0(Model):
    def __init__(self, params):
        super(PretrainedB0, self).__init__(params)
        self.net = EfficientNet.from_pretrained('efficientnet-b0',
                                                num_classes=params.output_channels)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class EfficientNetB0E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0', num_classes=params.output_channels,
                                          image_size=(params.size, params.size))
        self._fc = None

    def forward(self, x: torch.Tensor):
        return torch.flatten(self.net.extract_features(x), start_dim=1)


class PretrainedB0_512(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_pretrained('efficientnet-b0', num_classes=512)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class EfficientNetB0C(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0', num_classes=params.feature_dims,
                                          image_size=(params.size, params.size))
        self.fc = nn.Linear(params.feature_dims, params.output_channels)

    def forward(self, x: torch.Tensor):
        features = self.net(x)
        return features, self.fc(torch.relu(features))
