from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F

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
        self.net._fc = None

    def forward(self, x: torch.Tensor):
        return torch.flatten(self.net.extract_features(x), start_dim=1)


class EfficientNetB2E(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b2', num_classes=2,
                                          image_size=(params.size, params.size))
        self.net._fc = None

    def forward(self, x: torch.Tensor):
        return torch.flatten(self.net.extract_features(x), start_dim=1)


class EfficientNetB0_512(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0', num_classes=512)

    def forward(self, x: torch.Tensor):
        return F.normalize(self.net(x))


# class EfficientNetB0C(Model):
#     def __init__(self, params):
#         super().__init__(params)
#         self.net = EfficientNet.from_name('efficientnet-b0', num_classes=params.feature_dims,
#                                           image_size=(params.size, params.size))
#         self.fc = nn.Linear(params.feature_dims, params.output_channels)
#
#     def forward(self, x: torch.Tensor):
#         features = torch.flatten(self.net.extract_features(x), start_dim=1)
#         return features, self.fc(features)


class EfficientNetB4C(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b4',
                                          num_classes=self.params.output_channels)

    def forward(self, x: torch.Tensor):
        features = torch.flatten(self.net._avg_pooling(self.net.extract_features(x)), start_dim=1)
        return F.normalize(features), self.net._fc(self.net._dropout(features))


class EfficientNetB0C(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0',
                                          num_classes=self.params.output_channels)

    def forward(self, x: torch.Tensor):
        features = torch.flatten(self.net._avg_pooling(self.net.extract_features(x)), start_dim=1)
        return F.normalize(features), self.net._fc(self.net._dropout(features))


class EfficientNetB0C512(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0',
                                          num_classes=512)
        self.fc = nn.Linear(512, self.params.output_channels)

    def forward(self, x: torch.Tensor):
        features = F.relu(self.net(x))
        return F.normalize(features), self.fc(features)


class EfficientNetB0CS(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0',
                                          num_classes=self.params.output_channels)
        self.feature_fc = nn.Linear(1280, self.params.feature_dims)

    def forward(self, x: torch.Tensor):
        features = torch.flatten(self.net._avg_pooling(self.net.extract_features(x)), start_dim=1)
        features_out = F.relu(self.feature_fc(features))
        return features_out, self.net._fc(self.net._dropout(features))


class EfficientNetB0CDS(Model):
    def __init__(self, params):
        super().__init__(params)
        self.net = EfficientNet.from_name('efficientnet-b0',
                                          num_classes=self.params.output_channels)
        self.feature_fc = nn.Linear(1280, self.params.feature_dims)

    def forward(self, x: torch.Tensor):
        features = torch.flatten(self.net._avg_pooling(self.net.extract_features(x)), start_dim=1)
        features = self.net._dropout(features)
        return F.relu(self.feature_fc(features)), self.net._fc(features)
