import torch
from torch import nn

import numpy as np


class SpatialEncoder(nn.Module):
    def __init__(self, h_dim, input_shapes, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.input_shapes = input_shapes  # dict of {key: (num_classes, H, W)}
        self.embedding_dim = 32

        # embedding layers and convolutional encoders for each key
        self.encoders = nn.ModuleDict()
        for key, (num_classes, _, _) in input_shapes.items():
            # embedding layer for the key
            embedding = nn.Embedding(
                num_embeddings=num_classes, embedding_dim=self.embedding_dim
            ).to(device)

            # convolutional encoder for the key
            conv = nn.Sequential(
                nn.Conv2d(self.embedding_dim, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.h_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            ).to(device)

            # combine embedding and conv into a sequential module
            self.encoders[key] = nn.ModuleDict({"embedding": embedding, "conv": conv})

    def forward(self, inputs):
        features = []
        for key, modules in self.encoders.items():
            embedding = modules["embedding"]
            conv = modules["conv"]

            x = inputs[key].long().to(self.device)  # (B, H, W)
            x_embedded = embedding(x)  # (B, H, W, E)
            x_embedded = x_embedded.permute(0, 3, 1, 2)  # (B, E, H, W)
            x_feature = conv(x_embedded)  # (B, h_dim, 1, 1)
            x_feature = x_feature.view(x_feature.size(0), -1)  # (B, h_dim)

            features.append(x_feature)

        combined_features = torch.cat(features, dim=1)  # (B, h_dim * num_keys)
        return combined_features


class SpatialDecoder(nn.Module):
    def __init__(self, h_dim, output_shapes, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.output_shapes = output_shapes  # dict of {key: (num_classes, H, W)}
        # tODO this should probably be an argument
        self.embedding_dim = 32

        # linear layers and deconvolutional decoders for each key
        self.decoders = nn.ModuleDict()
        for key, (num_classes, H, W) in output_shapes.items():
            # compute the required number of upsampling layers to reach the desired H and W
            upsample_layers = self._compute_upsample_layers(H, W)

            # linear layer to expand latent vector
            fc = nn.Sequential(
                nn.Linear(h_dim, 128 * upsample_layers["start_size"] ** 2),
                nn.ReLU(),
            ).to(device)

            deconv_layers = []
            in_channels = 128
            for i in range(upsample_layers["num_layers"]):
                out_channels = (
                    64 if i < upsample_layers["num_layers"] - 1 else self.embedding_dim
                )
                deconv_layers.append(
                    nn.ConvTranspose2d(
                        in_channels, out_channels, kernel_size=4, stride=2, padding=1
                    )
                )
                deconv_layers.append(nn.ReLU())
                in_channels = out_channels

            deconv = nn.Sequential(*deconv_layers).to(device)
            # output layer to produce logits
            output_layer = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1).to(
                device
            )

            # combine into a module
            self.decoders[key] = nn.ModuleDict(
                {
                    "fc": fc,
                    "deconv": deconv,
                    "output_layer": output_layer,
                    "output_shape": (H, W),
                }
            )

    def forward(self, x):
        outputs = {}
        for key, modules in self.decoders.items():
            fc = modules["fc"]
            deconv = modules["deconv"]
            output_layer = modules["output_layer"]
            H, W = modules["output_shape"]
            upsample_info = self._compute_upsample_layers(H, W)

            # expand latent vector
            x_fc = fc(x)  # (B, 128 * start_size * start_size)
            x_fc = x_fc.view(
                x.size(0), 128, upsample_info["start_size"], upsample_info["start_size"]
            )
            # pass through deconvolutional layers
            x_deconv = deconv(x_fc)
            # output layer
            logits = output_layer(x_deconv)
            # adjust dimensions to match output_shape
            logits = logits[:, :, :H, :W]
            outputs[key] = logits  # (B, num_classes, H, W)

        return outputs

    def _compute_upsample_layers(self, H, W):
        """Computes the number of upsampling layers needed based on H and W."""
        num_layers = max(
            int(np.ceil(np.log2(max(H, W))) - 2), 1
        )  # subtracting 2 for the initial size
        start_size = H // (2**num_layers)
        return {"num_layers": num_layers, "start_size": start_size}


class InventoryEncoder(nn.Module):
    def __init__(self, h_dim, inv_shapes, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.inv_shapes = inv_shapes  # dict of {key: (shape, num_classes)}
        # TODO as argument, probably
        self.embedding_dim = 32

        # embedding layers and encoders for each key
        self.encoders = nn.ModuleDict()
        for key, (shape, num_classes) in self.inv_shapes.items():
            # embedding layer for the key
            embedding = nn.Embedding(
                num_embeddings=num_classes, embedding_dim=self.embedding_dim
            ).to(device)
            # linear layer to produce feature vector
            fc = nn.Sequential(
                nn.Linear(np.prod(shape) * self.embedding_dim, h_dim),
                nn.ReLU(),
            ).to(device)

            self.encoders[key] = nn.ModuleDict({"embedding": embedding, "fc": fc})

    def forward(self, inputs):
        features = []
        for key, modules in self.encoders.items():
            x = inputs[key].long().to(self.device)
            embedding = modules["embedding"]
            fc = modules["fc"]
            x_embedded = embedding(x)  # (B, ..., E)
            x_embedded = x_embedded.view(
                x_embedded.size(0), -1
            )  # flatten to (B, N * E)
            x_feature = fc(x_embedded)  # (B, h_dim)
            features.append(x_feature)

        combined_features = torch.cat(features, dim=1)  # (B, h_dim * num_keys)
        return combined_features


class InventoryDecoder(nn.Module):
    def __init__(self, h_dim, inv_shapes, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.inv_shapes = inv_shapes  # dict of {key: (shape, num_classes)}
        self.embedding_dim = 32

        # decoders for each key
        self.decoders = nn.ModuleDict()
        for key, (shape, num_classes) in self.inv_shapes.items():
            # linear layer to map h_dim to output logits
            output_dim = int(np.prod(shape)) * num_classes
            decoder = nn.Sequential(
                nn.Linear(h_dim, output_dim),
                # output reshaping and activation will be handled in forward
            ).to(device)
            self.decoders[key] = decoder

    def forward(self, x):
        outputs = {}

        # compute h_dim_total and split x accordingly
        h_dim = self.h_dim
        num_keys = len(self.decoders)
        split_sizes = [h_dim] * num_keys
        x_split = torch.split(
            x, split_sizes, dim=1
        )  # list of tensors of shape (B, h_dim)
        for (key, decoder), x_key in zip(self.decoders.items(), x_split):
            logits = decoder(x_key)  # (B, output_dim)
            shape, num_classes = self.inv_shapes[key]
            logits = logits.view(
                -1, *shape, num_classes
            )  # reshape to (B, ..., num_classes)
            outputs[key] = logits
        return outputs


class MessageEncoder(nn.Module):
    def __init__(self, h_dim, message_length, vocab_size, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=32).to(
            device
        )
        self.fc = nn.Sequential(
            nn.Linear(message_length * 32, h_dim),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        x = x.long().to(self.device)  # (B, message_length)
        x_embedded = self.embedding(x)  # (B, message_length, E)
        x_embedded = x_embedded.view(
            x.size(0), -1
        )  # flatten to (B, message_length * E)
        x = self.fc(x_embedded)  # (B, h_dim)
        return x  # (B, h_dim)


class MessageDecoder(nn.Module):
    def __init__(self, h_dim, message_length, vocab_size, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.message_length = message_length
        self.vocab_size = vocab_size

        self.decoder = nn.Linear(h_dim, message_length * vocab_size).to(device)

    def forward(self, x):
        logits = self.decoder(x)
        logits = logits.view(
            -1, self.message_length, self.vocab_size
        )  # (B, message_length, vocab_size)
        return logits


class BlstatsEncoder(nn.Module):
    def __init__(self, h_dim, blstats_size, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim

        self.fc = nn.Sequential(
            nn.Linear(blstats_size, h_dim),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        x = x.float().to(self.device)
        x = self.fc(x)  # (B, h_dim)
        return x  # (B, h_dim)


class BlstatsDecoder(nn.Module):
    def __init__(self, h_dim, blstats_size, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.blstats_size = blstats_size

        self.decoder = nn.Linear(h_dim, blstats_size).to(device)

    def forward(self, x):
        blstats_recon = self.decoder(x)  # (B, blstats_size)
        return blstats_recon


class ScreenDescriptionsEncoder(nn.Module):
    def __init__(self, h_dim, input_shape, vocab_size, embedding_dim, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.input_shape = input_shape  # (H, W, D)
        self.vocab_size = vocab_size

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        ).to(device)

        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=self.embedding_dim,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, h_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # global average pooling
        ).to(device)

    def forward(self, x):
        B = x.size(0)

        x = x.long().to(self.device)  # (B, H, W, D)
        x = self.embedding(x)  # (B, H, W, D, E)
        x = x.permute(0, 4, 1, 2, 3)  # (B, E, H, W, D)
        x = self.conv(x)  # (B, h_dim, 1, 1, 1)
        x = x.view(B, -1)  # flatten to (B, h_dim)
        return x


class ScreenDescriptionsDecoder(nn.Module):
    def __init__(self, h_dim, output_shape, vocab_size, embedding_dim, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.output_shape = output_shape  # (H, W, D)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # linear layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(h_dim, 128 * 2 * 2 * 2),
            nn.ReLU(),
        ).to(device)

        # deconvolutional layers
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(
                64, self.embedding_dim, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
        ).to(device)

        # output layer to produce logits over vocab_size
        self.output_layer = nn.Conv3d(
            self.embedding_dim, self.vocab_size, kernel_size=1
        ).to(device)

    def forward(self, x):
        B = x.size(0)

        x = self.fc(x)  # (B, 128 * 2 * 2 * 2)
        x = x.view(B, 128, 2, 2, 2)  # reshape to start deconvolution
        x = self.deconv(x)  # shape will increase with each layer

        logits = self.output_layer(x)  # (B, vocab_size, H, W, D)
        logits = logits[
            :, :, : self.output_shape[0], : self.output_shape[1], : self.output_shape[2]
        ]
        return logits


class TtyCursorEncoder(nn.Module):
    def __init__(self, h_dim, device):
        super().__init__()
        self.device = device
        self.h_dim = h_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, h_dim),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        x = x.float().to(self.device)  # (B, 2)
        x = self.encoder(x)  # (B, h_dim)
        return x


class TtyCursorDecoder(nn.Module):
    def __init__(self, h_dim, device):
        super().__init__()
        self.device = device

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # output between 0 and 1
        ).to(device)

    def forward(self, x):
        x = x.to(self.device)  # (B, h_dim)
        x = self.decoder(x) * 255  # (B, 2), scaled to [0, 255]
        return x
