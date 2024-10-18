from typing import Dict, Tuple

import torch
from torch import nn

import numpy as np


class SpatialEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        input_shapes: Dict[str, Tuple[int, Tuple[int, int]]],
        embedding_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.input_shapes = input_shapes  # dict of {key: (num_classes, H, W)}
        self.embedding_dim = embedding_dim

        # embedding layers and convolutional encoders for each key
        self.encoders = nn.ModuleDict()
        for key, (num_classes, _) in input_shapes.items():
            # embedding layer for the key
            embedding = nn.Embedding(
                num_embeddings=num_classes, embedding_dim=self.embedding_dim
            )

            # convolutional encoder for the key
            conv = nn.Sequential(
                nn.Conv2d(self.embedding_dim, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.h_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            )

            # combine embedding and conv into a module
            self.encoders[key] = nn.ModuleDict({"embedding": embedding, "conv": conv})

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for key, modules in self.encoders.items():
            embedding = modules["embedding"]
            conv = modules["conv"]

            x = inputs[key].long()  # (B, H, W)
            x_embedded = embedding(x)  # (B, H, W, E)
            x_embedded = x_embedded.permute(0, 3, 1, 2)  # (B, E, H, W)
            x_feature = conv(x_embedded)  # (B, h_dim, 1, 1)
            x_feature = x_feature.view(x_feature.size(0), -1)  # (B, h_dim)

            features.append(x_feature)

        combined_features = torch.cat(features, dim=1)  # (B, h_dim * num_keys)
        return combined_features


class SpatialDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        output_shapes: Dict[str, Tuple[int, Tuple[int, int]]],
        embedding_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.output_shapes = output_shapes  # dict of {key: (num_classes, H, W)}
        self.embedding_dim = embedding_dim

        # linear layers and deconvolutional decoders for each key
        self.decoders = nn.ModuleDict()
        for key, (num_classes, shape) in output_shapes.items():
            H, W = shape
            # compute the required number of upsampling layers to reach the desired H and W
            upsample_layers = self._compute_upsample_layers(H, W)

            # linear layer to expand latent vector
            fc = nn.Sequential(
                nn.Linear(h_dim, 128 * upsample_layers["start_size"] ** 2),
                nn.ReLU(),
            )

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

            deconv = nn.Sequential(*deconv_layers)
            # output layer to produce logits
            output_layer = nn.Conv2d(self.embedding_dim, num_classes, kernel_size=1)

            # combine into a module
            self.decoders[key] = nn.ModuleDict(
                {
                    "fc": fc,
                    "deconv": deconv,
                    "output_layer": output_layer,
                    "output_shape": (H, W),
                }
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        num_keys = len(self.decoders)
        split_sizes = [self.h_dim] * num_keys
        x_split = torch.split(x, split_sizes, dim=1)  # tensors of shape (B, h_dim)

        for (key, modules), x_key in zip(self.decoders.items(), x_split):
            fc = modules["fc"]
            deconv = modules["deconv"]
            output_layer = modules["output_layer"]
            H, W = modules["output_shape"]
            upsample_info = self._compute_upsample_layers(H, W)

            # expand latent vector
            x_fc = fc(x_key)  # (B, 128 * start_size * start_size)
            x_fc = x_fc.view(
                x_key.size(0),
                128,
                upsample_info["start_size"],
                upsample_info["start_size"],
            )
            # pass through deconvolutional layers
            x_deconv = deconv(x_fc)
            # output layer
            logits = output_layer(x_deconv)
            # adjust dimensions to match output_shape
            logits = logits[:, :, :H, :W]
            outputs[key] = logits  # (B, num_classes, H, W)

        return outputs

    def _compute_upsample_layers(self, H: int, W: int) -> Dict[str, int]:
        """Computes the number of upsampling layers needed based on H and W."""
        num_layers = max(
            int(np.ceil(np.log2(max(H, W))) - 2), 1
        )  # subtracting 2 for the initial size
        start_size = H // (2**num_layers)
        return {"num_layers": num_layers, "start_size": start_size}


class InventoryEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        inv_shapes: Dict[str, Tuple[Tuple[int, ...], int]],
        embedding_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.inv_shapes = inv_shapes  # dict of {key: (shape, num_classes)}
        self.embedding_dim = embedding_dim

        # embedding layers and encoders for each key
        self.encoders = nn.ModuleDict()
        for key, (num_classes, shape) in self.inv_shapes.items():
            # embedding layer for the key
            embedding = nn.Embedding(
                num_embeddings=num_classes, embedding_dim=self.embedding_dim
            )
            # linear layer to produce feature vector
            fc = nn.Sequential(
                nn.Linear(np.prod(shape) * self.embedding_dim, h_dim),
                nn.ReLU(),
            )

            self.encoders[key] = nn.ModuleDict({"embedding": embedding, "fc": fc})

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        for key, modules in self.encoders.items():
            x = inputs[key].long()
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
    def __init__(
        self,
        h_dim: int,
        inv_shapes: Dict[str, Tuple[Tuple[int, ...], int]],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.inv_shapes = inv_shapes  # dict of {key: (shape, num_classes)}

        # decoders for each key
        self.decoders = nn.ModuleDict()
        for key, (num_classes, shape) in self.inv_shapes.items():
            # linear layer to map h_dim to output logits
            output_dim = int(np.prod(shape)) * num_classes
            decoder = nn.Sequential(
                nn.Linear(h_dim, output_dim),
                # output reshaping and activation will be handled in forward
            )
            self.decoders[key] = decoder

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        # compute h_dim_total and split x accordingly
        h_dim = self.h_dim
        num_keys = len(self.decoders)
        split_sizes = [h_dim] * num_keys
        x_split = torch.split(
            x, split_sizes, dim=1
        )  # List of tensors of shape (B, h_dim)
        for (key, decoder), x_key in zip(self.decoders.items(), x_split):
            logits = decoder(x_key)  # (B, output_dim)
            shape, num_classes = self.inv_shapes[key]
            logits = logits.view(
                -1, *shape, num_classes
            )  # Reshape to (B, ..., num_classes)
            outputs[key] = logits
        return outputs


class MessageEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        message_shape: Tuple[int, Tuple[int]],
        device: torch.device = torch.device("cpu"),
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.vocab_size, message_length = (
            message_shape  # vocab_size, (message_length, )
        )
        self.message_length = message_length[0]

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embedding_dim
        )
        self.fc = nn.Sequential(
            nn.Linear(self.message_length * embedding_dim, h_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()  # (B, message_length)
        x_embedded = self.embedding(x)  # (B, message_length, E)
        x_embedded = x_embedded.view(
            x.size(0), -1
        )  # flatten to (B, message_length * E)
        x = self.fc(x_embedded)  # (B, h_dim)
        return x  # (B, h_dim)


class MessageDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        message_shape: Tuple[int, Tuple[int]],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.vocab_size, message_length = (
            message_shape  # vocab_size, (message_length, )
        )
        self.message_length = message_length[0]

        self.decoder = nn.Linear(h_dim, self.message_length * self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(x)
        logits = logits.view(
            -1, self.message_length, self.vocab_size
        )  # (B, message_length, vocab_size)
        return logits


class BlstatsEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        blstats_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim

        self.fc = nn.Sequential(
            nn.Linear(blstats_size, h_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.fc(x)  # (B, h_dim)
        return x  # (B, h_dim)


class BlstatsDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        blstats_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.blstats_size = blstats_size

        self.decoder = nn.Linear(h_dim, blstats_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blstats_recon = self.decoder(x)  # (B, blstats_size)
        return blstats_recon


class ScreenDescriptionsEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        input_shape: Tuple[int, Tuple[int, int, int]],
        embedding_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.vocab_size, self.input_shape = input_shape  # vocab_size, (H, W, D)

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim
        )

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x = x.long()  # (B, H, W, D)
        x = self.embedding(x)  # (B, H, W, D, E)
        x = x.permute(0, 4, 1, 2, 3)  # (B, E, H, W, D)
        x = self.conv(x)  # (B, h_dim, 1, 1, 1)
        x = x.view(B, -1)  # Flatten to (B, h_dim)
        return x


class ScreenDescriptionsDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        output_shape: Tuple[int, Tuple[int, int, int]],
        embedding_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.vocab_size, self.output_shape = output_shape  # vocab_size, (H, W, D)
        self.embedding_dim = embedding_dim

        # linear layer to expand latent vector
        self.fc = nn.Sequential(
            nn.Linear(h_dim, 128 * 2 * 2 * 2),
            nn.ReLU(),
        )

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
        )

        # output layer to produce logits over vocab_size
        self.output_layer = nn.Conv3d(
            self.embedding_dim, self.vocab_size, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x = self.fc(x)  # (B, 128 * 2 * 2 * 2)
        x = x.view(B, 128, 2, 2, 2)  # reshape to start deconvolution
        x = self.deconv(x)

        logits = self.output_layer(x)  # (B, vocab_size, H, W, D)
        logits = logits[
            :, :, : self.output_shape[0], : self.output_shape[1], : self.output_shape[2]
        ]
        return logits


class TtyCursorEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, h_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()  # (B, 2)
        x = self.encoder(x)  # (B, h_dim)
        return x


class TtyCursorDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # output between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x) * 255  # (B, 2), scaled to [0, 255]
        return x
