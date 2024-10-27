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
        self.input_shapes = input_shapes  # dict of {key: (num_classes, shape)}
        self.embedding_dim = embedding_dim

        # embedding layers and convolutional encoders for each key
        self.encoders = nn.ModuleDict()
        for key, (num_classes, _) in input_shapes.items():
            # embedding layer for the key
            embedding = nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=self.embedding_dim,
                device=self.device,
            )

            # convolutional encoder for the key
            conv = nn.Sequential(
                nn.Conv2d(self.embedding_dim, 64, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(128, self.h_dim, kernel_size=3, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            ).to(self.device)

            # combine embedding and conv into a module
            self.encoders[key] = nn.ModuleDict({"embedding": embedding, "conv": conv})

    def forward(self, inputs: Dict[str, np.ndarray]) -> torch.Tensor:
        features = []
        for key, modules in self.encoders.items():
            embedding = modules["embedding"]
            conv = modules["conv"]

            x = torch.as_tensor(inputs[key], device=self.device).long()  # (B, H, W)
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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.output_shapes = output_shapes  # dict of {key: (num_classes, (H, W))}

        self.decoders = nn.ModuleDict()
        for key, (num_classes, shape) in self.output_shapes.items():
            H, W = shape
            decoder = nn.Sequential(
                nn.Linear(h_dim, h_dim * H * W),
                nn.SiLU(),
                nn.Unflatten(1, (h_dim, H, W)),  # (B, h_dim, H, W)
                nn.Conv2d(h_dim, h_dim // 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(h_dim // 2, num_classes, kernel_size=3, padding=1),
            ).to(device)
            self.decoders[key] = decoder

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}
        num_keys = len(self.decoders)
        split_sizes = [self.h_dim] * num_keys
        x_split = torch.split(x, split_sizes, dim=1)
        for (key, decoder), x_key in zip(self.decoders.items(), x_split):
            logits = decoder(x_key)  # (B, num_classes, H, W)
            outputs[key] = logits
        return outputs


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
        self.inv_shapes = inv_shapes  # dict of {key: (num_classes, shape)}
        self.embedding_dim = embedding_dim

        # embedding layers and encoders for each key
        self.encoders = nn.ModuleDict()
        for key, (num_classes, shape) in self.inv_shapes.items():
            # embedding layer for the key
            embedding = nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=self.embedding_dim,
                device=self.device,
            )
            # linear layer to produce feature vector
            fc = nn.Sequential(
                nn.Linear(np.prod(shape) * self.embedding_dim, h_dim),
                nn.SiLU(),
            ).to(device)

            self.encoders[key] = nn.ModuleDict({"embedding": embedding, "fc": fc})

    def forward(self, inputs: Dict[str, np.ndarray]) -> torch.Tensor:
        features = []
        for key, modules in self.encoders.items():
            x = torch.as_tensor(inputs[key], device=self.device).long()
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
            decoder = nn.Linear(
                h_dim, output_dim, device=self.device
            )  # output reshaping and activation will be handled in forward
            self.decoders[key] = decoder

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}

        # compute h_dim_total and split x accordingly
        num_keys = len(self.decoders)
        split_sizes = [self.h_dim] * num_keys
        x_split = torch.split(x, split_sizes, dim=1)  # tensors of shape (B, h_dim)
        for (key, decoder), x_key in zip(self.decoders.items(), x_split):
            logits = decoder(x_key)  # (B, output_dim)
            num_classes, shape = self.inv_shapes[key]
            logits = logits.view(-1, num_classes, *shape)  # (B, num_clases, ...)
            outputs[key] = logits
        return outputs


class EgocentricEncoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        input_shape: Tuple[int, Tuple[int, int]],
        embedding_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_classes, self.input_shape = (
            input_shape  # (num_classes, (H_crop, W_crop))
        )

        self.embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=embedding_dim,
            device=self.device,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, h_dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
        ).to(self.device)

    def forward(self, inputs):
        x = torch.as_tensor(inputs, device=self.device).long()  # (B, H_crop, W_crop)
        x_embedded = self.embedding(x)  # (B, H_crop, W_crop, E)
        x_embedded = x_embedded.permute(0, 3, 1, 2)  # (B, E, H_crop, W_crop)
        x_feature = self.conv(x_embedded)  # (B, h_dim, 1, 1)
        x_feature = x_feature.view(x_feature.size(0), -1)  # (B, h_dim)
        return x_feature


class EgocentricDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        output_shape: Tuple[int, Tuple[int, int]],
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.num_classes, (H, W) = output_shape

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim * H * W),
            nn.SiLU(),
            nn.Unflatten(1, (h_dim, H, W)),
            nn.Conv2d(h_dim, h_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(h_dim // 2, self.num_classes, kernel_size=3, padding=1),
        ).to(self.device)

    def forward(self, x):
        logits = self.decoder(x)  # (B, num_classes, H, W)
        return logits


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
            num_embeddings=self.vocab_size,
            embedding_dim=embedding_dim,
            device=self.device,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.message_length * embedding_dim, h_dim),
            nn.SiLU(),
        ).to(self.device)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).long()  # (B, message_length)
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

        self.decoder = nn.Linear(
            h_dim, self.message_length * self.vocab_size, device=self.device
        )

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
            nn.SiLU(),
        ).to(self.device)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()
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

        self.decoder = nn.Linear(h_dim, blstats_size, device=self.device)

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
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            device=self.device,
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
            nn.SiLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(128, h_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # global average pooling
        ).to(self.device)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        B = torch.as_tensor(x, device=self.device).size(0)

        x = torch.as_tensor(x, device=self.device).long()  # (B, H, W, D)
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
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.h_dim = h_dim
        self.vocab_size, (H, W, D) = output_shape

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim * H * W * D),
            nn.SiLU(),
            nn.Unflatten(1, (h_dim, H, W, D)),
            nn.Conv3d(h_dim, h_dim // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(h_dim // 2, self.vocab_size, kernel_size=3, padding=1),
        ).to(self.device)

    def forward(self, x):
        logits = self.decoder(x)  # (B, vocab_size, H, W, D)
        return logits


class TTYCursorEncoder(nn.Module):
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
            nn.SiLU(),
            nn.Linear(64, h_dim),
            nn.SiLU(),
        ).to(self.device)

    def forward(self, x: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()  # (B, 2)
        x = self.encoder(x)  # (B, h_dim)
        return x


class TTYCursorDecoder(nn.Module):
    def __init__(
        self,
        h_dim: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.device = device

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
            nn.Sigmoid(),  # output between 0 and 1
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x) * 255  # (B, 2), scaled to [0, 255]
        return x
        return x  # (B, h_dim)
