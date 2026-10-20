import math

import torch
import torch.nn as nn
from torch.nn import functional as F

_dim = 3


class ScaleAdaptLayer(nn.Module):
    """
    Scale Adapt Layer.

    Args:
    ----
        n_sin_basis (int): Number of sinusoidal basis functions to use.
        sigmoid_centers (list[float] | None): List of sigmoid centers.
        sigmoid_orientations (list[float] | None): List of sigmoid
            orientations.
        basis_scale (float): Scale of the basis functions. Default: 1.0.
        shift (float): Shift applied to the input. Default: 0.0.
        xyz (bool): Whether to include xyz (shifted inputs with the
            shift applied again, and scaled by the basis_scale) in the
            output. Default: False.

    Returns:
    -------
        torch.Tensor: Output tensor of shape (B, T, -1), where B is the
            batch size, T is the number of time steps, and -1 is the
            number of features.
    """

    def __init__(
        self,
        n_sin_basis: int,
        sigmoid_centers: list[float] | None = None,
        sigmoid_orientations: list[float] | None = None,
        basis_scale: float = 1.0,
        shift: float = 0.0,
        xyz: bool = False,
    ):
        super().__init__()

        self.has_sigmoids = self.has_sin_basis = False
        if n_sin_basis > 0:
            basis = [math.pi * 2.0 * basis_scale]
            for _ in range(n_sin_basis - 1):
                basis.append(basis[-1] * 2.0)
            self.register_parameter(
                "basis", param=torch.nn.Parameter(torch.tensor(basis))
            )
            self.has_sin_basis = True

        if sigmoid_centers:
            self.register_buffer(
                "sigmoid_orientations", torch.tensor(sigmoid_orientations)
            )
            self.register_buffer("sigmoid_centers", torch.tensor(sigmoid_centers))
            self.has_sigmoids = True

        self.shift = shift
        self.xyz = xyz
        self.basis_scale = basis_scale

    def forward(self, x):
        B, T, _ = x.shape
        x = x - self.shift
        x_s = x[..., None]
        x = x_s * self.basis
        features = [torch.sin(x), torch.cos(x)] if self.has_sin_basis else []
        if self.has_sigmoids:
            features.append(
                torch.sigmoid(x_s * self.sigmoid_orientations - self.sigmoid_centers)
            )
        if self.xyz:
            xyz = (x_s - self.shift) * self.basis_scale
            features.append(xyz)
        x = torch.cat(features, dim=-1).reshape(B, T, -1)
        return x


class ScaleAdaptEncoder(nn.Module):
    """
    Base class for Scale Adapt Encoder Modules. Uses `ScaleAdaptLayer`
        as the core of the encoder.

    Args:
    ----
        n_sin_basis (int): Number of sinusoidal basis functions to use.
        sigmoid_centers (list[float] | None): List of sigmoid centers.
        sigmoid_orientations (list[float] | None): List of sigmoid
            orientations.
        basis_scale (float): Scale of the basis functions. Default: 1.0.
        shift (float): Shift applied to the input. Default: 0.0.
        xyz (bool): Whether to include xyz (shifted inputs with the
            shift applied again, and scaled by the basis_scale) in the
            output. Default: False.
        first_layer_only (bool): Whether to apply the encoder only to
            the first layer. Default: False.
        gates (bool): Whether to use gates in the encoder. Default:
            False.

    Methods:
    -------
        build_encoder (layer_index: int, n_embd: int): Builds the
            encoder, in this base class case using a linear layer
            after the ScaleAdaptLayer

    Returns:
    -------
        torch.Tensor: Output tensor after passing through the encoder
    """

    def __init__(
        self,
        n_sin_basis: int,
        sigmoid_centers: list[float] | None = None,
        sigmoid_orientations: list[float] | None = None,
        basis_scale: float = 1.0,
        shift: float = 0.0,
        xyz: bool = False,
        first_layer_only: bool = False,
        gates: bool = False,
    ):
        super().__init__()
        self.encoder_layer = ScaleAdaptLayer(
            n_sin_basis, sigmoid_centers, sigmoid_orientations, basis_scale, shift, xyz
        )
        n_sigmoid_centers = len(sigmoid_centers) if sigmoid_centers else 0
        self.n_basis = n_sin_basis * 2 + n_sigmoid_centers
        if xyz:
            self.n_basis += 1
        self.first_layer_only = first_layer_only
        self.gates = gates

    def forward(self, x):
        return self.encoder_layer(x)

    def build_encoder(self, layer_index: int, n_embd: int):
        if self.first_layer_only and layer_index > 0:
            return None
        if self.gates and layer_index > 0:
            return EncoderGates(self.n_basis, n_embd)

        return nn.Linear(self.n_basis * _dim, n_embd, bias=False)


class ScaleAdaptMLPEncoder(ScaleAdaptEncoder):
    """
    ScaleAdaptMLPEncoder class inherits ScaleAdaptEncoder. Uses
        `ScaleAdaptLayer` as the core of the encoder, with 2-layer MLP
        after.

    Args:
    ----
        n_sin_basis (int): Number of sinusoidal basis functions to use.
        sigmoid_centers (list[float] | None): List of sigmoid centers.
        sigmoid_orientations (list[float] | None): List of sigmoid
            orientations.
        basis_scale (float): Scale of the basis functions. Default: 1.0.
        shift (float): Shift applied to the input. Default: 0.0.
        xyz (bool): Whether to include xyz (shifted inputs with the
            shift applied again, and scaled by the basis_scale) in the
            output. Default: False.
        first_layer_only (bool): Whether to apply the encoder only to
            the first layer. Default: False.

    Methods:
    -------
        build_encoder (layer_index: int, n_embd: int): Builds the
            encoder, in this base class case using a two-layer MLP
            after the ScaleAdaptLayer

    Returns:
    -------
        torch.Tensor: Output tensor after passing through the encoder
    """

    def __init__(
        self,
        n_sin_basis: int,
        sigmoid_centers: list[float] | None = None,
        sigmoid_orientations: list[float] | None = None,
        basis_scale: float = 1.0,
        shift: float = 0.0,
        hidden_units=128,
        xyz: bool = False,
        first_layer_only: bool = False,
    ):
        super().__init__(
            n_sin_basis,
            sigmoid_centers,
            sigmoid_orientations,
            basis_scale,
            shift,
            xyz,
            first_layer_only,
        )
        self.hidden_units = hidden_units

    def build_encoder(self, layer_index: int, n_embd: int):
        if self.first_layer_only and layer_index > 0:
            return None
        model = nn.Sequential(
            nn.Linear(self.n_basis * _dim, self.hidden_units, bias=False),
            nn.ReLU(),
            nn.Linear(self.hidden_units, n_embd, bias=False),
        )
        return model


class EncoderGates(nn.Module):
    """
    EncoderGates class to pad the output of the encoder with zeros to
        the embedding dimension.

    Args:
    ----
        n_basis (int): Number of basis functions.
        n_embd (int): Embedding dimension.

    Returns:
    -------
        torch.Tensor: Encoded tensor with gate weights applied.
    """

    def __init__(self, n_basis, n_embd):
        self.register_parameter("weight", param=torch.nn.Parameter(torch.ones(n_basis)))
        self.pad = n_embd - n_basis

    def forward(self, x):
        x = self.weight * x
        B, T, _ = x.size()
        zeros = torch.zeros(B, T, self.pad, device=x.device, dtype=x.dtype)
        x = torch.concat([x, zeros], dim=-1)
        return x


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP).

    Args:
    ----
        input_size (int): Input size of the neural network.
        hidden_units (int): Number of hidden units in the neural network.
        output_size (int): Output size of the neural network.
        shift (float): Shift value for input data.
        basis_scale (float): Scale factor for input data.

    Returns:
    -------
        torch.Tensor: MLP Module output tensor.
    """

    def __init__(
        self,
        input_size: int,
        hidden_units: int,
        output_size: int,
        shift: float,
        basis_scale: float,
    ):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_units)
        self.l2 = nn.Linear(hidden_units, output_size)
        self.shift = shift
        self.basis_scale = basis_scale

    def forward(self, x):
        return self.l2(F.relu(self.l1(x - self.shift) * self.basis_scale))


class ProjectionLayer(nn.Module):
    """
    Simple linear projection module.

    Args:
    ----
        input_size (int): Input size of the neural network.
        output_size (int): Output size of the neural network.
        shift (float): Shift value for input data.
        basis_scale (float): Scale factor for input data.

    Returns:
    -------
        torch.Tensor: ProjectionLayer Module output tensor.
    """

    def __init__(self, input_size, output_size, shift, scale):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.projection = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.projection((x - self.shift) * self.scale)
        return x


class ProjectionEncoder(nn.Module):
    """
    Encoder based on the ProjectionLayer.

    Args:
    ----
        basis_scale (float): Scale factor for the basis functions.
            Defaults to 1.0.
        shift (float): Shift value for the basis functions. Defaults to
            0.0.
        first_layer_only (bool): Whether to apply the projection to
            only the first layer. Defaults to True.

    Methods:
    -------
        build_encoder (layer_index: int, n_embd: int): BSetup an encoder
            for a `layer`, using the ProjectionLayer. Uses
            `layer_index` to determine if we should apply the encoding
            to this layer.

    Returns:
    -------
        torch.nn.Module: The encoder module.
    """

    def __init__(
        self,
        basis_scale: float = 1.0,
        shift: float = 0.0,
        first_layer_only: bool = True,
    ):
        super().__init__()
        self.shift = shift
        self.first_layer_only = first_layer_only
        self.basis_scale = basis_scale

    def forward(self, x):
        return x

    def build_encoder(self, layer_index: int, n_embd: int):
        if self.first_layer_only and layer_index > 0:
            return None

        projection = ProjectionLayer(_dim, n_embd, self.shift, self.basis_scale)
        return projection


class MLPEncoder(nn.Module):
    """
    Encoder based on the MLP module.

    Args:
    ----
        basis_scale (float): Scale factor for the basis functions.
            Defaults to 1.0.
        shift (float): Shift value for the basis functions. Defaults to
            0.0.
        hidden_units (int): Number of hidden units in the neural
            network. Defaults to 128.
        output_size (int): Output size of the neural network.
        first_layer_only (bool): Whether to apply the projection to
            only the first layer. Defaults to True.

    Methods:
    -------
        build_encoder (layer_index: int, n_embd: int): Setup an encoder
            for a `layer`, using the MLP module. Uses `layer_index` to
            determine if we should apply the encoding to this layer.

    Returns:
    -------
        torch.nn.Module: The encoder module.
    """

    def __init__(
        self,
        basis_scale: float = 1.0,
        shift: float = 0.0,
        hidden_units: int = 128,
        output_size: int | None = None,
        first_layer_only: bool = True,
    ):
        super().__init__()
        self.hidden_units = hidden_units
        self.shift = shift
        self.output_size = output_size
        self.first_layer_only = first_layer_only
        self.basis_scale = basis_scale

    def forward(self, x):
        return x

    def build_encoder(self, layer_index: int, n_embd: int):
        if self.first_layer_only and layer_index > 0:
            return None
        output_size = self.output_size if self.output_size else n_embd
        mlp = MLP(_dim, self.hidden_units, output_size, self.shift, self.basis_scale)
        return mlp


def encoder_factory(encoder_kwargs):
    kind = encoder_kwargs["kind"]
    kwargs = encoder_kwargs.copy()
    del kwargs["kind"]

    if kind == "ScaleAdapt":
        return ScaleAdaptEncoder(**kwargs)
    if kind == "MLP":
        return MLPEncoder(**kwargs)
    if kind == "ScaleAdaptMLP":
        return ScaleAdaptMLPEncoder(**kwargs)
    if kind == "Projection":
        return ProjectionEncoder(**kwargs)

    raise ValueError(
        f"Wrong encoder kind {kind}. Options are ScaleAdapt, MLP, Projection."
    )


class TimeEncodingHead(nn.Module):
    """
    Encoding for diffusion time points using a set of basis functions.
    The basis functions are defined as `math.pi * 2.0` and
    `2.0 * basis[-1]`, where `basis` is a list of basis values.

    Args:
    ----
        n_basis (int): Number of basis functions to use for encoding.

    Returns:
    -------
        torch.Tensor: Encoded tensor of shape
            (batch_size, seq_len, n_basis * 2).
    """

    def __init__(self, n_basis: int):
        super().__init__()
        n_basis //= 2
        basis = [math.pi * 2.0]
        for _ in range(n_basis - 1):
            basis.append(basis[-1] * 2.0)
        self.register_buffer("basis", torch.tensor(basis))
        self.n_basis = n_basis

    def forward(self, x):
        x = x[..., None] * self.basis
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x
