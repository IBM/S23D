import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .moduleparams import GraphEncoderParams, ScoreNetParams
from .structural_layers import MLP, RMSNorm


def graph_encoder_factory(pars, score_network_pars):
    """
    Creates an instance of a graph encoder based on the provided
        parameters.

    Args:
    ----
        pars (dict): Parameters for the graph encoder. Key "kind"
            defines the type of graph encoder model to be used.
            Supported "kind": "GPS".
        score_network_pars (ScoreNetParams): Dataclass of parameters
            for the score network.

    Returns:
    -------
        Graph2DEncoder: An instance of the Graph2DEncoder class.

    Raises:
    ------
        ValueError: If the "kind" parameter is not "GPS".
    """
    kind = pars.get("kind")
    if kind != "GPS":
        raise ValueError(f"Unknown graph encoder kind {kind!r}, expected 'GPS'")
    pars = pars.copy()
    del pars["kind"]
    return Graph2DEncoder(score_network_pars, **pars)


class Encoder2DTransformerBlock(nn.Module):
    """
    A Transformer block for processing the input given the molecule
    graph. Wraps RMSNorm around a Graph2DAttentionBlock, with MLP.

    Args:
    ----
        score_net_params (ScoreNetParams): Dataclass with parameters
            for the score network in Graph2DAttentionBlock.
        graph_encoder_params (GraphEncoderParams): Dataclass with
            parameters for the graph encoder in Graph2DAttentionBlock.

    Returns:
    -------
        torch.Tensor: The encoded molecule features after passing
            through the 2D Transformer block.
    """

    def __init__(
        self, score_net_params: ScoreNetParams, graph_encoder_params: GraphEncoderParams
    ) -> None:
        super().__init__()

        n_embd = score_net_params.n_embd
        self.rms_1 = RMSNorm(n_embd)
        self.attn = Graph2DAttentionBlock(score_net_params, graph_encoder_params)
        self.rms_2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(
        self,
        x: torch.Tensor,
        features: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pass input to the Encoder block to get outputs.
        Input -> RMSNorm -> Graph2DAttentionBlock -> RMSNorm -> MLP -> Output.

        Args:
        ----
            x (torch.Tensor): Input tensor of shape
                (batch_size, sequence_length, hidden_size).
            features (dict[str, torch.Tensor]): Graph features
                processed by the data collator.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
        -------
            torch.Tensor: Output tensor.
        """
        x = x + self.attn(self.rms_1(x), features, attention_mask)
        x = x + self.mlp(self.rms_2(x))
        return x


class Graph2DEncoder(nn.Module):
    """
    Graph2DEncoder class for encoding molecule graphs.
    Initial inputs is the dictionary of atom features processed by the
    data collator.


    Args:
    ----
        score_net_params (ScoreNetParams): A dataclass with parameters
            for the score network in the Encoder2DTransformerBlock.
        use_degree (bool, optional): Whether node degree is being used
            as a feature. Defaults to False.
        random_bits (int, optional): Number of random bits to append to
            the input embeddings. Defaults to 0.
        use_charges (bool, optional): Whether atom charge is being used
            as a feature. Defaults to True.
        use_bonds (bool, optional): Whether bond information is being
            used as features. Defaults to False.
        use_gcn (bool, optional): Deprecated, maintained for
            compatibility. Defaults to False.

    Returns:
    -------
        torch.Tensor: The encoded graph features.
    """

    def __init__(
        self,
        score_net_params: ScoreNetParams,
        use_degree: bool = False,
        random_bits: int = 0,
        use_charges: bool = True,
        use_bonds: bool = False,
        use_gcn: bool = False,  # for compatibility
    ) -> None:
        super().__init__()

        self.n_layers = score_net_params.layers[0]
        graph_encoder_pars = GraphEncoderParams(use_degree, use_charges, use_bonds)
        self.embedding = nn.Embedding(
            num_embeddings=score_net_params.vocab_size,
            embedding_dim=score_net_params.n_embd - random_bits,
        )

        self.blocks = nn.ModuleList(
            [
                Encoder2DTransformerBlock(
                    score_net_params,
                    graph_encoder_pars,
                )
                for _ in range(self.n_layers)
            ]
        )

        if self.n_layers:
            ratio = 2 ** (-8 / score_net_params.n_head)
            m = [ratio]
            for i in range(score_net_params.n_head - 1):
                m.append(m[-1] * ratio)
            self.register_buffer("alibi", torch.tensor(m)[None, :, None, None])

        self.random_bits = random_bits

    def forward(
        self, features: dict[str, torch.Tensor], attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Graph2DEncoder.
        Input -> Embedding -> [Encoder2DTransformerBlock]*n_layers -> Output.

        Args:
        ----
            features (dict[str, torch.Tensor]): Graph features
                processed by the data collator.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
        -------
            torch.Tensor: Output tensor.
        """
        if self.n_layers:
            attention_mask = attention_mask[:, None, :, :] * self.alibi

        x = self.embedding(features["atoms"])
        if self.random_bits:
            x = torch.cat(
                [
                    x,
                    torch.randint(
                        -1,
                        2,
                        x.shape[:-1] + (self.random_bits,),
                        device=x.device,
                        dtype=x.dtype,
                    ),
                ],
                dim=-1,
            )

        for b in self.blocks:
            x = checkpoint(b, x, features, attention_mask, use_reentrant=False)

        return x


class Graph2DAttentionBlock(nn.Module):
    """
    Graph2DAttentionBlock class implementing a graph transformer
    attention block.

    Args:
    ----
        score_net_params (ScoreNetParams): Dataclass with parameters
            for the score network in Graph2DAttentionBlock.
        graph_encoder_params (GraphEncoderParams): Dataclass with
            parameters for the graph encoder in Graph2DAttentionBlock.

    Attributes:
    ----------
        n_embd (int): The dimensionality of the input and output
            embeddings.
        n_head (int): The number of attention heads.
        proj_bonds (nn.Linear): Layer for projecting bond features.
        proj_charges (nn.Linear): Layer for projecting charge features.
        c_attn (nn.Linear): Layer for computing attention scores.
        proj (nn.Linear): Layer for projecting the attention output.
        theta1 (nn.Parameter): Degree feature scaling.
        theta2 (nn.Parameter): Degree feature scaling.
        graph_encoder_params (GraphEncoderParams): Dataclass with
            parameters for the graph encoder in Graph2DAttentionBlock.

    Returns:
    -------
        torch.Tensor: The output tensor after attention operation.
    """

    def __init__(
        self, score_net_params: ScoreNetParams, graph_encoder_params: GraphEncoderParams
    ):
        super().__init__()
        n_embd = score_net_params.n_embd
        n_head = score_net_params.n_head

        assert n_embd % n_head == 0

        if graph_encoder_params.use_bonds:
            self.proj_bonds = nn.Linear(3, n_embd, bias=False)
        if graph_encoder_params.use_charges:
            self.proj_charges = nn.Linear(2, n_embd, bias=False)

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        if graph_encoder_params.use_degree:
            self.register_parameter(
                "theta1", param=nn.Parameter(torch.ones(1, 1, n_embd))
            )
            self.register_parameter("theta2", param=nn.Parameter(torch.zeros(n_embd)))

        self.graph_encoder_params = graph_encoder_params
        self.n_embd = n_embd
        self.n_head = n_head

    def forward(
        self,
        x,
        features: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        (B, T, C) = x.size()

        if self.graph_encoder_params.use_degree:
            x = x * (self.theta1 + torch.log(1 + features["degree"]) * self.theta2)

        if self.graph_encoder_params.use_charges:
            x = x + self.proj_charges(features["charges"])

        if self.graph_encoder_params.use_bonds:
            x = x + self.proj_bonds(features["bonds"])

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        return y
