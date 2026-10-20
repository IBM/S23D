import torch
import torch.nn as nn


def masked_mean(x, padding_mask):
    mean = torch.sum(x * padding_mask[..., None], dim=1) / torch.sum(
        padding_mask, dim=-1, keepdim=True
    )
    return mean[:, None, :]


class CoMVPSDE(nn.Module):
    """
    VPSDE class for Center of Mass subspace
    References [1] https://arxiv.org/pdf/2011.13456.pdf.
    """

    @classmethod
    def clone(cls, obj):
        """
        Clone to get a new copy of a CoMVPSDE instance.

        Args:
        ----
            obj (CoMVPSDE): Instance to clone.

        Returns:
        -------
            CoMVPSDE: Clone of the instance.
        """
        beta_s, beta_e = obj.beta_s, obj.beta_e
        center_score = obj.center_score
        reparametrization = obj.reparametrization
        scale = obj.scale
        return cls(beta_s, beta_e, scale, center_score, reparametrization)

    def __init__(
        self,
        beta_s: float,
        beta_e: float,
        scale: None | float = None,
        center_score: bool = True,
        reparametrization: bool = False,
    ):
        """
        Create a CoMVPSDE with a linear beta schedule.

        Args:
        ----
            beta_s (float): First value of beta in the linear schedule.
            beta_e (float): Slope of the linear schedule, which is
                beta(t) = beta_s + beta_e * t, so the last value of beta is
                beta_s + beta_e.
            scale (float, optional):
            center_score (bool): Defaults to True.
            reparametrization (bool): Defaults to False.

        """
        super().__init__()
        self.beta_s, self.beta_e = beta_s, beta_e
        self.center_score = center_score
        self.reparametrization = reparametrization
        self.scale = scale

    def f(self, beta):
        return -beta * 0.5

    def g(self, beta):
        return torch.sqrt(beta)

    def schedule(self, t):
        return self.linear_schedule(t)

    def linear_schedule(self, t):
        return self.beta_e * t + self.beta_s

    def linear_transition_integral(self, t):
        return (0.5 * self.beta_e * t + self.beta_s) * t

    @torch.no_grad()
    def transition_kernel(self, t):
        integral = self.linear_transition_integral(t)
        mean = torch.exp(-0.5 * integral)
        std = torch.sqrt(1.0 - mean**2)
        return mean, std

    def loss(self, x, padding_mask, score_network):
        batch_size = x.shape[0]
        t = torch.rand((batch_size,), dtype=x.dtype, device=x.device)
        noise = torch.randn_like(x)
        mean, std = self.transition_kernel(t.view(-1, 1, 1))

        noise -= masked_mean(noise, padding_mask)
        if self.scale is None:
            z = x * mean + noise * std
        else:
            z = (x * mean + noise * std) * (
                self.scale / torch.sqrt(mean**2 * (self.scale**2 - 1.0) + 1.0)
            )

        score = score_network(t, z)
        if self.center_score:
            score -= masked_mean(score, padding_mask)
        if not self.reparametrization:
            score = score * std
        return torch.mean(masked_mean((score - noise) ** 2, padding_mask))

    @torch.no_grad()
    def reverse(
        self,
        score_network,
        padding_mask,
        device=None,
        n_steps=200,
    ):
        x = torch.randn((*padding_mask.shape, 3), device=device)
        x -= masked_mean(x, padding_mask)

        delta_t = torch.tensor(1.0 / (n_steps - 1))
        sqrt_delta_t = torch.sqrt(delta_t)

        for i in reversed(range(n_steps)):
            noise = torch.randn_like(x)
            noise -= masked_mean(noise, padding_mask)
            t_flat = torch.full((x.shape[0],), (i + 1) / n_steps, device=device)
            t = t_flat.view(-1, 1, 1)
            beta = self.schedule(t)
            f, g = self.f(beta), self.g(beta)
            mean, std = self.transition_kernel(t)
            if self.scale is None:
                score = score_network(t_flat, x)
            else:
                scale_factor = self.scale / torch.sqrt(
                    mean**2 * (self.scale**2 - 1.0) + 1.0
                )
                score = score_network(t_flat, x * scale_factor)

            if self.center_score:
                score -= masked_mean(score, padding_mask)
            f = f * delta_t
            g = g * sqrt_delta_t

            if self.reparametrization:
                x += -f * x - g * g * score / std - g * noise
            else:
                x += -f * x - g * g * score - g * noise
            x -= masked_mean(x, padding_mask)

        return x


class DDIM(CoMVPSDE):
    """
    Implementing DDIM in reverse (diffusion) step.
    Reference: https://arxiv.org/abs/2010.02502.
    """

    @torch.no_grad()
    def reverse(
        self,
        score_network,
        padding_mask,
        device=None,
        n_steps=200,
    ):
        x = torch.randn((*padding_mask.shape, 3), device=device)
        x -= masked_mean(x, padding_mask)

        for i in reversed(range(n_steps)):
            t_flat = torch.full((x.shape[0],), (i + 1) / n_steps, device=device)
            t = t_flat.view(-1, 1, 1)
            t_flat1 = torch.full((x.shape[0],), (i) / n_steps, device=device)
            t_1 = t_flat1.view(-1, 1, 1)

            mean, std = self.transition_kernel(t)
            mean_1, std_1 = self.transition_kernel(t_1)
            if self.scale is None:
                score = score_network(t_flat, x)
            else:
                scale_factor = self.scale / torch.sqrt(
                    mean**2 * (self.scale**2 - 1.0) + 1.0
                )
                score = score_network(t_flat, x * scale_factor)
            if self.center_score:
                score -= masked_mean(score, padding_mask)
            x = mean_1 * (x - score * std * std) / mean + std_1 * score * std
            x -= masked_mean(x, padding_mask)
        return x


def diffusion_factory(diffusion_model, **kwargs):
    kind = kwargs.pop("kind", None)
    n_steps = kwargs["n_steps"]
    if kind == "DDIM":
        model = DDIM.clone(diffusion_model)
    elif kind == "VPSDE":
        model = CoMVPSDE.clone(diffusion_model)
    else:
        raise ValueError(f"Unknown diffusion kind {kind!r}, expected 'VPSDE' or 'DDIM'")
    return model, n_steps
