import torch


class FlowScheduler:
    def training_pair(self, x1):
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device, dtype=x1.dtype)
        xt = (1-t[:, None, None])*x0 + t[:, None, None]*x1
        return xt, t, x1-x0

    @torch.no_grad()
    def integrate(self, velocity, shape, device, dtype, steps, initial_noise=None):
        if steps < 1:
            raise ValueError('sampling steps must be positive')
        x = torch.randn(shape, device=device, dtype=dtype) if initial_noise is None else initial_noise.clone()
        if tuple(x.shape) != tuple(shape) or x.device != device:
            raise ValueError('initial noise shape/device mismatch')
        for i in range(steps):
            t = torch.full((shape[0],), i/steps, device=device, dtype=dtype)
            x = x + velocity(x, t)/steps
        return x
