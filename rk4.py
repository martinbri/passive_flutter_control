import torch

def rk4(f, t0, y0, h, steps):
    t = torch.tensor(t0, dtype=y0.dtype, device=y0.device, requires_grad=t0.requires_grad if isinstance(t0, torch.Tensor) else False)
    y = y0
    
    traj = [(t, y)]

    
    for i in range(steps):
        k1 = h * f(t, y)
        k2 = h * f(t + h / 2, y + k1 / 2)
        k3 = h * f(t + h / 2, y + k2 / 2)
        k4 = h * f(t + h, y + k3)

        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h
        traj.append((t, y))

    return traj
