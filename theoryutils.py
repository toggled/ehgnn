import torch 

def estimate_layer_spectral_norm(weight):
    """Compute spectral norm of a weight matrix."""
    W = weight.data
    # Try exact SVD, fallback to power iteration
    try:
        s = torch.linalg.svdvals(W)
        return float(s.max())  
    except:
        # Power iteration (2 steps)
        u = torch.randn(W.shape[0], device=W.device)
        u = u / (u.norm() + 1e-8)
        for _ in range(2):
            v = W.t().mv(u)
            v = v / (v.norm() + 1e-8)
            u = W.mv(v)
            u = u / (u.norm() + 1e-8)
        return abs(u.dot(W.mv(v)).item())

def estimate_model_lipschitz(model):
    norms = []
    for name, p in model.named_parameters():
        if "weight" in name and p.dim() == 2:
            sn = estimate_layer_spectral_norm(p)
            norms.append((name, sn))
    # total Lipschitz upper bound
    L = 1.0
    for _, sn in norms:
        L *= sn
    return norms, L

# class GradSignTracker:
#     def __init__(self):
#         self.prev_grad = None
#         self.history = []

#     def track(self, tensorT, epoch):
#         print(tensorT.requires_grad)
#         assert tensorT.requires_grad is True
#         g = tensorT.grad.detach().clone()
#         if self.prev_grad is None:
#             self.prev_grad = g
#             self.history.append((epoch, 0.0))
#             return 0.0

#         flips = ((self.prev_grad * g) < 0).float()
#         frac = flips.mean().item()
#         self.history.append((epoch, frac))
#         self.prev_grad = g
#         return frac

class GradSignTracker:
    def __init__(self):
        self.prev_grad = None
        self.history = []

    def track(self, tensor, epoch):
        g = tensor.grad
        if g is None:
            # No gradient this step
            self.history.append((epoch, None))
            print(f"[epoch {epoch}] WARNING: mask logits received no gradient.")
            return None

        g = g.detach().clone()

        if self.prev_grad is None:
            self.prev_grad = g
            self.history.append((epoch, 0.0))
            return 0.0

        flips = ((self.prev_grad * g) < 0).float()
        frac = flips.mean().item()
        self.history.append((epoch, frac))
        self.prev_grad = g
        return frac


def estimate_output_variance(model, data, p, num_samples=50):
    """
    For a fixed p, repeatedly sample stochastic masks inside the model
    (as it already does with STE), and measure variance of logits.
    Assumes model(data, is_test=False, force_p=p) or similar hook.
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        for _ in range(num_samples):
            # You may need to modify your model forward to accept 'force_p'
            out = model(data, is_test=False, force_p=p)  # logits (before softmax)
            outputs.append(out.unsqueeze(0))

    outputs = torch.cat(outputs, dim=0)  # [num_samples, N, C]
    # Variance over samples
    var = outputs.var(dim=0).mean().item()  # mean over nodes/classes
    return var

# ps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
# for p in ps:
#     v = estimate_output_variance(model, data, p)
#     print(f"p={p:.2f}, variance={v:.6f}")
