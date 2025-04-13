# grl.py
import torch

class GRL(torch.autograd.Function):
    """
    Gradient Reverse Layer (GRL) for adversarial domain adaptation.
    Forward pass: Identity function.
    Backward pass: Gradient sign is reversed and scaled by lambda.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient direction and scale by lambda
        return grad_output.neg() * ctx.lambda_, None


class GradientReverseLayer(torch.nn.Module):
    """
    Wrapper class to easily integrate GRL into a model.
    Usage:
        grl = GradientReverseLayer(lambda_=0.1)
        x = grl(x)
    """
    
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GRL.apply(x, self.lambda_)

    def extra_repr(self):
        return f"lambda={self.lambda_}"