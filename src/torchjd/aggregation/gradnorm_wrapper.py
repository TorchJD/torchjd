import torch
import torch.nn as nn
import torch.nn.functional as F


class GradNormWrapper(nn.Module):
    r"""Loss-Balancing Wrapper using GradNorm.

    This module implements the adaptive loss-balancing mechanism proposed in
    "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    (Chen. et al, ICML 2018).
    Paper : https://arxiv.org/pdf/1711.02257

    It is designed to work on a list of task losses (scalar tensors) and re-weight
    them before aggregation. The learned weights are updated using a balancing loss heuristic computed by
    comparing the per-task gradient norms to target values (which depend on the relative loss ratios).

    The overall forward pass computes:

       weighted_loss = \sum_{i=1}^{T} \alpha_i * L_i,

    where the weights are computed as:

       \alpha = T * softmax(loss_scale)

    and T is the number of tasks.

    During training, after the backward pass on the aggregated loss,
    balance(losses, shared_params) is called to update the internal loss_scale parameters.

    Note: Unlike torchjd aggregators (e.g. MGDA, GradDrop, Sum) that expect a matrix input,
    GradNormWrapper is designed only to work with a list of losses. Indeed, GradNorm is not
    a gradient aggregator per se, but a loss balancing heuristic.

    Args:
        num_tasks (int): Number of tasks (loss scalars).
        alpha (float, optional): The hyperparameter controlling the strength of the balancing force.
                                 (Equivalent to “alpha” in LibMTL’s implementation.)
        lr (float, optional): Learning rate for updating the loss_scale parameters.
        device (torch.device, optional): Device on which to run the module.
    """

    def _init_(self, num_tasks: int, alpha: float = 1.5, lr: float = 1e-3, device=None):
        super()._init_()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device if device is not None else torch.device("cpu")
        # Initialize loss_scale on the specified device.
        self.loss_scale = nn.Parameter(torch.ones(num_tasks, device=self.device))
        self.optimizer = torch.optim.Adam([self.loss_scale], lr=lr)
        self.initial_losses = None  # Recorded on the first forward pass

            @property
    def loss_weights(self) -> torch.Tensor:
        """Computes normalized loss weights so that sum(alpha)=num_tasks."""
        return self.num_tasks * torch.softmax(self.loss_scale, dim=0)

    def forward(self, losses: list) -> torch.Tensor:
        """
        Forward pass computes the aggregated (i.e. weighted) loss.

        Args:
            losses (list of Tensor): List of scalar losses (each a 0-dim tensor) expected to be on self.device.

        Returns:
            Tensor: The aggregated loss (a 0-dim tensor).
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}.")
        losses = [l.to(self.device) for l in losses]
        if self.initial_losses is None:
            self.initial_losses = torch.stack([l.detach() for l in losses])
        weights = self.loss_weights
        weighted_losses = [weights[i] * losses[i] for i in range(self.num_tasks)]
        return sum(weighted_losses)


    def balance(self, losses: list, shared_params: list) -> torch.Tensor:
        """
        Updating the internal "loss scale" parameters by balancing the gradient norms.

        For each task i:
          - The gradient norm G_i of the weighted loss (alpha_i * L_i) is computed
          with respect to the shared parameters.
          - G is the average of these gradient norms.
          - Then, we compute the ratio r_i = (current_loss / initial_loss) for each task.
          - Normalization step : r_i_bar = r_i / mean(r_i).
          - Target gradient norm for task i is set as: target_i = G * (r_i_bar ** alpha).
          - Balancing loss is computed as the sum over tasks of |G_i - target_i|.
          - Backpropagation
          - "Loss scale" are updated  using its optimizer.

        Args:
            losses (list of Tensor): List of current task losses (each a 0-dim tensor) on self.device.
            shared_params (list): List of shared network parameters (if any).

        Returns:
            Tensor: The balancing loss (a 0-dim tensor).
        """
        losses = [l.to(self.device) for l in losses]
        weights = self.loss_weights
        grad_norms = []
        for i in range(self.num_tasks):
            weighted_loss = weights[i] * losses[i]
            grads = torch.autograd.grad(
                weighted_loss,
                shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            squared_norm = sum(
                [
                    g.pow(2).sum() if g is not None else torch.tensor(0.0, device=self.device)
                    for g in grads
                ]
            )
            grad_norm = torch.sqrt(squared_norm)
            grad_norms.append(grad_norm)
        grad_norms_tensor = torch.stack(grad_norms)
        G = grad_norms_tensor.mean().detach()
        current_losses = torch.stack([l.detach() for l in losses])
        L_i = current_losses / self.initial_losses
        r_i = L_i / L_i.mean()
        target = G * (r_i**self.alpha)
        balance_loss = torch.sum(torch.abs(grad_norms_tensor - target))
        if not balance_loss.requires_grad:
            balance_loss = balance_loss + 1e-6 * torch.sum(
                self.loss_scale
            )  # small term to ensure gradient flow
        self.optimizer.zero_grad()
        balance_loss.backward()
        self.optimizer.step()
        return balance_loss.detach()

    def reset(self) -> None:
        """Resets the recorded initial losses."""
        self.initial_losses = None

    def repr(self) -> str:
        return f"GradNormWrapper(num_tasks={self.num_tasks}, alpha={self.alpha})"

    def str(self) -> str:
        return "GradNormWrapper"
