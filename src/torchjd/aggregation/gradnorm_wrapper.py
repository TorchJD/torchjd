import torch
import torch.nn as nn
import torch.nn.functional as F

class GradNormWrapper(nn.Module):
    r"""
    Loss-Balancing Wrapper using GradNorm.

    This module implements the adaptive loss-balancing mechanism proposed in
    "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    (Chen et al., ICML 2018).  See the paper: https://arxiv.org/pdf/1711.02257

    It is designed to work on a list of task losses (scalar tensors) and re-weight
    them before aggregation. The aggregated loss is computed as:

    .. math::
       L_{\text{total}} = \sum_{i=1}^{T} \alpha_i \, L_i,
    
    where the task weights are computed as:

    .. math::
       \alpha = T \cdot \mathrm{softmax}(\text{loss\_scale})
    
    and T is the number of tasks.

    The internal parameters (``loss_scale``) are updated by the
    :meth:`balance` method, which computes per-task gradient norms and compares them
    to target values derived from the ratio of current losses to the initially recorded losses.
    Note that GradNormWrapper is a loss-balancing heuristic rather than a gradient aggregator
    (like MGDA or GradDrop).

    **Example**

    .. code-block:: python

       import torch
       from torchjd.aggregation.gradnorm_wrapper import GradNormWrapper

       # Use CUDA if available
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       # Initialize GradNormWrapper for 2 tasks with alpha=1.5 and learning rate 1e-2.
       gradnorm = GradNormWrapper(num_tasks=2, alpha=1.5, lr=1e-2, device=device)

       # Compute individual task losses.
       loss1 = (torch.randn(10, device=device, requires_grad=True).sum()) ** 2
       loss2 = (torch.randn(10, device=device, requires_grad=True).mean()) ** 2

       # Forward pass: compute aggregated (weighted) loss.
       total_loss = gradnorm([loss1, loss2])
       total_loss.backward(retain_graph=True)

       # Suppose shared parameters are provided (e.g., from a shared model)
       shared_params = [torch.randn(10, device=device, requires_grad=True)]
       # Update the loss_scale parameters based on the balancing loss.
       balance_loss = gradnorm.balance([loss1, loss2], shared_params)
       print("Balancing loss:", balance_loss.item())

    Args:
        num_tasks (int): Number of tasks (loss scalars).
        alpha (float, optional): Hyperparameter controlling the strength of the balancing force
                                 (equivalent to “alpha” in LibMTL’s implementation).
        lr (float, optional): Learning rate for updating the loss_scale parameters.
        device (torch.device, optional): Device on which to run the module.

    Returns:
        When calling :meth:`forward`, returns a scalar aggregated loss.
        When calling :meth:`balance`, returns a scalar balancing loss.


   # ----- Composing with an Underlying Aggregator -----
   # If you wish to further aggregate re-weighted losses using a gradient aggregator like MGDA,
   # you can re-weight the losses and then form a matrix to pass to the aggregator.
   weights = gradnorm.loss_weights
   reweighted_losses = [weights[i] * loss for i, loss in enumerate([loss1, loss2])]
   # Stack the re-weighted losses as a column vector.
   matrix = torch.stack(reweighted_losses).unsqueeze(1)
   # Initialize the MGDA aggregator.
   mgda = MGDA(epsilon=0.001, max_iters=100)
   # Use MGDA to aggregate the matrix.
   aggregated_update = mgda(matrix)
   print("Aggregated update:", aggregated_update)


    Note
    ----
    While GradNormWrapper itself is not designed as a gradient aggregator (i.e. it does not accept a matrix
    input), it can be composed with other torchjd aggregators. In the example above, we first use GradNormWrapper
    to compute adaptive weights from a list of losses, and then we use these weights to re-weight the losses.
    The reweighted losses are then stacked into a matrix, which can be fed into an aggregator like MGDA.
    """

    def __init__(self, num_tasks: int, alpha: float = 1.5, lr: float = 1e-3, device=None):
        super().__init__()
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
        Computes the aggregated (weighted) loss.

        Args:
            losses (list of Tensor): List of scalar losses (each a 0-dim tensor) that are on ``self.device``.

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
        Updates the internal "loss_scale" parameters by balancing the gradient norms.

        For each task *i*:
        
        - Computes the gradient norm *G_i* of the weighted loss (``alpha_i * L_i``)
          with respect to the shared parameters.
        - Computes *G*, the average gradient norm.
        - Computes the ratio *r_i* = (current_loss / initial_loss) for each task, and normalizes
          it: *r_i_bar* = *r_i* / mean(*r_i*).
        - Sets the target gradient norm as: **target_i** = *G* · (r_i_bar<sup>alpha</sup>).
        - The balancing loss is defined as the sum over tasks of |G_i - target_i|.
        - Backpropagation is then performed and the internal loss_scale is updated.

        Args:
            losses (list of Tensor): List of current task losses (each a 0-dim tensor) on ``self.device``.
            shared_params (list): List of shared network parameters.

        Returns:
            Tensor: The balancing loss (a 0-dim tensor).
        """
        losses = [l.to(self.device) for l in losses]
        weights = self.loss_weights
        grad_norms = []
        for i in range(self.num_tasks):
            weighted_loss = weights[i] * losses[i]
            grads = torch.autograd.grad(
                weighted_loss, shared_params,
                retain_graph=True, create_graph=True, allow_unused=True,
            )
            squared_norm = sum([
                g.pow(2).sum() if g is not None else torch.tensor(0.0, device=self.device)
                for g in grads
            ])
            grad_norm = torch.sqrt(squared_norm)
            grad_norms.append(grad_norm)
        grad_norms_tensor = torch.stack(grad_norms)
        G = grad_norms_tensor.mean().detach()
        current_losses = torch.stack([l.detach() for l in losses])
        L_i = current_losses / self.initial_losses
        r_i = L_i / L_i.mean()
        target = G * (r_i ** self.alpha)
        balance_loss = torch.sum(torch.abs(grad_norms_tensor - target))
        if not balance_loss.requires_grad:
            balance_loss = balance_loss + 1e-6 * torch.sum(self.loss_scale)
        self.optimizer.zero_grad()
        balance_loss.backward()
        self.optimizer.step()
        return balance_loss.detach()

    def reset(self) -> None:
        """Resets the recorded initial losses."""
        self.initial_losses = None

    def __repr__(self) -> str:
        return f"GradNormWrapper(num_tasks={self.num_tasks}, alpha={self.alpha})"

    def __str__(self) -> str:
        return "GradNormWrapper"
