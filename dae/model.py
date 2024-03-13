from typing import Iterable, List, Tuple, Union

import numpy as np
import torch

bce_logits = torch.nn.functional.binary_cross_entropy_with_logits
mse = torch.nn.functional.mse_loss


class TransformerEncoder(torch.nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, feedforward_dim: int
    ) -> None:
        """
        Initialize a TransformerEncoder object.

        Args:
            embed_dim (int): The dimension of the embedding.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate.
            feedforward_dim (int): The dimension of the feedforward layer.
        """
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TransformerEncoder.

        Args:
            x_in (torch.Tensor): The input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor `x` and the hidden tensor `hidden`.
        """
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        hidden = self.linear_1(x)
        ff_out = self.linear_2(torch.nn.functional.relu(hidden))
        x = self.layernorm_2(x + ff_out)
        return x, hidden


class TransformerAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_inputs: int,
        n_cats: int,
        n_nums: int,
        hidden_size: int = 1024,
        num_subspaces: int = 8,
        embed_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.0,
        feedforward_dim: int = 512,
        emphasis: float = 0.75,
        task_weights: Iterable[float] = [1, 1],
        mask_loss_weight: float = 2,
        encoder_fdim: int = 128,
    ) -> None:
        """
        Initialize the TransformerAutoEncoder model.

        Args:
            num_inputs (int): The number of inputs to the model.
            n_cats (int): The number of categorical features.
            n_nums (int): The number of numerical features.
            hidden_size (int, optional): The size of the hidden layer. Defaults to 1024.
            num_subspaces (int, optional): The number of subspaces. Defaults to 8.
            embed_dim (int, optional): The dimension of the embedding. Defaults to 128.
            num_heads (int, optional): The number of attention heads. Defaults to 8.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            feedforward_dim (int, optional): The dimension of the feedforward layer. Defaults to 512.
            emphasis (float, optional): The weight of the reconstruction loss. Defaults to 0.75.
            task_weights (List[float], optional): The weights of the categorical and numerical features. Defaults to [1, 1].
            mask_loss_weight (float, optional): The weight of the mask loss. Defaults to 2.
            encoder_fdim (int, optional): The dimension of the encoder feedforward layer. Defaults to 128.
        """
        super().__init__()
        assert hidden_size == embed_dim * num_subspaces
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.num_subspaces = num_subspaces
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.emphasis = emphasis
        self.task_weights = np.array(task_weights) / sum(task_weights)
        self.mask_loss_weight = mask_loss_weight
        self.encoder_fdim = encoder_fdim

        self.excite = torch.nn.Linear(in_features=num_inputs, out_features=hidden_size)

        self.encoder_1 = TransformerEncoder(
            embed_dim, num_heads, dropout, feedforward_dim
        )
        self.encoder_2 = TransformerEncoder(
            embed_dim, num_heads, dropout, feedforward_dim
        )
        self.encoder_3 = TransformerEncoder(
            embed_dim, num_heads, dropout, feedforward_dim
        )

        self.mask_predictor = torch.nn.Linear(
            in_features=hidden_size, out_features=num_inputs
        )
        self.reconstructor = torch.nn.Linear(
            in_features=hidden_size + num_inputs, out_features=num_inputs
        )

    def divide(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide the input tensor into subspaces.

        Args:
            x (torch.Tensor): The input tensor with shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: The divided tensor with shape (num_subspaces, batch_size, embed_dim).
        """
        batch_size = x.shape[0]
        x = x.reshape((batch_size, self.num_subspaces, self.embed_dim)).permute(
            (1, 0, 2)
        )
        return x

    def combine(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the divided tensor into a single tensor.

        Args:
            x (torch.Tensor): The divided tensor with shape (num_subspaces, batch_size, embed_dim).

        Returns:
            torch.Tensor: The combined tensor with shape (batch_size, num_inputs).
        """
        batch_size = x.shape[1]
        x = x.permute((1, 0, 2)).reshape((batch_size, -1))
        return x

    def forward(self, x: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass of the TransformerAutoEncoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_inputs).

        Returns:
            Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                  Tuple[torch.Tensor, torch.Tensor],
                  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                  A tuple containing:
                  the intermediate tensors x1, x2, x3;
                  the final tensors reconstruction and predicted_mask;
                  the hidden tensors hidden1, hidden2, hidden3.
        """
        x = torch.nn.functional.relu(self.excite(x))

        x = self.divide(x)
        x1, hidden1 = self.encoder_1(x)
        x2, hidden2 = self.encoder_2(x1)
        x3, hidden3 = self.encoder_3(x2)

        x = self.combine(x3)

        predicted_mask = self.mask_predictor(x)
        reconstruction = self.reconstructor(torch.cat([x, predicted_mask], dim=1))

        return (
            (x1, x2, x3),
            (reconstruction, predicted_mask),
            (hidden1, hidden2, hidden3),
        )

    def split(self, t: torch.Tensor) -> List[torch.Tensor]:
        """
        Split the input tensor into two parts, based on the number of categorical and numerical features.

        Args:
            t (torch.Tensor): The input tensor to be split.

        Returns:
            List[torch.Tensor]: A list of two tensors, the first containing the categorical features,
                               the second containing the numerical features.
        """
        return torch.split(t, [self.n_cats, self.n_nums], dim=1)

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the feature representation of the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: The feature representation of the input tensor.
        """
        _, _, hidden = self.forward(x)

        return self.combine(hidden[-1])

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Calculate the loss for the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_features).
            y (torch.Tensor): The target tensor of shape (batch_size, num_features).
            mask (torch.Tensor): The mask of corrupted x tensor of shape (batch_size, num_features).
            reduction (str, optional): The reduction method for the loss. Defaults to 'mean'.

        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: The reconstruction loss and mask loss, or a list of both if reduction='none'.
        """
        _, (reconstruction, predicted_mask), _ = self.forward(x)
        x_cats, x_nums = self.split(reconstruction)
        y_cats, y_nums = self.split(y)
        w_cats, w_nums = self.split(
            mask * self.emphasis + (1 - mask) * (1 - self.emphasis)
        )

        cat_loss = self.task_weights[0] * torch.mul(
            w_cats, bce_logits(x_cats, y_cats, reduction="none")
        )
        num_loss = self.task_weights[1] * torch.mul(
            w_nums, mse(x_nums, y_nums, reduction="none")
        )

        reconstruction_loss = (
            torch.cat([cat_loss, num_loss], dim=1)
            if reduction == "none"
            else cat_loss.mean() + num_loss.mean()
        )
        mask_loss = self.mask_loss_weight * bce_logits(
            predicted_mask, mask, reduction=reduction
        )

        return (
            reconstruction_loss + mask_loss
            if reduction == "mean"
            else [reconstruction_loss, mask_loss]
        )


class SwapNoiseMasker(object):
    def __init__(self, probas: List[float]) -> None:
        """
        Initialize a SwapNoiseMasker object.

        Args:
            probas (List[float]): The probabilities of swapping a feature.

        Returns:
            None
        """
        self.probas = torch.from_numpy(np.array(probas, dtype=np.float32))

    def apply(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the swap noise mask to the input tensor.

        Args:
            X (torch.Tensor): The input tensor of shape (batch_size, num_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The corrupted input tensor and mask tensor.
        """
        should_swap = torch.bernoulli(self.probas.to(X.device) * torch.ones_like(X))
        corrupted_X = torch.where(should_swap == 1, X[torch.randperm(X.shape[0])], X)
        mask = (corrupted_X != X).float()

        return corrupted_X, mask
