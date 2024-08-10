import torch

from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(
            self,
            mapping: torch.LongTensor,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean squared error between the prediction and the target.

        Args:
            mapping: (flatten_batch) - association between the prediction and the target.
            prediction: (flatten_batch, output_dim) - prediction of the context
            target: (num_abstracts, output_dim) - target context

        Returns:
            loss: () - mean squared error
        """

        mapped_target = target[mapping]

        loss = self.loss(prediction, mapped_target)
        return loss


def test_mse_loss():
    loss = MSELoss()
    prediction = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [1.7, 1.9, 2.1],
        [1.3, 1.4, 1.5],
    ])
    target = torch.tensor([
        [1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1],
        [1.7, 1.9, 2.1],
        [0.0, 1.0, 0.0],
    ])
    mapping = torch.tensor([0, 1, 2, 3])

    loss1 = loss(mapping, prediction, target)

    prediction = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 1.5, 0.6],
        [1.7, 1.9, 2.1],
        [1.3, 1.4, 1.5],
    ])
    target = torch.tensor([
        [1.6, 1.7, 1.8],
        [1.9, 2.0, 2.1],
        [1.7, 1.9, 2.1],
        [0.0, 1.0, 0.0],
    ])

    loss2 = loss(mapping, prediction, target)

    assert loss1 > loss2


if __name__ == "__main__":
    test_mse_loss()
    print("MseLoss test passed")
