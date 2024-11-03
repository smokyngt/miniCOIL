import torch

from torch import nn


class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    @classmethod
    def cosine_distance(
            cls,
            mapping: torch.LongTensor,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        # (flatten_batch, output_dim)
        mapped_target = target[mapping]

        # Cosine similarity
        # (flatten_batch)
        prediction_norm = torch.norm(prediction, dim=1)

        # (flatten_batch)
        target_norm = torch.norm(mapped_target, dim=1)

        # If prediction_norm or target_norm is zero, exclude them from the calculation
        mask1 = prediction_norm < 1e-6
        mask2 = target_norm < 1e-6
        mask = mask1 + mask2

        prediction = prediction[~mask]
        mapped_target = mapped_target[~mask]
        prediction_norm = prediction_norm[~mask]
        target_norm = target_norm[~mask]

        # Pairwise cosine similarity
        # (flatten_batch)
        cosine_similarity = torch.einsum('bi,bi->b', prediction, mapped_target) / (prediction_norm * target_norm)

        # Cosine distance
        # (flatten_batch)
        cosine_distance = 1 - cosine_similarity

        return cosine_distance

    def forward(
            self,
            mapping: torch.LongTensor,
            prediction: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the mean cosine distance between the prediction and the target.

        Args:
            mapping: (flatten_batch) - association between the prediction and the target.
            prediction: (flatten_batch, output_dim) - prediction of the context
            target: (num_abstracts, output_dim) - target context

        Returns:
            loss: () - mean squared error
        """

        cosine_distance = self.cosine_distance(mapping, prediction, target)

        loss = cosine_distance.mean()

        return loss


def test_cosine_loss():
    loss = CosineLoss()
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

    cosine_distance = loss.cosine_distance(mapping, prediction, target)

    assert cosine_distance.shape == (4,)
    assert cosine_distance[2] == 0.0


if __name__ == "__main__":
    test_cosine_loss()
    print("CosineLoss test passed")
