from enum import Enum
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor


class Distance(str, Enum):
    """An enumerator to pass distance metric names across the package."""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


class BaseDistance:
    def distance_matrix(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError()


class Cosine(BaseDistance):
    def distance_matrix(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        if y is None:
            y = x
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        return 1.0 - torch.mm(x_norm, y_norm.t())


def get_triplet_mask(labels: torch.Tensor) -> torch.Tensor:
    """Creates a 3D mask of valid triplets for the batch-all strategy.

    Given a batch of labels with `shape = (batch_size,)`
    the number of possible triplets that can be formed is:
    batch_size^3, i.e. cube of batch_size,
    which can be represented as a tensor with `shape = (batch_size, batch_size, batch_size)`.
    However, a triplet is valid if:
    `labels[i] == labels[j] and labels[i] != labels[k]`
    and `i`, `j` and `k` are distinct indices.
    This function calculates a mask indicating which ones of all the possible triplets
    are actually valid triplets based on the given criteria above.

    Args:
        labels (torch.Tensor): Labels associated with embeddings in the batch. Shape: (batch_size,)

    Returns:
        torch.Tensor: Triplet mask. Shape: (batch_size, batch_size,  batch_size)
    """
    indices_equal = torch.eye(labels.size()[0], dtype=torch.bool, device=labels.device)
    indices_not_equal = torch.logical_not(indices_equal)

    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = torch.logical_and(
        torch.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k
    )

    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = labels_equal.unsqueeze(2)
    i_equal_k = labels_equal.unsqueeze(1)
    valid_indices = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    mask = torch.logical_and(distinct_indices, valid_indices)
    return mask


def get_anchor_positive_mask(
        labels_a: torch.Tensor, labels_b: Optional[torch.Tensor] = None
) -> Tensor:
    """Creates a 2D mask of valid anchor-positive pairs.

    Args:
        labels_a (torch.Tensor): Labels associated with embeddings in the batch A. Shape: (batch_size_a,)
        labels_b (torch.Tensor): Labels associated with embeddings in the batch B. Shape: (batch_size_b,)
        If `labels_b is None`, it assigns `labels_a` to `labels_b`.

    Returns:
        torch.Tensor: Anchor-positive mask. Shape: (batch_size_a, batch_size_b)
    """
    if labels_b is None:
        labels_b = labels_a

    mask = labels_a.expand(labels_b.shape[0], labels_a.shape[0]).t() == labels_b.expand(
        labels_a.shape[0], labels_b.shape[0]
    )

    if torch.equal(labels_a, labels_b):
        indices_equal = torch.eye(
            labels_a.size()[0], dtype=torch.bool, device=labels_a.device
        )
        indices_not_equal = torch.logical_not(indices_equal)
        mask = torch.logical_and(indices_not_equal, mask)

    return mask


def get_anchor_negative_mask(
        labels_a: torch.Tensor, labels_b: Optional[torch.Tensor] = None
) -> Tensor:
    if labels_b is None:
        labels_b = labels_a

    mask = labels_a.expand(labels_b.shape[0], labels_a.shape[0]).t() != labels_b.expand(
        labels_a.shape[0], labels_b.shape[0]
    )
    return mask


class TripletLoss(torch.nn.Module):
    """Implements Triplet Loss as defined in https://arxiv.org/abs/1503.03832

    It supports batch-all, batch-hard and batch-semihard strategies for online triplet mining.

    Args:
        margin: Margin value to push negative examples
            apart.
        distance_metric_name: Name of the distance function, e.g.,
            :class:`~quaterion.distances.Distance`.
        mining: Triplet mining strategy. One of
            `"all"`, `"hard"`, `"semi_hard"`.
        soft: If `True`, use soft margin variant of Hard Triplet Loss. Ignored in all other cases.
    """

    def __init__(
            self,
            margin: float = 0.5,
            distance_metric: str = Distance.COSINE,
            mining: str = "hard",
            soft: bool = False
    ):
        super().__init__()

        mining_types = ["all", "hard", "semi_hard"]
        if mining not in mining_types:
            raise ValueError(
                f"Unrecognized mining strategy: {mining}. Must be one of {', '.join(mining_types)}"
            )

        self._margin = margin
        self._mining = mining
        self._soft = soft

        if distance_metric == Distance.COSINE:
            self.distance_metric = Cosine()
        else:
            raise ValueError(f"Currently only cosine distance is implemented")

    def _hard_triplet_loss(
            self,
            embeddings_a: Tensor,
            groups_a: LongTensor,
            embeddings_b: Tensor,
            groups_b: LongTensor,
    ) -> Tensor:
        """
        Calculates Triplet Loss with hard mining between two sets of embeddings.

        Args:
            embeddings_a: (batch_size_a, vector_length) - Batch of embeddings.
            groups_a: (batch_size_a,) - Batch of labels associated with `embeddings_a`
            embeddings_b: (batch_size_b, vector_length) - Batch of embeddings.
            groups_b: (batch_size_b,) - Batch of labels associated with `embeddings_b`

        Returns:
            torch.Tensor: Scalar loss value.
        """
        dists = self.distance_metric.distance_matrix(embeddings_a, embeddings_b)

        anchor_positive_mask = get_anchor_positive_mask(groups_a, groups_b).float()
        anchor_positive_dists = anchor_positive_mask * dists
        hardest_positive_dists = anchor_positive_dists.max(dim=1)[0]

        anchor_negative_mask = get_anchor_negative_mask(groups_a, groups_b).float()
        anchor_negative_dists = dists + dists.max(dim=1, keepdim=True)[0] * (
                1.0 - anchor_negative_mask
        )
        hardest_negative_dists = anchor_negative_dists.min(dim=1)[0]

        triplet_loss = (
            F.softplus(hardest_positive_dists - hardest_negative_dists)
            if self._soft
            else F.relu(
                (hardest_positive_dists - hardest_negative_dists)
                / hardest_negative_dists.mean()
                + self._margin
            )
        )

        return triplet_loss.mean()

    def _semi_hard_triplet_loss(
            self,
            embeddings_a: Tensor,
            groups_a: Tensor,
            embeddings_b: Tensor,
            groups_b: Tensor,
    ) -> Tensor:
        """Compute triplet loss with semi-hard mining as described in https://arxiv.org/abs/1703.07737

        It encourages the positive distances to be smaller than the minimum negative distance
        among which are at least greater than the positive distance plus the margin
        (called semi-hard negative),
        i.e., D(a, p) < D(a, n) < D(a, p) + margin.
            If no such negative exists, it uses the largest negative distance instead.

            Inspired by https://github.com/tensorflow/addons/blob/master/tensorflow_addons/losses/triplet.py

        Args:
            embeddings_a: shape: (batch_size_a, vector_length) - Output embeddings from the
                encoder.
            groups_a: shape: (batch_size_a,) - Group ids associated with embeddings.
            embeddings: shape: (batch_size_b, vector_length) - Batch of bmbeddings
            groups_b: shape: (batch_size_b,) - Groups ids associated with `embeddings_b`

        Returns:
            Tensor: zero-size tensor, XBM loss value.
        """
        distances = self.distance_metric.distance_matrix(embeddings_a, embeddings_b)

        positive_indices = groups_a[:, None] == groups_b[None, :]
        negative_indices = groups_a[:, None] != groups_b[None, :]

        pos_distance = torch.masked_select(distances, positive_indices)
        neg_distance = torch.masked_select(distances, negative_indices)

        basic_loss = pos_distance[:, None] - neg_distance[None, :] + self._margin
        zero_loss = torch.clamp(basic_loss, min=0.0)
        semi_hard_loss = torch.clamp(zero_loss, max=self._margin)

        return torch.mean(semi_hard_loss)

    def forward(
            self,
            embeddings: Tensor,
            groups: LongTensor,
    ) -> Tensor:
        """Calculates Triplet Loss with specified embeddings and labels.

        Args:
            embeddings: shape: (batch_size, vector_length) - Batch of embeddings.
            groups: shape: (batch_size,) - Batch of labels associated with `embeddings`

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if self._mining == "all":
            dists = self.distance_metric.distance_matrix(embeddings)

            anchor_positive_dists = dists.unsqueeze(2)
            anchor_negative_dists = dists.unsqueeze(1)
            triplet_loss = anchor_positive_dists - anchor_negative_dists + self._margin

            mask = get_triplet_mask(groups).float()
            triplet_loss = mask * triplet_loss
            triplet_loss = F.relu(triplet_loss)

            num_positive_triplets = torch.sum((triplet_loss > 1e-16).float())
            triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        elif self._mining == "hard":
            triplet_loss = self._hard_triplet_loss(
                embeddings, groups, embeddings, groups
            )
        else:
            triplet_loss = self._semi_hard_triplet_loss(
                embeddings, groups, embeddings, groups
            )

        return triplet_loss

    def xbm_loss(
            self,
            embeddings: Tensor,
            groups: LongTensor,
            memory_embeddings: Tensor,
            memory_groups: LongTensor,
    ) -> Tensor:
        """Implement XBM loss computation for this loss.

        Args:
            embeddings: shape: (batch_size, vector_length) - Output embeddings from the
                encoder.
            groups: shape: (batch_size,) - Group ids associated with embeddings.
            memory_embeddings: shape: (memory_buffer_size, vector_length) - Embeddings stored
                in a ring buffer
            memory_groups: shape: (memory_buffer_size,) - Groups ids associated with `memory_embeddings`

        Returns:
            Tensor: zero-size tensor, XBM loss value.
        """
        if len(memory_groups) == 0 or self._mining == "all":
            return torch.tensor(
                0, device=embeddings.device
            )

        return (
            self._hard_triplet_loss(
                embeddings, groups, memory_embeddings, memory_groups
            )
            if self._mining == "hard"
            else self._semi_hard_triplet_loss(
                embeddings, groups, memory_embeddings, memory_groups
            )
        )


def generate_test_data(batch_size: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    embeddings = torch.randn(batch_size, 4)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    return embeddings, groups


def test_cosine_distance():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    distance = Cosine()
    result = distance.distance_matrix(x)
    expected = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    torch.testing.assert_close(result, expected)


def test_get_triplet_mask():
    labels = torch.tensor([0, 0, 1, 1])
    mask = get_triplet_mask(labels)
    assert mask.shape == (4, 4, 4)
    assert mask[0, 1, 2].item()
    assert not mask[0, 0, 1].item()


def test_get_anchor_positive_mask():
    labels = torch.tensor([0, 0, 1, 1])
    mask = get_anchor_positive_mask(labels)
    expected = torch.tensor([
        [False, True, False, False],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, True, False]
    ])
    torch.testing.assert_close(mask, expected)


def test_get_anchor_negative_mask():
    labels = torch.tensor([0, 0, 1, 1])
    mask = get_anchor_negative_mask(labels)
    expected = torch.tensor([
        [False, False, True, True],
        [False, False, True, True],
        [True, True, False, False],
        [True, True, False, False]
    ])
    torch.testing.assert_close(mask, expected)


def test_triplet_loss_init():
    with pytest.raises(ValueError):
        TripletLoss(mining="invalid")

    with pytest.raises(ValueError):
        TripletLoss(distance_metric="manhattan")


def test_triplet_loss_forward_all():
    loss_fn = TripletLoss(margin=0.5, mining="all")
    embeddings, groups = generate_test_data()
    loss = loss_fn(embeddings, groups)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_triplet_loss_forward_hard():
    loss_fn = TripletLoss(margin=0.5, mining="hard")
    embeddings, groups = generate_test_data()
    loss = loss_fn(embeddings, groups)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_triplet_loss_forward_semi_hard():
    loss_fn = TripletLoss(margin=0.5, mining="semi_hard")
    embeddings, groups = generate_test_data()
    loss = loss_fn(embeddings, groups)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_triplet_loss_soft():
    loss_fn = TripletLoss(margin=0.5, mining="hard", soft=True)
    embeddings, groups = generate_test_data()
    loss = loss_fn(embeddings, groups)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0


def test_triplet_loss_xbm():
    loss_fn = TripletLoss(margin=0.5, mining="hard")
    embeddings, groups = generate_test_data(6)
    memory_embeddings = torch.randn(4, 4)
    memory_groups = torch.tensor([0, 0, 1, 1])

    loss = loss_fn.xbm_loss(embeddings, groups, memory_embeddings, memory_groups)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0

    loss_fn = TripletLoss(margin=0.5, mining="semi_hard")
    loss = loss_fn.xbm_loss(embeddings, groups, memory_embeddings, memory_groups)
    assert isinstance(loss.item(), float)
    assert loss.item() >= 0

    empty_memory = torch.tensor([])
    assert loss_fn.xbm_loss(embeddings, groups, empty_memory, empty_memory).item() == 0


if __name__ == "__main__":
    test_functions = [
        test_cosine_distance,
        test_get_triplet_mask,
        test_get_anchor_positive_mask,
        test_get_anchor_negative_mask,
        test_triplet_loss_init,
        test_triplet_loss_forward_all,
        test_triplet_loss_forward_hard,
        test_triplet_loss_forward_semi_hard,
        test_triplet_loss_xbm,
        test_triplet_loss_soft
    ]

    failed_tests = 0
    for test in test_functions:
        try:
            test()
            print(f"✓ {test.__name__}")
        except Exception as e:
            failed_tests += 1
            print(f"✗ {test.__name__}")
            print(f"  Error: {str(e)}")

    if failed_tests:
        print(f"\n{failed_tests} tests failed")
        exit(1)
    else:
        print("\nAll tests passed successfully!")
        exit(0)
