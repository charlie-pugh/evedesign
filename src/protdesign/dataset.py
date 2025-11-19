from typing import Sequence
from protdesign.entity import SystemInstance


class LabeledInstanceDataset:
    """
    Basic mapping of instances to labels. Can be used for regression and classification
    tasks.

    Note: not handling multi-label case
    Note: semi-supervised case (missing labels) can be encoded with np.nan
    """
    def __init__(
        self,
        train_instances: Sequence[SystemInstance],
        train_values: Sequence[float],
        test_instances: Sequence[SystemInstance] | None = None,
        test_values: Sequence[float] | None = None,
    ):
        if len(train_instances) != len(train_values):
            raise ValueError(
                "Length of instances and values does not agree for training set"
            )

        if (test_instances is None and test_values is not None) or (test_instances is not None and test_values is None):
            raise ValueError(
                "Must specify both test_instances and test_values"
            )

        if test_instances is not None and test_values is not None:
            if len(test_instances) != len(test_values):
                raise ValueError(
                    "Length of instances and values does not agree for test set"
                )

        self.train_instances = train_instances
        self.train_values = train_values
        self.test_instances = test_instances
        self.test_values = test_values
