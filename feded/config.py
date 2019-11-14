"""
Classes for representing configurations for training federated models.
"""


class TrainingConfig:
    """A container class for model training configurations."""
    def __init__(self, batch_size: int, epochs: int, shuffle_buffer: int):
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_buffer = shuffle_buffer


class FeatureConfig: #TODO(jpgard): holds a set of features and various configurations
    # (e.g. values, dtype, handling of missing data, filtering, etc).
    def __init__(self):
        pass