class TrainingConfig:
    """A container class for model training configurations."""
    def __init__(self, batch_size: int, epochs: int, shuffle_buffer: int,
                 num_train_clients: int, batches_to_take: int):
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle_buffer = shuffle_buffer
        self.num_train_clients = num_train_clients
        self.batches_to_take = batches_to_take