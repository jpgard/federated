import tensorflow as tf

class ClassBalancedBinaryCrossEntropy:
    """Class-balanced binary cross-entropy. Computes a weighted sum of binary
    cross-entropy, where each class receives equal weight regardless of sample size."""
    def __call__(self, y_true, y_pred, *unused_args, **unused_kwargs):
        bce = tf.losses.BinaryCrossentropy()
        pos_elements_loss = bce(y_true, y_pred,
                                sample_weight=tf.cast(y_true, dtype=tf.float32)
                                )
        pos_elements_loss = tf.reduce_mean(pos_elements_loss, name="PosClassMeanLoss")
        neg_elements_loss = bce(y_true, y_pred,
                                sample_weight=tf.math.subtract(
                                    1., tf.cast(y_true, dtype=tf.float32))
                                )
        neg_elements_loss = tf.reduce_mean(neg_elements_loss, name="NegClassMeanLoss")
        class_balanced_loss = tf.add(pos_elements_loss, neg_elements_loss,
                                             name="BalancedMeanLoss")
        return class_balanced_loss
