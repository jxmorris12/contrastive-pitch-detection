from typing import Callable, Dict
import tensorflow as tf

class ContrastiveModel(tf.keras.Model):
    embedding: tf.keras.layers.Embedding
    model: tf.keras.Model
    batch_size: int # needed for tf.eye() call in graph mode
    def __init__(self, model: tf.keras.Model, min_midi: int, max_midi: int, output_dim: int):
        super().__init__()
        self.temperature = 1.0 # TODO(jxm): consider making temperature learnable like in CLIP
        self.num_labels = (max_midi - min_midi + 1)
        self.embedding = tf.keras.layers.Embedding(
            self.num_labels, output_dim,
            embeddings_initializer="glorot_uniform",
            trainable=True,
        )
        self.model = model
        self.batch_size = -1 # set during call to build()

    @property
    def embedding_table(self):
        return self.embedding.trainable_weights[0]

    def build(self, input_shape) -> None:
        batch_size, frame_length = input_shape
        self.batch_size = batch_size
        self.embedding.build((batch_size, self.num_labels))
        super().build(input_shape)

    def get_loss_fn(self) -> Callable:
        def loss_fn(labels, A_f):
            # We sum feature representations of multiple notes to make a chord feature representation.
            # TODO optionally add MLP layer to joint embedding.
            N_f = labels @ self.embedding_table # [b,n] @ [n, d] -> [b, d]
            # Make sure the shapes match up.
            # TODO: eager mode kills this assertion
            # assert A_f.shape == N_f.shape
            # Normalize to create embeddings.
            # TODO(jxm) should I do bilinear interpolation here?
            A_e = tf.math.l2_normalize(A_f, axis=1)
            N_e = tf.math.l2_normalize(N_f, axis=1)
            logits = (A_e @ tf.transpose(N_e)) * tf.math.exp(self.temperature)
            # Symmetric loss function
            labels = tf.eye(self.batch_size) # Identity matrix
            loss_a = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=0)
            loss_n = tf.nn.softmax_cross_entropy_with_logits(labels, logits, axis=1)
            return (loss_a + loss_n)/2
        return loss_fn
    
    def get_probs(self, A_f) -> tf.Tensor:
        """Do inference by finding the most likely embedding.
        
        Args: 
            A_f (tf.Tensor): is a batch of representation vectors produced by the model
                for a given audio input. of shape [b, d]
        Returns:
            tf.Tensor of shape [b, n], probabilities for each note for each input in the
                batch
        """
        logits = A_f @ tf.transpose(self.embedding_table) # [b, d] @ [d, n] -> [b, n]
        return tf.math.sigmoid(logits)

    def call(self, inputs) -> Dict[str, tf.Tensor]:
        A_f = self.model(inputs) # [n, d]
        return {
            'probs': self.get_probs(A_f),
            'A_f': A_f
        }


