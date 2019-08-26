from typing import Tuple
import tensorflow as tf
import tensorflow_hub as hub


class MultiHopAttentionLightweightModel:
    def __init__(
        self, joint_space: int, num_layers: int, attn_size: int, attn_hops: int
    ):
        self.captions = tf.placeholder(shape=None, dtype=tf.string)
        caption_words = tf.string_split([self.captions]).values
        captions_len = tf.shape(caption_words)
        self.text_encoded = self.text_encoder_graph(
            [caption_words], captions_len, joint_space, num_layers
        )
        self.attended_captions, self.text_alphas = self.attention_graph(
            attn_size, attn_hops, self.text_encoded, "siamese_attention"
        )
        self.saver_loader = tf.train.Saver()

    @staticmethod
    def text_encoder_graph(
        captions: tf.Tensor, captions_len: tf.Tensor, joint_space: int, num_layers: int
    ):
        """Encodes the text it gets as input using a bidirectional rnn.

        Args:
            captions: The inputs.
            captions_len: The length of the inputs.
            joint_space: The space where the encoded images and text are going to be
            projected to.
            num_layers: The number of layers in the Bi-RNN.

        Returns:
            The encoded text.

        """
        with tf.variable_scope(name_or_scope="text_encoder"):
            elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
            embeddings = elmo(
                inputs={"tokens": captions, "sequence_len": captions_len},
                signature="tokens",
                as_dict=True,
            )["elmo"]
            cell_fw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(joint_space) for _ in range(num_layers)]
            )
            cell_bw = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.GRUCell(joint_space) for _ in range(num_layers)]
            )
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw,
                cell_bw,
                embeddings,
                sequence_length=captions_len,
                dtype=tf.float32,
            )

            return tf.add(output_fw, output_bw) / 2

    @staticmethod
    def attention_graph(
        attn_size: int, attn_hops: int, encoded_input: tf.Tensor, scope: str
    ):
        """Applies attention on the encoded image and the encoded text.

        As per: https://arxiv.org/abs/1703.03130

        The "A structured self-attentative sentence embedding" paper goes through
        the attention mechanism applied here.

        Args:
            attn_size: The size of the attention.
            attn_hops: How many hops of attention to apply.
            encoded_input: The encoded input, can be both the image and the text.
            scope: The scope of the graph block.

        Returns:
            Attended output.

        """
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            # Shape parameters
            time_steps = tf.shape(encoded_input)[1]
            hidden_size = encoded_input.get_shape()[2].value

            # As per: http://proceedings.mlr.press/v9/glorot10a.html
            # Trainable parameters
            w_omega = tf.get_variable(
                name="w_omega",
                shape=[hidden_size, attn_size],
                initializer=tf.glorot_uniform_initializer(),
            )
            b_omega = tf.get_variable(
                name="b_omega", shape=[attn_size], initializer=tf.zeros_initializer()
            )
            u_omega = tf.get_variable(
                name="u_omega",
                shape=[attn_size, attn_hops],
                initializer=tf.glorot_uniform_initializer(),
            )
            # Apply attention
            # [B * T, H]
            encoded_input_reshaped = tf.reshape(encoded_input, [-1, hidden_size])
            # [B * T, A_size]
            v = tf.tanh(tf.matmul(encoded_input_reshaped, w_omega) + b_omega)
            # [B * T, A_heads]
            vu = tf.matmul(v, u_omega)
            # [B, T, A_hops]
            vu = tf.reshape(vu, [-1, time_steps, attn_hops])
            # [B, A_hops, T]
            vu_transposed = tf.transpose(vu, [0, 2, 1])
            # [B, A_hops, T]
            alphas = tf.nn.softmax(vu_transposed, name="alphas", axis=2)
            # [B, A_hops, H]
            output = tf.matmul(alphas, encoded_input)
            # [B, A_hops * H]
            output = tf.layers.flatten(output)
            # [B, A_hops * H] normalized output
            output = tf.math.l2_normalize(output, axis=1)

            return output, alphas

    def init(self, sess: tf.Session, checkpoint_path: str = None) -> None:
        """Initializes all variables in the graph.

        Args:
            sess: The active session.
            checkpoint_path: Path to a valid checkpoint.

        Returns:
            None

        """
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        if checkpoint_path is not None:
            self.saver_loader.restore(sess, checkpoint_path)
