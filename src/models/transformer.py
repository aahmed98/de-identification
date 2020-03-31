import tensorflow as tf
import numpy as np
from dotmap import DotMap

def scaled_dot_product_attention(query, key, value, mask):
  """Calculate the attention weights. """
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask to zero out padding tokens
  if mask is not None:
    logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k)
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, hparams, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = hparams.num_heads
    self.d_model = hparams.d_model

    assert self.d_model % self.num_heads == 0

    self.depth = self.d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(self.d_model)
    self.key_dense = tf.keras.layers.Dense(self.d_model)
    self.value_dense = tf.keras.layers.Dense(self.d_model)

    self.dense = tf.keras.layers.Dense(self.d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # scaled dot-product attention
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concatenation of heads
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # final linear layer
    outputs = self.dense(concat_attention)

    return outputs


def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  mask =  mask[:, tf.newaxis, tf.newaxis, :]
  # print(mask.numpy())
  return mask


def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, hparams):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(hparams.vocab_size,
                                                 hparams.d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(hparams, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, hparams.d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      hparams, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(hparams.dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs +
                                                               attention)
  outputs = tf.keras.layers.Dense(
      hparams.num_units, activation="relu")(attention)
  outputs = tf.keras.layers.Dense(hparams.d_model)(outputs)
  outputs = tf.keras.layers.Dropout(hparams.dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention +
                                                             outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(hparams, name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(hparams.vocab_size,
                                         hparams.d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(hparams.d_model, tf.float32))
  embeddings = PositionalEncoding(hparams)(embeddings)

  outputs = tf.keras.layers.Dropout(hparams.dropout)(embeddings)

  for i in range(hparams.num_layers):
    outputs = encoder_layer(
        hparams,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def transformer(hparams, name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  # dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)

  enc_outputs = encoder(hparams)(inputs=[inputs, enc_padding_mask])

  outputs = tf.keras.layers.Dense(
      units=hparams.tag_size, name="outputs", activation = "softmax")(enc_outputs)

  # return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)
  return tf.keras.Model(inputs=[inputs], outputs=outputs, name=name)

class Transformer(tf.keras.Model):
    """
    Transformer seq2seq.
    """
    def __init__(self,vocab_sz,tag_size,max_len):
        super(Transformer,self).__init__()
        self.vocab_size =  vocab_sz + 1 #add 1 because of weird problem with embedding lookup. only happens on large data. CPU/GPU related I think
        self.tag_size = tag_size
        self.max_len = max_len
        self.embedding_size = 64
        self.rnn_size = 128
        self.title = "transformer-complex"

        hyperparams = DotMap({
            'vocab_size' : self.vocab_size,
            'tag_size' : self.tag_size,
            'num_layers':2,
            'num_units':self.rnn_size,
            'd_model': self.embedding_size,
            'num_heads':2,
            'dropout':0.3,
            'name':"sample_transformer"        
            })

        self.transformer_full = transformer(hparams = hyperparams)

    @tf.function(experimental_relax_shapes=True)
    def call(self,inputs):
        """
        Inputs: (batch_size, max_len)
        Output: (batch_size, max_len, tag_size) 
        """
        output = self.transformer_full.call(inputs = [inputs])
        return output

    def loss(self,prbs,labels):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs)
        mask = tf.cast(tf.not_equal(labels, 0), tf.float32)
        loss = tf.multiply(loss, mask)
        return tf.reduce_sum(loss)

    def predict(self,inputs):
        probs = self.call(inputs)
        mle_output = tf.argmax(probs,axis=2).numpy().flatten()
        return mle_output