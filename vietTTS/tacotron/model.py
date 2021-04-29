# haiku tacotron model


from .config import *

class Tacotron(hk.Module):
  """Tacotron 2."""
  def __init__(self, is_training=True):
    super().__init__()
    self.is_training = is_training

    ## encoder
    self.enc_embed = hk.Embed(FLAGS.alphabet_size, embed_dim=FLAGS.text_embed_dim)
    self.enc_conv1 = hk.Conv1D(FLAGS.text_embed_dim, 5)
    self.enc_conv2 = hk.Conv1D(FLAGS.text_embed_dim, 5)
    self.enc_conv3 = hk.Conv1D(FLAGS.text_embed_dim, 5)
    self.enc_bn1 = hk.BatchNorm(True, True, 0.99)
    self.enc_bn2 = hk.BatchNorm(True, True, 0.99)
    self.enc_bn3 = hk.BatchNorm(True, True, 0.99)
    self.enc_lstm_fwd = hk.LSTM(FLAGS.text_lstm_dim//2) # forward lstm
    self.enc_lstm_bwd = hk.ResetCore(hk.LSTM(FLAGS.text_lstm_dim//2)) # backward lstm

    ## prenet
    self.prenet_fc1 = hk.Linear(256, with_bias=False) # my prenet ;) inspired by Firefox TTS batchnorm prenet
    self.prenet_fc2 = hk.Linear(256, with_bias=False) # my prenet ;) inspired by Firefox TTS batchnorm prenet
    ## decoder
    self.attn_lstm = hk.LSTM(FLAGS.dec_lstm_dim)
    self.decoder_lstm =  hk.LSTM(FLAGS.dec_lstm_dim)
    # self.lstm_stack = hk.deep_rnn_with_skip_connections([
    #     hk.LSTM(FLAGS.dec_lstm_dim),
    #     hk.LSTM(FLAGS.dec_lstm_dim),
    # ])          
    self.dec_proj = hk.Linear((1 + FLAGS.mel_dim) * FLAGS.max_reduce_factor)

    ## attention
    self.attn_W = hk.Linear(128, with_bias=True)
    self.attn_V = hk.Linear(2, with_bias=False)
    ## postnet
    self.postnet_convs = [hk.Conv1D(FLAGS.postnet_dim, 5) for _ in range(4) ] + [hk.Conv1D(FLAGS.mel_dim, 5)]
    self.postnet_bns = [hk.BatchNorm(True, True, 0.99) for _ in range(4)] + [None]
  
  def prenet(self, x, dropout):
    x = jax.nn.relu( self.prenet_fc1(x) )
    x = hk.dropout(hk.next_rng_key(), dropout, x) if dropout > 0 else x

    x = jax.nn.relu( self.prenet_fc2(x) )
    x = hk.dropout(hk.next_rng_key(), dropout, x) if dropout > 0 else x
    return x

  def text_conv_encoder(self, text: ndarray) -> ndarray:
    x = text
    x = self.enc_conv1(x)
    x = self.enc_bn1(x, is_training=self.is_training)
    x = jax.nn.relu(x)
    x = hk.dropout(hk.next_rng_key(), 0.5, x) if self.is_training else x
    x = self.enc_conv2(x)
    x = self.enc_bn2(x, is_training=self.is_training)
    x = jax.nn.relu(x)
    x = hk.dropout(hk.next_rng_key(), 0.5, x) if self.is_training else x
    x = self.enc_conv3(x)
    x = self.enc_bn3(x, is_training=self.is_training)
    x = jax.nn.relu(x)
    x = hk.dropout(hk.next_rng_key(), 0.5, x) if self.is_training else x
    return x
  
  def text_lstm_encoder(self, text: ndarray, text_len: ndarray) -> ndarray:
    N, L, D = text.shape
    h0c0 = self.enc_lstm_fwd.initial_state(N)
    x1_fwd, _ = hk.dynamic_unroll(self.enc_lstm_fwd, text, h0c0, time_major=False) 

    reset_mask = jnp.arange(0, L)[None, ] >= (text_len[:, None] - 1)
    reset_mask_bwd = jnp.flip(reset_mask, axis=1)
    text_bwd = jnp.flip(text, axis=1)
    h0c0 = self.enc_lstm_bwd.initial_state(N)
    x2_bwd, _ = hk.dynamic_unroll(self.enc_lstm_bwd, (text_bwd, reset_mask_bwd), h0c0, time_major=False)
    x2_fwd = jnp.flip(x2_bwd, axis=1)
    x = jnp.concatenate((x1_fwd, x2_fwd), axis=-1)
    return x

  def encode_text(self, text: ndarray, text_len: ndarray) -> ndarray:
    x = self.enc_embed(text)
    x = self.text_conv_encoder(x)
    x = self.text_lstm_encoder(x, text_len)
    return x

  def decode_mel(self, text: ndarray, mel: ndarray, reduce_factor: int = 1) -> ndarray:
    N, L, D = text.shape
    def loop_fn(inputs, prev_state):
      mel = inputs
      attn_lstm_hxcx, decoder_lstm_hxcx, context, attn_loc = prev_state
      x = jnp.concatenate((mel, context), axis=-1)
      x, new_attn_lstm_hxcx = self.attn_lstm(x, attn_lstm_hxcx)
      attn_hx = x

      # attention -> new context
      attn_params = self.attn_V(jnp.tanh(self.attn_W(x)))
      delta, scale = jnp.split(attn_params, 2, axis=-1)
      delta = jax.nn.softplus(delta) / 2  * reduce_factor
      # google GMMA paper suggests std = 10
      scale = jax.nn.softplus(scale + 5)
      new_loc = attn_loc + delta
      j = jnp.arange(0, L)[None, :]
      up = (j - new_loc + 0.5) / scale
      down = (j - new_loc - 0.5) / scale
      weight = jax.nn.sigmoid(up) - jax.nn.sigmoid(down) # N L
      new_context = jnp.sum( weight[..., None] * text, axis=1) # N D

      x = jnp.concatenate((attn_hx, new_context), axis=-1)
      decoder_hx, new_decoder_lstm_hxcx = self.decoder_lstm(x, decoder_lstm_hxcx)
      output = decoder_hx

      return (output, weight[0]), (new_attn_lstm_hxcx, new_decoder_lstm_hxcx, new_context, new_loc)
    
    initial_state = (
        self.attn_lstm.initial_state(N),
        self.decoder_lstm.initial_state(N),
        jnp.zeros((N, D)),
        jnp.zeros((N, 1))
    )
    (output, attn_weight), _ = hk.dynamic_unroll(loop_fn, mel, initial_state, time_major=False)
    dec = self.dec_proj(output)
    dec = rearrange(dec, 'N L (R D) -> N L R D', R=FLAGS.max_reduce_factor, D=1 + FLAGS.mel_dim)
    dec = dec[:, :, :reduce_factor, :]
    dec = rearrange(dec, 'N L R D -> N (L R) D', R=reduce_factor, D=1 + FLAGS.mel_dim)
    stop_token, predicted_mel = jnp.split(dec, [1, ], axis=-1)

    return stop_token, predicted_mel, attn_weight


  def postnet(self, mel: ndarray) -> ndarray:
    x = mel
    for conv, bn in zip(self.postnet_convs, self.postnet_bns):
      x = conv(x)
      if bn is not None:
        x = bn(x, is_training=self.is_training)
        x = jnp.tanh(x)
      x = hk.dropout(hk.next_rng_key(), 0.5, x) if self.is_training else x
    return x

  def __call__(self, inputs: TacotronInput, reduce_factor=1, mel_dropout=0.5):
    text = self.encode_text(inputs.text, inputs.text_len)
    mel = self.prenet(inputs.mel, mel_dropout)
    # mel = hk.dropout(hk.next_rng_key(), mel_dropout, mel) if self.is_training and mel_dropout > 0. else mel
    stop_token, predicted_mel, attn_weight = self.decode_mel(text, mel, reduce_factor)
    mel_residual = self.postnet(predicted_mel)
    hk.set_state('attn', attn_weight) # for logging...
    return TacotronOutput(stop_token, predicted_mel, mel_residual)