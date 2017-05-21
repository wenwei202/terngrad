import tensorflow as tf

def ternary_encoder(input_data):
  """Encoding and compressing the signs """
  a = tf.sign(input_data) # -1, 0, 1
  a = tf.add(a,1) # shift -1,0,1 to 0,1,2 (2'b00,2'b01,2'b10)
  a = tf.reshape(a,[-1])
  pad_size = 4 - tf.mod(tf.size(a), 4)
  pad = tf.range(0.0, pad_size)
  a = tf.concat([a, pad], 0)
  a_split1, a_split2, a_split3, a_split4 = tf.split(a,4) # assume the size is dividable by 4

  # encode 4 grads into 1 Byte
  sum_1 = tf.add(a_split1, a_split2*4)
  sum_2 = tf.add(a_split3*16, a_split4*64)
  sum_all = tf.add(sum_1, sum_2)
  encoded = tf.cast(sum_all, tf.uint8)
  return encoded

def ternary_decoder(encoded_data, scaler, shape):
  """Decoding the signs to float format """
  a = tf.cast(encoded_data, tf.int32)
  a_split1 = tf.mod(a,4)
  a_split2 = tf.to_int32(tf.mod(a/4,4))
  a_split3 = tf.to_int32(tf.mod(a/16,4))
  a_split4 = tf.to_int32(tf.mod(a/64,4))
  a = tf.concat([a_split1, a_split2, a_split3, a_split4], 0)
  real_size = tf.reduce_prod(shape)
  a = tf.to_float(a)
  a = tf.gather(a, tf.range(0,real_size))
  a = tf.reshape(a, shape)
  a = tf.subtract(a, 1)
  decoded = a*scaler
  return decoded

shape=[33, 33, 33, 333]
scaler=0.002
with tf.device('/gpu:1'):
  # binary gradient generator
  gradient = tf.random_normal(shape, stddev=0.001, name='a')
  zeros = tf.zeros(shape)
  abs_gradient = tf.abs(gradient)
  sign_gradient = tf.sign( gradient )
  rnd_sample = tf.random_uniform(shape,0,scaler)
  where_cond = tf.less(rnd_sample, abs_gradient)
  bin_gradient = tf.where(where_cond, sign_gradient * scaler, zeros)

  # encoder:  -1 0 1
  encoded_a = ternary_encoder(bin_gradient)

with tf.device('/gpu:0'):
  # decoder
  decoded_a = ternary_decoder(encoded_a, scaler, shape)
 
  err = tf.reduce_sum( tf.squared_difference(bin_gradient, decoded_a)  )
   
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
#config.allow_soft_placement = True
with tf.Session(config=config) as sess:
  for i in range(2000):
    res = sess.run(err)
    print i, res
