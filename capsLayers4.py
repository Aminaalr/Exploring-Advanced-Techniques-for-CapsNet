import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, layers
import numpy as np


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            assert len(inputs) == 2
            inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(indices=tf.argmax(x, 1), depth=x.shape[1])
        masked = K.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config


import tensorflow as tf
#def squash(vectors, axis=-1):
    #s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    #scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    #squashed = scale * vectors
    #squashed = tf.tanh(squashed)  # Squash using tanh activation
    #return squashed
#def squash(vectors, axis=-1, alpha=0.5, beta=1.0):
    #s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    #scale = tf.sigmoid(beta * (s_squared_norm ** alpha)) 
    #return scale * vectors
#def squash(vectors, axis=-1, sharpness=0.5):
 #   s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
  #  scale = tf.sigmoid(sharpness * (s_squared_norm / (1 + s_squared_norm)))
   # return scale * vectors
#def squash(vectors, axis=-1, alpha=0.5):
 #   s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
  #  scale = (s_squared_norm ** alpha) / (1 + s_squared_norm) 
   # return scale * vectors
#def squash(vectors, axis=-1):
    #s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    #scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    #return scale * vectors
#def squash(vectors, axis=-1):
 #   s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
  #  scale = 1 / (1 + s_squared_norm)/4
   # return scale * vectors
#def squash(vectors, axis=-1):
  
  #with tf.name_scope('squash'):
    #squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    #norm = tf.sqrt(squared_norm + K.epsilon())
    #one_minus_exp_norm = 1 - tf.exp(norm)
    #scale = (1 / one_minus_exp_norm) * vectors / norm
    #return scale
''' 
def squash(vectors, axis=-1, alpha_param=None, epsilon=tf.keras.backend.epsilon()):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    if alpha_param is None:
       alpha_initializer = tf.ones_like(s_squared_norm)
       alpha = layers.Dense(1, use_bias=False, kernel_initializer=initializers.Constant(alpha_initializer))(s_squared_norm)
    else:
       alpha = alpha_param
    scale = alpha / (1.0 + alpha * s_squared_norm)
    return scale * vectors
''' 
'''
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + K.epsilon()
    squashing_factor = s_squared_norm / (0.5 + s_squared_norm)
    squash = squashing_factor * vectors / tf.sqrt(s_squared_norm)
    # Non-linear activation (tanh, sigmoid)
    squashed = tf.tanh(squash)
    return squashed
'''
''''
#s200
def squash(vectors, axis=-1):
  s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
  scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
  squashed_vectors = scale * vectors
  # Additional Analysis
  mean_magnitude = tf.reduce_mean(tf.sqrt(s_squared_norm), axis=axis)
  return squashed_vectors, mean_magnitude
  '''

#Litrature1 Squash by //Afriyie, Y., A. Weyori, B., & A. Opoku, A. (2022). Classification of blood cells using optimized Capsule networks. Neural Processing Letters, 54(6), 4809-4828///
'''
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = 1 / (1 + s_squared_norm)/2
    return scale *  vectors
'''
'''
#Litrature1 2 Squash User f(x) = (1 - 1 / exp(||x||)) * (x / ||x||) 
import tensorflow as tf

def squash(x):
    """
    Squashing function implementation using TensorFlow operations.
    
    Arguments:
    x -- Input tensor
    
    Returns:
    squashed -- Squashed output tensor
    """
    # Calculate the squared norm of the input tensor
    squared_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    
    # Compute the square root of the squared norm
    norm = tf.sqrt(squared_norm)
    
    # Compute the exponential term
    exp_term = tf.exp(-norm)
    
    # Compute the denominator term
    denom = 1 - (1 / (1 + exp_term))
    
    # Compute the squashed output
    squashed = (x / norm) * denom
    
    return squashed
'''

''''
#s18 Enhanced Squash ....
def squash(vectors, axis=-1):
    # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    
    # Layer normalization
    epsilon = 1e-6
    mean, variance = tf.nn.moments(vectors, axes=-1, keepdims=True)
    vectors = (vectors - mean) / tf.sqrt(variance + epsilon)
    
    return scale * vectors
'''
'''
#sTest2 Enhanced Squash ....
def squash(vectors, axis=-1, epsilon=1e-6):
 # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
  # LAN (implementation might vary depending on library)
    gamma = tf.keras.layers.Dense(units=vectors.shape[-1], activation='relu')(vectors)  # Learnable gain
    beta = tf.keras.layers.Dense(units=vectors.shape[-1], activation='tanh')(vectors)  # Learnable offset
    mean, variance = tf.nn.moments(vectors, axes=-1, keepdims=True)
    normalized_vectors = (vectors - mean) / tf.sqrt(variance + epsilon)
    return scale * normalized_vectors * gamma + beta  # Apply learned gain and offset
'''
'''''
#Ltrature 4
import tensorflow as tf
from tensorflow.keras import layers
def squash(vectors, axis=-1):
    # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors/5), axis, keepdims=True)
    scale =s_squared_norm / (0.5+s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors
''''''
'''
''''''
#depend on this equation
import tensorflow as tf
from tensorflow.keras import layers

def squash(vectors, axis=-1):
    # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors/5), axis, keepdims=True)
    #s_squared_norm2= tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = (0.5*s_squared_norm)/ (1+ 0.5*s_squared_norm) /  tf.sqrt(s_squared_norm + K.epsilon())
    # Layer normalization
    epsilon = 1e-6
    mean, variance = tf.nn.moments(vectors, axes=-1, keepdims=True)
    vectors = (vectors - mean) / tf.sqrt(variance + epsilon)
    
    return scale * vectors

'''
##S18
import tensorflow as tf
from tensorflow.keras import layers

def squash(vectors, axis=-1):
    # Non-linear activation function (squashing)
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1+ s_squared_norm) /  tf.sqrt(s_squared_norm + K.epsilon())
    # Layer normalization
    epsilon = 1e-6
    mean, variance = tf.nn.moments(vectors, axes=-1, keepdims=True)
    vectors = (vectors - mean) / tf.sqrt(variance + epsilon)
    
    return scale * vectors


'''
    
#s15  --> not good 
'''
def squash(vectors, axis=-1):
    epsilon = K.epsilon()  # Small constant for numerical stability
    squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    
    # Improved squash function using a dynamic squash factor
    squash_factor = squared_norm / (1 + squared_norm)
    scaled_vectors = vectors * squash_factor / (tf.sqrt(squared_norm + epsilon))
    
    # Apply a dynamic margin to enhance discriminative power
    margin = 0.2  # Adjust this margin based on your dataset and model behavior
    adjusted_norm = tf.clip_by_value(squared_norm - margin, 0, squared_norm)
    scaled_adjusted_vectors = scaled_vectors * (adjusted_norm / (1 + adjusted_norm))
    
    return scaled_adjusted_vectors
'''
'''
##s14 --> around 0.4 at epoch=2
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    
    # Learnable parameters for dynamic squashing
    alpha = K.random_uniform(shape=(1,), minval=0.9, maxval=1.1)  # Initialize alpha randomly
    beta = K.random_uniform(shape=(1,), minval=-0.1, maxval=0.1)  # Initialize beta randomly
    
    dynamic_scale = alpha * tf.math.sigmoid(s_squared_norm + beta)
    return dynamic_scale * vectors
'''

'''
##s13 --> 0.68 at epoch=2
def squash(vectors, axis=-1):
    scaling_factor=0.5
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + K.epsilon()
    safe_norm = tf.sqrt(s_squared_norm)
    squash_factor = tf.pow(safe_norm, scaling_factor)
    unit_vector = vectors / safe_norm
    return squash_factor * unit_vector
'''

'''
#s17 
class LearnableSquash(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LearnableSquash, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(units=input_shape[-1], activation='sigmoid', use_bias=True)
        super(LearnableSquash, self).build(input_shape)

    def call(self, inputs):
        norm = K.sqrt(K.sum(K.square(inputs), axis=-1, keepdims=True) + K.epsilon())
        squashing_factor = self.dense(norm)
        squashed_output = squashing_factor * (inputs / norm)
        return squashed_output

    def compute_output_shape(self, input_shape):
        return input_shape

# Example usage:
def squash(vectors, axis=-1):
    squash_layer = LearnableSquash()
    output_vectors = squash_layer(vectors)
    return output_vectors
'''

'''
##s16 --->acc=0.90 at epoch=10
def gelu(x):
    return 0.5 * x * (1 + K.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + K.epsilon()
    squashing_factor = s_squared_norm / (0.5 + s_squared_norm)
    squash = squashing_factor * gelu(vectors)  # Using GELU activation
    squash = squash / tf.sqrt(s_squared_norm)
    return squash
'''

'''
##s4_2 --> no meaningful difference
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + K.epsilon()
    squashing_factor = s_squared_norm / (0.25 + s_squared_norm)
    squash = squashing_factor * vectors / tf.sqrt(s_squared_norm)
    return squash
'''

'''
##s12 --> 0.93 at epoch =10
def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + K.epsilon()
    squashing_factor = s_squared_norm / (0.5 + s_squared_norm)
    squash = squashing_factor * vectors / tf.sqrt(s_squared_norm)
    # Non-linear activation (tanh, sigmoid)
    squashed = tf.tanh(squash)
    return squashed


'''

'''
#s10
def squash(vectors, axis=-1):
    epsilon = K.epsilon()
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + epsilon
    s_norm = tf.sqrt(s_squared_norm)
    
    # Sensitivity factor for small-scale features
    sensitivity_factor = 1.0 + tf.exp(-s_norm)
    
    # Apply the sensitivity factor to the vectors
    squash = sensitivity_factor * vectors / (s_norm + epsilon)
    return squash
'''




#s9
'''ACC:0.05 
def squash(vectors, axis=-1):
    epsilon = K.epsilon()
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + epsilon
    s_norm = tf.sqrt(s_squared_norm)
    
    # Calculate the angles of the vectors in radians
    vector_angles = tf.atan2(s_norm, epsilon)
    
    # Dynamic normalization factor based on the vector angles
    dynamic_norm_factor = tf.divide(tf.sin(vector_angles), vector_angles)
    
    # Apply the dynamic normalization factor to the vectors
    dynamic_vectors = tf.multiply(vectors, dynamic_norm_factor)
    
    # Squash the dynamic vectors
    squash = dynamic_vectors / (s_norm + epsilon)
    return squash


'''
#s8
'''def squash(vectors, axis=-1):
    epsilon = K.epsilon()
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + epsilon
    s_norm = tf.sqrt(s_squared_norm)
    
    # Calculate the rotation factor, which is a function of the vector norms
    rotation_factor = tf.sin(s_norm) / (1 + tf.cos(s_norm))
    
    # Apply the rotation factor to the vectors
    rotated_vectors = tf.multiply(vectors, rotation_factor)
    
    # Squash the rotated vectors
    squash = rotated_vectors / (s_norm + epsilon)
    return squash
'''

'''
#s7
def squash(vectors, axis=-1):
    epsilon = K.epsilon()
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + epsilon
    s_norm = tf.sqrt(s_squared_norm)
    
    # Introduce a learnable scaling parameter
    alpha = tf.Variable(initial_value=1.0, trainable=True, name='alpha')
    
    # Weighted dynamic scaling based on the vector norms
    weighted_dynamic_scale = tf.pow(s_norm, alpha) / (1 + tf.pow(s_norm, alpha))
    
    # Apply the weighted dynamic scaling to the vectors
    squash = weighted_dynamic_scale * vectors / (s_norm + epsilon)
    return squash
'''
'''
###s6
def squash(vectors, axis=-1):
    epsilon = K.epsilon()
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + epsilon
    s_norm = tf.sqrt(s_squared_norm)
    
    # Learnable attention weights
    attention_weights = tf.Variable(initial_value=tf.ones(shape=vectors.shape[1:]), trainable=True, name='attention_weights')
    
    # Apply attention weights to the vectors
    weighted_vectors = vectors * attention_weights
    
    # Recalculate the squared norm with attention weights applied
    weighted_squared_norm = tf.reduce_sum(tf.square(weighted_vectors), axis, keepdims=True) + epsilon
    weighted_norm = tf.sqrt(weighted_squared_norm)
    
    # Squash the weighted vectors
    squash = weighted_vectors / (weighted_norm + epsilon)
    return squash



'''
'''#s5
def squash(vectors, axis=-1):
    epsilon = K.epsilon()
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + epsilon
    scale_factors = (0.5 + s_squared_norm) / (1 + s_squared_norm)
    
    # Introduce an adaptive factor that adjusts based on the vector norms
    adaptive_factor = tf.pow(s_squared_norm, 0.5) / (1 + tf.pow(s_squared_norm, 0.5))
    
    # Apply the adaptive factor to the scale factors
    scale = scale_factors * adaptive_factor
    
    # Squash the vectors
    squash = scale * vectors / tf.sqrt(s_squared_norm + epsilon)
    return squash
'''


'''
'''''
#s4
#####resulted in val_accuracy=0.9653 at epoch=50, and image_size=25 (not 75)
#####resulted in val_accuracy=0.9653 at epoch=50, and image_size=25 (not 75)

'''def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True) + K.epsilon()
    squashing_factor = s_squared_norm / (0.5 + s_squared_norm)
    squash = squashing_factor * vectors / tf.sqrt(s_squared_norm)
    return squash
 
'''
##S3

'''def squash(vectors, axis=-1):
    # Initialize dynamic routing coefficients
    routing_coeff = tf.Variable(initial_value=tf.zeros(vectors.shape[1:]), trainable=True)
    
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    
    # Apply dynamic routing coefficients to the scale
    dynamic_scale = scale * routing_coeff
    
    return dynamic_scale * vectors 
  
  #S1
'''
'''def squash(vectors, axis=-1, alpha_param=None, epsilon=tf.keras.backend.epsilon()):
    # Apply ReLU activation to introduce non-linearity
    vectors = tf.nn.relu(vectors)
    
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    if alpha_param is None:
        #alpha_initializer = 1.0
        alpha_initializer = tf.ones_like(s_squared_norm) 
        alpha = layers.Dense(1, use_bias=False, kernel_initializer=initializers.Constant(alpha_initializer))(s_squared_norm)
    else:
        alpha = alpha_param
    scale = alpha / (1.0 + alpha * s_squared_norm)
    return scale * vectors
'''
    
'''#S2
def squash(vectors, axis=-1, alpha_param=None, epsilon=tf.keras.backend.epsilon()):
    Attention mechanism to determine feature importance
    attention = layers.Dense(units=vectors.shape[-1], activation='softmax')(vectors)
    weighted_vectors = vectors * attention
    
    s_squared_norm = tf.reduce_sum(tf.square(weighted_vectors), axis=axis, keepdims=True)
    if alpha_param is None:
       alpha_initializer = 1.0
       alpha_initializer = tf.ones_like(s_squared_norm)
       alpha = layers.Dense(1, use_bias=False, kernel_initializer=initializers.Constant(alpha_initializer))(s_squared_norm)
    else:
       alpha = alpha_param
    scale = alpha / (1.0 + alpha * s_squared_norm)
    return scale * weighted_vectors
    ''''''
#def squash(vectors, axis=-1):
  # with tf.name_scope('squash'):
  #  squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    # Normalize by the squared norm (optional)
    # norm = tf.sqrt(squared_norm + K.epsilon())
    # vectors = vectors / norm  # Uncomment for L2 norm normalization
    #scale = squared_norm / (1 + squared_norm) / tf.sqrt(squared_norm + K.epsilon())
   # return tf.clip_by_value(scale * vectors, clip_value_min=0.0, clip_value_max=1.0)
   
#def squash(vectors, axis=-1, alpha=1.0, epsilon=K.epsilon()):
 #   s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
  #  scale = 1.0 / (1.0 + alpha * s_squared_norm)  # Efficient scaling
   # return scale * vectors
#def squash(vectors, axis=-1, alpha_param=None, epsilon=K.epsilon()):
   # s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
   # if alpha_param is None:
      # Use fixed alpha (replace with your desired fixed value)
      #alpha = 1.0
   # else:
      # Use learnable alpha parameter
    #  alpha = alpha_param
    #scale = 1.0 / (1.0 + alpha * s_squared_norm)
   # return scale * vectors
#def squash(vectors, axis=-1):
    #s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    #scale = 1/ (1 + s_squared_norm)
    #return scale * vectors


#def squash(vectors, s_squared_norm, axis=-1, alpha_param=None, epsilon=K.epsilon()):
  #if alpha_param is None:
    # Use a Dense layer with one unit and no activation
    #alpha_layer = layers.Dense(1, use_bias=False, activation=None, name='alpha')
    # Set alpha_layer weights to ones
    #alpha_layer.set_weights([np.ones_like(s_squared_norm)])
    #alpha = alpha_layer(s_squared_norm)
  ##else:
    # Use provided learnable alpha parameter
    #alpha = alpha_param

  #scale = alpha / (1.0 + alpha * s_squared_norm)
  #squashed_vectors = scale * vectors
  #return squashed_vectors, s_squared_norm
  '''
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):        
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])
        inputs_hat = tf.squeeze(tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled))
        b = tf.zeros(shape=[inputs.shape[0], self.num_capsule, 1, self.input_num_capsule])
        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.matmul(c, inputs_hat))  
            if i < self.routings - 1:
                b += tf.matmul(outputs, inputs_hat, transpose_b=True)      

        return tf.squeeze(outputs)

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

        

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)