from keras import backend as K
from keras.layers.convolutional import _Conv
from keras import activations
from keras.engine import InputSpec
from keras.layers.convolutional import Conv2D



class MaxMin(Conv2D):

    def __init__(self, filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(MaxMin, self).__init__(

            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

    def build(self, input_shape):
        super(MaxMin, self).build(input_shape)

    def call(self, x):

        output = super(MaxMin, self).call(x)
        output = K.concatenate([output, -output], axis=3)
        return output

    def compute_output_shape(self, input_shape):
        """The output shape is doubled along the axis representing channels due
           to concatenation of two identical sized Convolution layers.
        """
        output_shape = super(MaxMin, self).compute_output_shape(input_shape)

        output_shape = list(output_shape)

        output_shape[3] *= 2

        return tuple(output_shape)
