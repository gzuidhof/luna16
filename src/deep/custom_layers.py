import theano
import theano.tensor as T
import lasagne as nn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class SpatialDropoutLayer(Layer):
    """Spatial dropout layer
    Sets whole filter activations to zero with probability p. See notes for
    disabling dropout during testing.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    p : float or scalar tensor
        The probability of setting a value to zero
    rescale : bool
        If true the input is rescaled with input / (1-p) when deterministic
        is False.
    Notes
    -----
    The spatial dropout layer is a regularizer that randomly sets whole the
    values of whole features to zero. This is an adaptation of normal dropout,
    which is generally useful in fully convolutional settings, such as [1]_.

    It is also called a feature dropout layer.

    During training you should set deterministic to false and during
    testing you should set deterministic to true.
    If rescale is true the input is scaled with input / (1-p) when
    deterministic is false, see references for further discussion. Note that
    this implementation scales the input at training time.
    References
    ----------
    .. [1] Oliveira, G. Valada, A., Bollen, C., Bugard, W., Brox. T. (2016):
           Deep Learning for Human Part Discovery in Images. IEEE
           International Conference on Robotics and Automation (ICRA), IEEE,
           2016.
    """
    def __init__(self, incoming, p=0.5, rescale=True, **kwargs):
        super(SpatialDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.p = p
        self.rescale = rescale

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.p == 0:
            return input
        else:
            # Using theano constant to prevent upcasting
            one = T.constant(1)

            retain_prob = one - self.p
            if self.rescale:
                input /= retain_prob

            mask = _srng.binomial(input.shape[:2], p=retain_prob,
                                      dtype=theano.config.floatX)
            axes = [0, 1] + (['x'] * (input.ndim - 2))
            mask = mask.dimshuffle(*axes)

            return input * mask
