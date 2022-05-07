import sonnet as snt
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfpd

softplus = tfm.softplus
softplus_inverse = tfp.math.softplus_inverse


class Layer(snt.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    @snt.once
    def _initialize(self, x):
        self.input_size = x.shape[-1] + 1
        initial_w = tf.random.normal([self.output_size, self.input_size]) / tf.sqrt(
            tf.cast(self.input_size, tf.float32)
        )
        self._loc = tf.Variable(initial_w, name="loc")
        self._scale = tf.Variable(
            softplus_inverse(1e-3) * tf.ones([self.output_size, self.input_size]),
            name="scale",
        )
        self._prior_scale = tf.Variable(softplus_inverse(1e-3), name="prior_scale")

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def prior_scale(self):
        return softplus(self._prior_scale)

    @property
    def post_dist(self):
        post_dist = tfpd.Normal(self.loc, self.scale)
        return post_dist

    @property
    def prior_dist(self):
        shp = (self.output_size, self.input_size)
        return tfpd.Normal(tf.zeros(shp), self.prior_scale * tf.ones(shp))

    def __call__(self, x):
        self._initialize(x)
        batch_size = tf.shape(x)[0]
        param = self.post_dist.sample(batch_size)
        w = param[..., :-1]
        b = param[..., -1]
        y = tf.linalg.matvec(w, x) + b
        return y

    def kl(self):
        return tf.math.reduce_sum(self.post_dist.kl_divergence(self.prior_dist))

    def predict(self, x):
        w = self.loc[..., :-1]
        b = self.loc[..., -1]
        return (x @ tf.transpose(w)) + b


def NJLayer(Layer):
    """Linear layer with a Normal-Jeffreys prior

    References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    @snt.once
    def _initialize(self, x):
        init_scale = softplus_inverse(1e-3)
        self.input_size = x.shape[-1] + 1
        self.param_shape = [self.output_size, self.input_size]
        initial_w = tf.random.normal(self.param_shape) / tf.sqrt(
            tf.cast(self.input_size, tf.float32)
        )
        self._loc = tf.Variable(initial_w, name="loc")
        self._scale = tf.Variable(
            init_scale * tf.ones(self.param_shape),
            name="scale",
        )
        self._z_loc = tf.Variable(self.param_shape, name="z_loc")
        self._z_scale = tf.Variable(
            init_scale * tf.ones(self.param_shape), name="z_scale"
        )

    @property
    def loc(self):
        return self._loc

    @property
    def scale(self):
        return tf.math.softplus(self._scale)

    @property
    def z_scale(self):
        return tf.math.softplus(self._z_scale)

    @property
    def z_loc(self):
        return self._z_loc

    @property
    def post_dist(self):
        (
            self.z_mu.pow(2) * weight_var
            + z_var * self.weight_mu.pow(2)
            + z_var * weight_var
        )
        w_var = self.scale**2
        z_var = self.z_scale**2
        w_loc2 = self.loc**2
        z_loc2 = self.z_loc**2
        p_var = z_loc2 * w_var + z_var * w_loc2 + z_var * w_var
        p_scale = tfm.exp(tfm.log(p_var) * 0.5)
        p_loc = self.loc * self.z_loc
        return tfpd.Normal(p_loc, p_scale)

    @property
    def log_dropout_rate(self):
        epsilon = 1e-8
        log_alpha = tfm.log(self.z_scale) - tfm.log(self.z_loc**2 + epsilon)
        return log_alpha

    def __call__(self, x):
        self._initialize(x)
        batch_size = tf.shape(x)[0]
        param = self.post_dist.sample(batch_size)
        w = param[..., :-1]
        b = param[..., -1]
        y = tf.linalg.matvec(w, x) + b
        return y

    def kl(self):
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.log_droput_rate
        kl1 = -tf.reduce_sum(
            k1 * tfm.sigmoid(k2 + k3 * log_alpha) - 0.5 * tfm.softplus(-log_alpha) - k1
        )
        # KL(q(w|z)||p(w|z))
        kl2 = -tf.math.log(self.scale) + 0.5 * (self.scale**2 + self.loc**2) - 0.5
        return kl1 + kl2

    def predict(self, x):
        p_loc = self.post_dist.loc
        w = p_loc[..., :-1]
        b = p_loc[..., -1]
        return (x @ tf.transpose(w)) + b


layer_collection = {
    "normal": Layer,
    "jeffreys": NJLayer,
}


class BNN(snt.Module):
    def __init__(
        self, hidden_size: int, n_layers: int, weight_prior: str = "normal", name=None
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.flatten = snt.Flatten()
        build_layer = layer_collection[weight_prior]
        self.layers = [
            build_layer(hidden_size, name=f"layer{l}") for l in range(n_layers - 1)
        ]
        self.logits = build_layer(10, name="logit_layer")

    def __call__(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer(x)
            x = tf.nn.relu(x)
        x = self.logits(x)
        return x

    def predict(self, x):
        x = self.flatten(x)
        for layer in self.layers:
            x = layer.predict(x)
            x = tf.nn.relu(x)
        x = self.logits(tf.nn.relu(x))
        return x

    @property
    def kl(self):
        return sum(layer.kl() for layer in self.layers)

    @property
    def scale(self):
        return tf.math.reduce_mean(
            [tf.math.reduce_mean(tf.math.abs(layer.scale)) for layer in self.layers]
        )
