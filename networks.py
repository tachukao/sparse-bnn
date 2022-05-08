from typing import Iterable, Optional
import sonnet as snt
import tensorflow as tf
import tensorflow.math as tfm
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfpd

softplus = tfm.softplus
softplus_inverse = tfp.math.softplus_inverse


class BaseLinear(snt.Module):
    def __init__(self, output_size, name=None):
        super().__init__(name=name)
        self.output_size = output_size

    @property
    def loc(self):
        raise NotImplementedError("loc not implemented.")

    @property
    def scale(self):
        raise NotImplementedError("scale not implemented.")

    def kl(self):
        raise NotImplementedError("KL not implemented.")

    def sample_posterior(self, batch_size: int):
        raise NotImplementedError("sample_posterior not implemented.")

    @property
    def prior_scale(self):
        return tfm.exp(0.5 * self.prior_log_var)

    @snt.once
    def _initialize(self, x):
        self.input_size = x.shape[-1] + 1
        self.param_shape = [self.output_size, self.input_size]

        # initialize parameter values
        stdev = 1.0 / tf.sqrt(tf.cast(self.input_size, tf.float32))
        q_loc = stdev * tf.random.normal(self.param_shape)
        q_log_var = -9.0 + 1e-2 * tf.random.normal(self.param_shape)
        prior_log_var = 0.0

        self.q_loc = tf.Variable(q_loc, name="q_loc")
        self.q_log_var = tf.Variable(q_log_var, name="q_log_var")
        self.prior_log_var = tf.Variable(prior_log_var, name="prior_log_var")

    def __call__(self, x):
        self._initialize(x)
        batch_size = tf.shape(x)[0]
        param = self.sample_posterior(batch_size)
        w = param[..., :-1]
        b = param[..., -1]
        y = tf.linalg.matvec(w, x) + b
        return y

    def predict(self, x):
        loc = self.loc
        w = loc[..., :-1]
        b = loc[..., -1]
        return (x @ tf.transpose(w)) + b

    @property
    def kl(self):
        q_scale = tfm.exp(0.5 * self.q_log_var)
        d1 = tfpd.Normal(self.q_loc, q_scale)
        d2 = tfpd.Normal(
            tf.zeros(self.param_shape), self.prior_scale * tf.ones(self.param_shape)
        )
        return tf.reduce_sum(d1.kl_divergence(d2))


class NormalLinear(BaseLinear):
    """Linear layer with a standard Normal prior."""

    @property
    def loc(self):
        return self.q_loc

    @property
    def scale(self):
        return tfm.exp(0.5 * self.q_log_var)

    @property
    def predict_loc(self):
        return self.loc

    def sample_posterior(self, batch_size):
        return tfpd.Normal(self.loc, self.scale).sample(batch_size)


class JeffreysLinear(NormalLinear):
    """Linear layer with a Normal-Jeffreys prior

    References:
    [1] Kingma, Diederik P., Tim Salimans, and Max Welling. "Variational dropout and the local reparameterization trick." NIPS (2015).
    [2] Molchanov, Dmitry, Arsenii Ashukha, and Dmitry Vetrov. "Variational Dropout Sparsifies Deep Neural Networks." ICML (2017).
    [3] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    @snt.once
    def _initialize(self, x):
        super()._initialize(x)

        # initialize parameter values
        stdev = 1.0 / tf.sqrt(tf.cast(self.input_size, tf.float32))
        z_loc = tf.ones(self.param_shape)
        z_log_var = -12.0 + 1e-2 * tf.random.normal(self.param_shape)

        self.z_loc = tf.Variable(z_loc, name="z_loc")
        self.z_log_var = tf.Variable(z_log_var, name="z_log_var")

    @property
    def loc(self):
        return self.q_loc * self.z_loc

    @property
    def scale(self):
        q_var = tfm.exp(self.q_log_var)
        z_var = tfm.exp(self.z_log_var)
        p_var = q_var * (self.z_loc**2)
        p_var += z_var * (self.q_loc**2 + q_var)
        return tfm.exp(0.5 * tfm.log(p_var))

    @property
    def log_dropout_rate(self):
        epsilon = 1e-8
        log_alpha = self.z_log_var - tfm.log(self.z_loc**2 + epsilon)
        return log_alpha

    @property
    def z_kl(self):
        # KL(q(z)||p(z))
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self.log_dropout_rate
        kl = -k1 * tfm.sigmoid(k2 + k3 * log_alpha)
        kl += 0.5 * tfm.softplus(-log_alpha) + k1
        return tf.reduce_sum(kl)

    @property
    def kl(self):
        return super().kl + self.z_kl


class HorseshoeLinear(BaseLinear):
    """Linear layer with a horseshoe prior

    References:
    [1] Louizos, Christos, Karen Ullrich, and Max Welling. "Bayesian Compression for Deep Learning." NIPS (2017).
    """

    @snt.once
    def _initialize(self, x):
        super()._initialize(x)

        tau0 = 1.0
        sa_loc = 0.0
        sb_loc = 0.0
        sa_log_var = -9.0 + 1e-2 * tf.random.normal(())
        sb_log_var = -9.0 + 1e-2 * tf.random.normal(())

        a_loc = tf.zeros(self.param_shape)
        b_loc = tf.zeros(self.param_shape)
        a_log_var = -9.0 + 1e-2 * tf.random.normal(self.param_shape)
        b_log_var = -9.0 + 1e-2 * tf.random.normal(self.param_shape)

        self.tau0 = tf.Variable(tau0, name="tau0")
        self.sa_loc = tf.Variable(sa_loc, name="sa_loc")
        self.sb_loc = tf.Variable(sb_loc, name="sb_loc")
        self.sa_log_var = tf.Variable(sa_log_var, name="sa_log_var")
        self.sb_log_var = tf.Variable(sb_log_var, name="sb_log_var")
        self.a_loc = tf.Variable(a_loc, name="a_loc")
        self.b_loc = tf.Variable(b_loc, name="b_loc")
        self.a_log_var = tf.Variable(a_log_var, name="a_log_var")
        self.b_log_var = tf.Variable(b_log_var, name="b_log_var")

    @property
    def z_loc(self):
        s_loc = 0.5 * (self.sa_loc + self.sb_loc)
        zt_loc = 0.5 * (self.a_loc + self.b_loc)
        z_loc = zt_loc + s_loc
        return z_loc

    @property
    def z_var(self):
        sa_var = tfm.exp(self.sa_log_var)
        sb_var = tfm.exp(self.sb_log_var)
        s_var = 0.25 * (sa_var + sb_var)
        a_var = tfm.exp(self.a_log_var)
        b_var = tfm.exp(self.b_log_var)
        zt_var = 0.25 * (a_var + b_var)
        z_var = zt_var + s_var
        return z_var

    @property
    def scale(self):
        q_var = tfm.exp(self.q_log_var)
        z_loc = self.z_loc
        z_var = self.z_var
        v = (
            (tfm.exp(z_var) - 1)
            * tfm.exp(2.0 * z_loc + z_var)
            * (q_var + (self.q_loc**2.0))
        )
        v += q_var * tfm.exp(2.0 * z_loc + z_var)
        return tfm.exp(0.5 * tfm.log(v))

    @property
    def loc(self):
        return tfm.exp(self.z_loc + 0.5 * self.z_var) * self.q_loc

    def sample_posterior(self, batch_size: int):
        z = tfpd.LogNormal(self.z_loc, self.z_var).sample(batch_size)
        q_scale = tfm.exp(0.5 * self.q_log_var)
        wt = tfpd.Normal(self.q_loc, q_scale).sample(batch_size)
        return z * wt

    @property
    def sa_kl(self):
        kl = -tfm.log(self.tau0)
        sa_var = tfm.exp(self.sa_log_var)
        kl -= tfm.exp(self.sa_loc + 0.5 * sa_var)
        kl -= 0.5 * (self.sa_loc + self.sa_log_var + 1.0 + tfm.log(2.0))
        return kl

    @property
    def sb_kl(self):
        sb_var = tfm.exp(self.sb_log_var)
        kl = tfm.exp(0.5 * sb_var - self.sb_loc)
        kl -= 0.5 * (-self.sb_loc + self.sb_log_var + 1 + tfm.log(2.0))
        return kl

    @property
    def a_kl(self):
        a_var = tfm.exp(self.a_log_var)
        kl = tfm.exp(self.a_loc + 0.5 * a_var)
        kl -= 0.5 * (self.a_loc + self.a_log_var + 1.0 + tfm.log(2.0))
        return tf.reduce_sum(kl)

    @property
    def b_kl(self):
        b_var = tfm.exp(self.b_log_var)
        kl = tfm.exp(0.5 * b_var - self.b_loc)
        kl -= 0.5 * (-self.b_loc + self.b_log_var + 1.0 + tfm.log(2.0))
        return tf.reduce_sum(kl)

    @property
    def kl(self):
        return super().kl + self.sa_kl + self.sb_kl + self.a_kl + self.b_kl


layer_mapping = {
    "normal": NormalLinear,
    "jeffreys": JeffreysLinear,
    "horseshoe": HorseshoeLinear,
}


class BNN(snt.Module):
    def __init__(
        self,
        output_sizes: Iterable[int] = [300, 100, 10],
        prior: str = "normal",
        name=None,
    ):
        super().__init__(name=name)
        self.output_sizes = output_sizes
        self.flatten = snt.Flatten()
        build_layer = layer_mapping[prior]
        self.layers = [
            build_layer(size, name=f"layer{l}") for (l, size) in enumerate(output_sizes)
        ]

    def __call__(self, x):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = tf.nn.relu(x)
        x = self.layers[-1](x)
        return x

    def predict(self, x):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = layer.predict(x)
            x = tf.nn.relu(x)
        x = self.layers[-1](x)
        return x

    @property
    def kl(self):
        return sum(layer.kl for layer in self.layers)

    @property
    def scale(self):
        return tf.math.reduce_mean(
            [tf.math.reduce_mean(tf.math.abs(layer.scale)) for layer in self.layers]
        )
