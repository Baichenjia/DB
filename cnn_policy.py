import tensorflow as tf
from baselines.common.distributions import make_pdtype
from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim


class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        """ ob_space: (84,84,4);        ac_space: 4;
            ob_mean.shape=(84,84,4);    ob_std=1.7;            hidsize: 512;
            feat_dim: 512;              layernormalize: False;      nl: tf.nn.leaky_relu.
        """
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        with tf.variable_scope(scope):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space) 
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0] 

            sh = tf.shape(self.ph_ob)               # ph_ob.shape = (None,None,84,84,4)
            x = flatten_two_dims(self.ph_ob)        # x.shape = (None,84,84,4)

            self.flat_features = self.get_features(x, reuse=False)       # shape=(None,512)
            self.features = unflatten_first_dim(self.flat_features, sh)  # shape=(None,None,512)

            with tf.variable_scope(scope, reuse=False):
                x = fc(self.flat_features, units=hidsize, activation=activ)            # activ=tf.nn.relu
                x = fc(x, units=hidsize, activation=activ)                             # value and policy
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)         # logits, shape=(None,4)
                vpred = fc(x, name='value_function_output', units=1, activation=None)  # shape=(None,1)
            pdparam = unflatten_first_dim(pdparam, sh)             # shape=(None,None,4)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]   # value function shape=(None,None)
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)      # mean,neglogp,kl,entropy,sample
            self.a_samp = pd.sample()                              # 
            self.entropy = pd.entropy()                            # (None,None)
            self.nlp_samp = pd.neglogp(self.a_samp)                # -log pi(a|s)  (None,None)

    def get_features(self, x, reuse):    
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob):
        # ob.shape=(128,84,84,1),  ob[:,None].shape=(128,1,84,84,4)
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]


