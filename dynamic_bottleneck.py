import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utils import getsess

tfd = tfp.distributions

from utils import flatten_two_dims, unflatten_first_dim, SmallConv, TransitionNetwork, normal_parse_params, \
        ProjectionHead, ContrastiveHead, rec_log_prob, GenerativeNetworkGaussianFix


class DynamicBottleneck(object):
    def __init__(self, policy, tau, loss_kl_weight, loss_l2_weight, loss_nce_weight, aug, feat_dim=512, scope='DB'):
        self.scope = scope
        self.feat_dim = feat_dim
        self.policy = policy
        self.hidsize = policy.hidsize         # 512
        self.ob_space = policy.ob_space       # Box(84, 84, 4)
        self.ac_space = policy.ac_space       # Discrete(4)
        self.obs = self.policy.ph_ob          # shape=(None,None,84,84,4)
        self.ob_mean = self.policy.ob_mean    # shape=(None,None,84,84,4)
        self.ob_std = self.policy.ob_std      # 1.8
        self.tau = tau                        # tau for update the momentum network
        self.loss_kl_weight = loss_kl_weight    # loss_kl_weight
        self.loss_l2_weight = loss_l2_weight    # loss_l2_weight
        self.loss_nce_weight = loss_nce_weight  # loss_nce_weight
        self.aug = aug

        with tf.variable_scope(scope):
            self.feature_conv = SmallConv(feat_dim=self.feat_dim, name="DB_main")  #  (None, None, 512)
            self.feature_conv_momentum = SmallConv(feat_dim=self.feat_dim, name="DB_momentum")  # (None, None, 512)
            self.transition_model = TransitionNetwork(name="DB_transition")          # (None, None, 256)
            self.generative_model = GenerativeNetworkGaussianFix(name="DB_generative")           # (None, None, 512)
            self.projection_head = ProjectionHead(name="DB_projection_main")              # projection head
            self.projection_head_momentum = ProjectionHead(name="DB_projection_momentum")   # projection head Momentum
            self.contrastive_head = ContrastiveHead(temperature=1.0, name="DB_contrastive")

            # (None,1,84,84,4)
            self.last_ob = tf.placeholder(dtype=tf.int32, shape=(None, 1) + self.ob_space.shape, name='last_ob')
            self.next_ob = tf.concat([self.obs[:, 1:], self.last_ob], 1)  # (None,None,84,84,4)

            self.features = self.get_features(self.obs)                   # (None,None,512)
            self.next_features = self.get_features(self.next_ob, momentum=True)    # (None,None,512) stop gradient
            self.ac = self.policy.ph_ac             # (None, None)
            self.ac_pad = tf.one_hot(self.ac, self.ac_space.n, axis=2)

            # transition model
            latent_params = self.transition_model([self.features, self.ac_pad])     # (None, None, 256)
            self.latent_dis = normal_parse_params(latent_params, 1e-3)              # Gaussian. mu, sigma=(None, None, 128)

            # prior
            sh = tf.shape(self.latent_dis.mean())                                   # sh=(None, None, 128)
            self.prior_dis = tfd.Normal(loc=tf.zeros(sh), scale=tf.ones(sh))

            # kl
            kl = tfp.distributions.kl_divergence(self.latent_dis, self.prior_dis)     # (None, None, 128)
            kl = tf.reduce_sum(kl, axis=-1)                                           # (None, None)

            # generative network
            latent = self.latent_dis.sample()                       # (None, None, 128) 
            rec_params = self.generative_model(latent)              # (None, None, 1024)
            assert rec_params.get_shape().as_list()[-1] == 1024 and len(rec_params.get_shape().as_list()) == 3
            rec_dis = normal_parse_params(rec_params, 0.1)          # distribution

            rec_vec = rec_dis.sample()                              # mean of rec_params
            assert rec_vec.get_shape().as_list()[-1] == 512 and len(rec_vec.get_shape().as_list()) == 3

            # contrastive projection
            z_a = self.projection_head(rec_vec)                                           # (None, 128)
            z_pos = tf.stop_gradient(self.projection_head_momentum(self.next_features))   # (None, 128)
            assert z_a.get_shape().as_list()[-1] == 128 and len(z_a.get_shape().as_list()) == 2

            # contrastive loss
            logits = self.contrastive_head([z_a, z_pos])                 # (batch_size, batch_size) 
            labels = tf.one_hot(tf.range(int(16*128)), depth=16*128)     # (batch_size, batch_size)
            rec_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # (batch_size, )
            rec_log_nce = -1. * rec_loss
            rec_log_nce = unflatten_first_dim(rec_log_nce, sh)           # shape=(None, None)   (128,128)

            # L2 loss
            log_prob = rec_dis.log_prob(self.next_features)              # (None, None, 512)
            assert len(log_prob.get_shape().as_list()) == 3 and log_prob.get_shape().as_list()[-1] == 512
            rec_log_l2 = tf.reduce_sum(log_prob, axis=-1)
            rec_log = rec_log_nce * self.loss_nce_weight + rec_log_l2 * self.loss_l2_weight

            # loss
            self.loss = kl * self.loss_kl_weight - rec_log               # kl 
            self.loss_info = {"DB_NCELoss": -1.*tf.reduce_mean(rec_log_nce),
                              "DB_NCELoss_w": -1. * tf.reduce_mean(rec_log_nce) * self.loss_nce_weight,
                              "DB_L2Loss": -1.*tf.reduce_mean(rec_log_l2),
                              "DB_L2Loss_w": -1.*tf.reduce_mean(rec_log_l2) * self.loss_l2_weight,
                              "DB_KLLoss": tf.reduce_mean(kl),
                              "DB_KLLoss_w": tf.reduce_mean(kl) * self.loss_kl_weight,
                              "DB_Loss": tf.reduce_mean(self.loss)}

            # intrinsic reward
            self.intrinsic_reward = self.intrinsic_contrastive()
            self.intrinsic_reward = tf.stop_gradient(self.intrinsic_reward)

        # update the momentum network
        self.init_updates, self.momentum_updates = self.get_momentum_updates(tau=self.tau)
        print("*** DB Total Components:", len(self.ib_get_vars(name='DB/')), ", Total Variables:", self.ib_get_params(self.ib_get_vars(name='DB/')), "\n")

    @staticmethod
    def ib_get_vars(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

    @staticmethod
    def ib_get_params(vars):
        return np.sum([np.prod(v.shape) for v in vars])

    def get_momentum_updates(self, tau):       # tau=0.001
        main_var = self.ib_get_vars(name='DB/DB_features/DB_main') + self.ib_get_vars(name="DB/DB_projection_main")
        momentum_var = self.ib_get_vars(name='DB/DB_features_1/DB_momentum') + self.ib_get_vars(name="DB/DB_projection_momentum")

        # print("\n\n momentum_var:", momentum_var)
        assert len(main_var) > 0 and len(main_var) == len(momentum_var)
        print("***In DB, feature & projection has ", len(main_var), "components, ", self.ib_get_params(main_var), "parameters.")

        soft_updates = []
        init_updates = []
        assert len(main_var) == len(momentum_var)
        for var, tvar in zip(main_var, momentum_var):
            init_updates.append(tf.assign(tvar, var))
            soft_updates.append(tf.assign(tvar, (1. - tau) * tvar + tau * var))
        assert len(init_updates) == len(main_var)
        assert len(soft_updates) == len(main_var)
        return tf.group(*init_updates), tf.group(*soft_updates)

    def get_features(self, x, momentum=False):     # x.shape=(None,None,84,84,4)
        x_has_timesteps = (x.get_shape().ndims == 5)      # True
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)                       # (None,84,84,4)

        if self.aug:
            print(x.get_shape().as_list())
            x = tf.image.random_crop(x, size=[128*16, 80, 80, 4])            # (None,80,80,4)
            x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], "SYMMETRIC")     # (None,88,88,4)
            x = tf.image.random_crop(x, size=[128*16, 84, 84, 4])            # (None,84,84,4)

        with tf.variable_scope(self.scope + "_features"):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            if momentum:
                x = tf.stop_gradient(self.feature_conv_momentum(x))   # (None,512)
            else:
                x = self.feature_conv(x)                              # (None,512)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)                            # (None,None,512)
        return x

    def intrinsic_contrastive(self):
        kl = tfp.distributions.kl_divergence(self.latent_dis, self.prior_dis)  # (None, None, 128)
        rew = tf.reduce_sum(kl, axis=-1)        # (None, None)
        return rew

    def calculate_db_reward(self, ob, last_ob, acs):
        n_chunks = 8
        n = ob.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)

        # compute reward
        rew = np.concatenate([getsess().run(self.intrinsic_reward,
                                            {self.obs: ob[sli(i)], self.last_ob: last_ob[sli(i)],
                                            self.ac: acs[sli(i)]}) for i in range(n_chunks)], 0)
        return rew
