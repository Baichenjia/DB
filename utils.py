import multiprocessing
import os
import platform
from functools import partial

import numpy as np
import tensorflow as tf
from baselines.common.tf_util import normc_initializer
from mpi4py import MPI
import tensorflow_probability as tfp
import os
import numpy as np
tfd = tfp.distributions

layers = tf.keras.layers


def bcast_tf_vars_from_root(sess, vars):
    """
    Send the root node's parameters to every worker.

    Arguments:
      sess: the TensorFlow session.
      vars: all parameter variables including optimizer's
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for var in vars:
        if rank == 0:
            MPI.COMM_WORLD.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))


def get_mean_and_std(array):
    comm = MPI.COMM_WORLD
    task_id, num_tasks = comm.Get_rank(), comm.Get_size()
    local_mean = np.array(np.mean(array))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    mean = sum_of_means / num_tasks

    n_array = array - mean
    sqs = n_array ** 2
    local_mean = np.array(np.mean(sqs))
    sum_of_means = np.zeros((), dtype=np.float32)
    comm.Allreduce(local_mean, sum_of_means, op=MPI.SUM)
    var = sum_of_means / num_tasks
    std = var ** 0.5
    return mean, std


def guess_available_gpus(n_gpus=None):
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("Couldn't guess the available gpus on this machine")


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    available_gpus = guess_available_gpus()

    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])


def guess_available_cpus():
    return int(multiprocessing.cpu_count())


def setup_tensorflow_session():
    num_cpu = guess_available_cpus()

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu
    )
    tf_config.gpu_options.allow_growth = True
    return tf.Session(config=tf_config)


def random_agent_ob_mean_std(env, nsteps=10000):
    ob = np.asarray(env.reset())
    if MPI.COMM_WORLD.Get_rank() == 0:
        obs = [ob]
        for _ in range(nsteps):
            ac = env.action_space.sample()
            ob, _, done, _ = env.step(ac)
            if done:
                ob = env.reset()
            obs.append(np.asarray(ob))
        mean = np.mean(obs, 0).astype(np.float32)
        std = np.std(obs, 0).mean().astype(np.float32)
    else:
        mean = np.empty(shape=ob.shape, dtype=np.float32)
        std = np.empty(shape=(), dtype=np.float32)
    MPI.COMM_WORLD.Bcast(mean, root=0)
    MPI.COMM_WORLD.Bcast(std, root=0)
    return mean, std


def layernorm(x):
    m, v = tf.nn.moments(x, -1, keep_dims=True)
    return (x - m) / (tf.sqrt(v) + 1e-8)


getsess = tf.get_default_session

fc = partial(tf.layers.dense, kernel_initializer=normc_initializer(1.))
activ = tf.nn.relu


def flatten_two_dims(x):
    return tf.reshape(x, [-1] + x.get_shape().as_list()[2:])


def unflatten_first_dim(x, sh):
    return tf.reshape(x, [sh[0], sh[1]] + x.get_shape().as_list()[1:])


def add_pos_bias(x):
    with tf.variable_scope(name_or_scope=None, default_name="pos_bias"):
        b = tf.get_variable(name="pos_bias", shape=[1] + x.get_shape().as_list()[1:], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        return x + b


def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    # nl=512, feat_dim=None, last_nl=0, layernormalize=0, batchnorm=False
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(x, units=feat_dim, activation=None))
    if last_nl is not None:
        x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x


# new add
class SmallConv(tf.keras.Model):
    def __init__(self, feat_dim, name=None):
        super(SmallConv, self).__init__(name=name)
        self.conv1 = layers.Conv2D(filters=32, kernel_size=8, strides=(4, 4), activation=tf.nn.leaky_relu)
        self.conv2 = layers.Conv2D(filters=64, kernel_size=4, strides=(2, 2), activation=tf.nn.leaky_relu)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation=tf.nn.leaky_relu)
        self.fc = layers.Dense(units=feat_dim, activation=None)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
        x = self.fc(x)
        return x


# new add
class ResBlock(tf.keras.Model):
    def __init__(self, hidsize):
        super(ResBlock, self).__init__()
        self.hidsize = hidsize
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.dense2 = layers.Dense(hidsize, activation=None)

    def call(self, xs):
        x, a = xs             
        res = self.dense1(tf.concat([x, a], axis=-1))
        res = self.dense2(tf.concat([res, a], axis=-1))
        assert x.get_shape().as_list()[-1] == self.hidsize and res.get_shape().as_list()[-1] == self.hidsize
        return x + res


# new add
class TransitionNetwork(tf.keras.Model):
    def __init__(self, hidsize=256, name=None):
        super(TransitionNetwork, self).__init__(name=name)
        self.hidsize = hidsize
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.residual_block1 = ResBlock(hidsize)
        self.residual_block2 = ResBlock(hidsize)
        self.dense2 = layers.Dense(hidsize, activation=None)

    def call(self, xs):
        s, a = xs
        sh = tf.shape(a)                                   # sh=(None,None,4)
        assert len(s.get_shape().as_list()) == 3 and s.get_shape().as_list()[-1] in [512, 256]
        assert len(a.get_shape().as_list()) == 3

        x = flatten_two_dims(s)                            # shape=(None,512)
        a = flatten_two_dims(a)                            # shape=(None,4)

        #
        x = self.dense1(tf.concat([x, a], axis=-1))        # (None, 256)
        x = self.residual_block1([x, a])                   # (None, 256)
        x = self.residual_block2([x, a])                   # (None, 256)
        x = self.dense2(tf.concat([x, a], axis=-1))        # (None, 256)
        x = unflatten_first_dim(x, sh)                     # shape=(None, None, 256)
        return x


class GenerativeNetworkGaussianFix(tf.keras.Model):
    def __init__(self, hidsize=256, outsize=512, name=None):
        super(GenerativeNetworkGaussianFix, self).__init__(name=name)
        self.outsize = outsize
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.dense2 = layers.Dense(outsize, activation=tf.nn.leaky_relu)
        self.var_single = tf.Variable(1.0, trainable=True)

        self.residual_block1 = tf.keras.Sequential([
            layers.Dense(hidsize, activation=tf.nn.leaky_relu),   # 256
            layers.Dense(hidsize, activation=None)
        ])
        self.residual_block2 = tf.keras.Sequential([
            layers.Dense(hidsize, activation=tf.nn.leaky_relu),   # 256
            layers.Dense(hidsize, activation=None)
        ])
        self.residual_block3 = tf.keras.Sequential([
            layers.Dense(outsize, activation=tf.nn.leaky_relu),   # 512
            layers.Dense(outsize, activation=None)
        ])

    def call(self, z):
        sh = tf.shape(z)                       # z, sh=(None,None,128)
        assert z.get_shape().as_list()[-1] == 128 and len(z.get_shape().as_list()) == 3
        z = flatten_two_dims(z)                # shape=(None,128)

        x = self.dense1(z)                            # (None, 256)
        x = x + self.residual_block1(x)               # (None, 256)
        x = x + self.residual_block2(x)               # (None, 256)

        # variance
        var_tile = tf.tile(tf.expand_dims(tf.expand_dims(self.var_single, axis=0), axis=0), [16*128, self.outsize])

        # mean
        x = self.dense2(x)                            # (None, 512)
        x = x + self.residual_block3(x)               # (None, 512) mean

        # concat and return
        x = tf.concat([x, var_tile], axis=-1)         # (None, 1024)
        x = unflatten_first_dim(x, sh)                # shape=(None, None, 1024)
        return x


class GenerativeNetworkGaussian(tf.keras.Model):
    def __init__(self, hidsize=256, outsize=512, name=None):
        super(GenerativeNetworkGaussian, self).__init__(name=name)
        self.dense1 = layers.Dense(hidsize, activation=tf.nn.leaky_relu)
        self.dense2 = layers.Dense(outsize, activation=tf.nn.leaky_relu)
        self.dense3 = layers.Dense(outsize*2, activation=tf.nn.leaky_relu)

        self.residual_block1 = tf.keras.Sequential([
            layers.Dense(hidsize, activation=tf.nn.leaky_relu),   # 256
            layers.Dense(hidsize, activation=None)
        ])
        self.residual_block2 = tf.keras.Sequential([
            layers.Dense(hidsize, activation=tf.nn.leaky_relu),   # 256
            layers.Dense(hidsize, activation=None)
        ])
        self.residual_block3 = tf.keras.Sequential([
            layers.Dense(outsize, activation=tf.nn.leaky_relu),   # 512
            layers.Dense(outsize, activation=None)
        ])

    def call(self, z):
        sh = tf.shape(z)                       # z, sh=(None,None,128)
        assert z.get_shape().as_list()[-1] == 128 and len(z.get_shape().as_list()) == 3
        z = flatten_two_dims(z)                # shape=(None,128)

        x = self.dense1(z)                     # (None, 256)
        x = x + self.residual_block1(x)        # (None, 256)
        x = x + self.residual_block2(x)        # (None, 256)
        x = self.dense2(x)                     # (None, 512)
        x = x + self.residual_block3(x)        # (None, 512)
        x = self.dense3(x)                     # (None, 1024)
        x = unflatten_first_dim(x, sh)         # shape=(None, None, 1024)
        return x


class ProjectionHead(tf.keras.Model):
    def __init__(self, name=None):
        super(ProjectionHead, self).__init__(name=name)
        self.dense1 = layers.Dense(256, activation=None)
        self.dense2 = layers.Dense(128, activation=None)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x, ln=False):
        assert x.get_shape().as_list()[-1] == 512 and len(x.get_shape().as_list()) == 3
        x = flatten_two_dims(x)        # shape=(None,512)
        x = self.dense1(x)             # shape=(None,256)
        x = self.ln1(x)                # layer norm
        x = tf.nn.relu(x)              # relu
        x = self.dense2(x)             # shape=(None,128)
        x = self.ln2(x)
        return x


class ContrastiveHead(tf.keras.Model):
    def __init__(self, temperature, z_dim=128, name=None):
        super(ContrastiveHead, self).__init__(name=name)
        self.W = tf.Variable(tf.random.uniform((z_dim, z_dim)), name='W_Contras')
        self.temperature = temperature

    def call(self, z_a_pos):
        z_a, z_pos = z_a_pos
        Wz = tf.linalg.matmul(self.W, z_pos, transpose_b=True)  # (z_dim,B) Wz.shape = (50,32)
        logits = tf.linalg.matmul(z_a, Wz)                      # (B,B)     logits.shape = (32,32)
        logits = logits - tf.reduce_max(logits, 1)[:, None]     # logits
        logits = logits * self.temperature
        return logits


def rec_log_prob(rec_params, s_next, min_sigma=1e-2):
    # rec_params.shape = (None, None, 1024)
    distr = normal_parse_params(rec_params, min_sigma)
    log_prob = distr.log_prob(s_next)               # (None, None, 512)
    assert len(log_prob.get_shape().as_list()) == 3 and log_prob.get_shape().as_list()[-1] == 512
    return tf.reduce_sum(log_prob, axis=-1)


def normal_parse_params(params, min_sigma=0.0):
    n = params.shape[0]
    d = params.shape[-1]                    # channel
    mu = params[..., :d // 2]               # 
    sigma_params = params[..., d // 2:]
    sigma = tf.math.softplus(sigma_params)
    sigma = tf.clip_by_value(t=sigma, clip_value_min=min_sigma, clip_value_max=1e5)

    distr = tfd.Normal(loc=mu, scale=sigma)   # 
    return distr


def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)



import distutils.spawn
import subprocess


def save_np_as_mp4(frames, filename, frames_per_sec=30):
    print(filename)
    if distutils.spawn.find_executable('avconv') is not None:
        backend = 'avconv'
    elif distutils.spawn.find_executable('ffmpeg') is not None:
        backend = 'ffmpeg'
    else:
        raise NotImplementedError(
            """Found neither the ffmpeg nor avconv executables. On OS X, you can install ffmpeg via `brew install ffmpeg`. On most Ubuntu variants, `sudo apt-get install ffmpeg` should do it. On Ubuntu 14.04, however, you'll need to install avconv with `sudo apt-get install libav-tools`.""")

    h, w = frames[0].shape[:2]
    output_path = filename
    cmdline = (backend,
               '-nostats',
               '-loglevel', 'error',  # suppress warnings
               '-y',
               '-r', '%d' % frames_per_sec,

               # input
               '-f', 'rawvideo',
               '-s:v', '{}x{}'.format(w, h),
               '-pix_fmt', 'rgb24',
               '-i', '-',  # this used to be /dev/stdin, which is not Windows-friendly

               # output
               '-vcodec', 'libx264',
               '-pix_fmt', 'yuv420p',
               output_path)

    print('saving ', output_path)
    if hasattr(os, 'setsid'):                            # setsid not present on Windows
        process = subprocess.Popen(cmdline, stdin=subprocess.PIPE, preexec_fn=os.setsid)
    else:
        process = subprocess.Popen(cmdline, stdin=subprocess.PIPE)
    process.stdin.write(np.array(frames).tobytes())
    process.stdin.close()
    ret = process.wait()
    if ret != 0:
        print("VideoRecorder encoder exited with status {}".format(ret))


# ExponentialSchedule
class ExponentialSchedule(object):
    def __init__(self, start_value, decay_factor, end_value, outside_value=None):
        """Exponential Schedule.
           y = start_value * (1.0 - decay_factor) ^ t
        """
        assert 0.0 <= decay_factor <= 1.0
        self.start_value = start_value
        self.decay_factor = decay_factor
        self.end_value = end_value

    def value(self, t):
        v = self.start_value * np.power(1.0 - self.decay_factor,  t/int(1e5))
        return np.maximum(v, self.end_value)
