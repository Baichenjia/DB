#!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial
import os
import gym
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from dynamic_bottleneck import DynamicBottleneck
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit, StickyActionEnv
import datetime
from wrappers import PixelNoiseWrapper, RandomBoxNoiseWrapper
import json

getsess = tf.get_default_session


def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log, tf_sess, saver, logger_dir = get_experiment_environment(**args)
    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train(saver, logger_dir)


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()   

        self.policy = CnnPolicy(scope='pol',
                                ob_space=self.ob_space,
                                ac_space=self.ac_space,
                                hidsize=512,
                                feat_dim=512,
                                ob_mean=self.ob_mean,
                                ob_std=self.ob_std,
                                layernormalize=False,
                                nl=tf.nn.leaky_relu)

        self.dynamic_bottleneck = DynamicBottleneck(
                    policy=self.policy, feat_dim=512, tau=hps['momentum_tau'], loss_kl_weight=hps['loss_kl_weight'],
                    loss_nce_weight=hps['loss_nce_weight'], loss_l2_weight=hps['loss_l2_weight'], aug=hps['aug'])

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamic_bottleneck=self.dynamic_bottleneck
        )

        self.agent.to_report['db'] = tf.reduce_mean(self.dynamic_bottleneck.loss)
        self.agent.total_loss += self.agent.to_report['db']

        self.agent.db_loss = tf.reduce_mean(self.dynamic_bottleneck.loss)

        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.dynamic_bottleneck.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        # ob_space.shape=(84, 84, 4)     ac_space.shape=Discrete(4)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self, saver, logger_dir):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamic_bottleneck=self.dynamic_bottleneck)
        previous_saved_tcount = 0

        # add bai. initialize IB parameters
        print("***Init Momentum Network in Dynamic-Bottleneck.")
        getsess().run(self.dynamic_bottleneck.init_updates)

        while True:
            info = self.agent.step()         # 
            if info['DB_loss_info']:         # add bai. for debug
                logger.logkvs(info['DB_loss_info'])
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.hps["save_period"] and (int(self.agent.rollout.stats['tcount'] / self.hps["save_freq"]) > previous_saved_tcount):
                previous_saved_tcount += 1
                save_path = saver.save(tf.get_default_session(), os.path.join(logger_dir, "model_"+str(previous_saved_tcount)+".ckpt"))
                print("Periodically model saved in path:", save_path)
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                save_path = saver.save(tf.get_default_session(), os.path.join(logger_dir, "model_last.ckpt"))
                print("Model saved in path:", save_path)
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        if args["stickyAtari"]:               # 
            env._max_episode_steps = args['max_episode_steps'] * 4
            env = StickyActionEnv(env)
        else:
            env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)            # 
        if args['pixelNoise']:                      # add pixel noise
            env = PixelNoiseWrapper(env)
        if args['randomBoxNoise']:
            env = RandomBoxNoiseWrapper(env)
        env = ProcessFrame84(env, crop=False)       #
        env = FrameStack(env, 4)                    #
        # env = ExtraTimeLimit(env, args['max_episode_steps'])
        if not args["stickyAtari"]:
            env = ExtraTimeLimit(env, args['max_episode_steps'])  # 
        if 'Montezuma' in args['env']:              # 
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':               # 
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":         # 
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    # log dir name
    logger_dir = './logs/' + args["env"].replace("NoFrameskip-v4", "")
    # logger_dir += "-KLloss-"+str(args["loss_kl_weight"])     
    # logger_dir += "-NCEloss-" + str(args["loss_nce_weight"]) 
    # logger_dir += "-L2loss-" + str(args["loss_l2_weight"])
    if args['pixelNoise'] is True:
        logger_dir += "-pixelNoise"
    if args['randomBoxNoise'] is True:
        logger_dir += "-randomBoxNoise"
    if args['stickyAtari'] is True:
        logger_dir += "-stickyAtari"
    if args["comments"] != "":
        logger_dir += '-' + args["comments"]
    logger_dir += datetime.datetime.now().strftime("-%m-%d-%H-%M-%S")

    # write config
    logger.configure(dir=logger_dir)
    with open(os.path.join(logger_dir, 'parameters.txt'), 'w') as f:
        f.write("\n".join([str(x[0]) + ": " + str(x[1]) for x in args.items()]))

    logger_context = logger.scoped_configure(
        dir=logger_dir,
        format_strs=['stdout', 'log', 'csv'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()

    # saver 
    saver = tf.train.Saver()
    return logger_context, tf_context, saver, logger_dir


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4', type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)
    parser.add_argument('--stickyAtari', action='store_true', default=False)
    parser.add_argument('--pixelNoise', action='store_true', default=False)
    parser.add_argument('--randomBoxNoise', action='store_true', default=False)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)                  # lambda, gamma 用于计算 GAE advantage
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)                    # 
    parser.add_argument('--norm_rew', type=int, default=1)                    # 
    parser.add_argument('--lr', type=float, default=1e-4)                     # 
    parser.add_argument('--ent_coeff', type=float, default=0.001)             # 
    parser.add_argument('--nepochs', type=int, default=3)                     # 
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))
    parser.add_argument('--save_period', action='store_true', default=False)  # 1e7
    parser.add_argument('--save_freq', type=int, default=int(1e7))            # 1e7
    # Parameters of Dynamic-Bottleneck
    parser.add_argument('--loss_kl_weight', type=float, default=0.1)          # KL loss weight
    parser.add_argument('--loss_l2_weight', type=float, default=0.1)        # l2 loss weight
    parser.add_argument('--loss_nce_weight', type=float, default=0.01)         # nce loss weight
    parser.add_argument('--momentum_tau', type=float, default=0.001)          # momentum tau
    parser.add_argument('--aug', action='store_true', default=False)          # data augmentation (bottleneck)
    parser.add_argument('--comments', type=str, default="")


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=128)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)     
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)

    args = parser.parse_args()

    # load paramets
    with open("para.json") as f:
        d = json.load(f) 
    env_name_para = args.env.replace("NoFrameskip-v4", "")
    if env_name_para not in list(d["standard"].keys()):
        env_name_para = "other"
    
    if args.pixelNoise is True:
        print("pixel noise")
        args.loss_kl_weight = d["pixelNoise"][env_name_para]["kl"]
        args.loss_nce_weight = d["pixelNoise"][env_name_para]["nce"]
    elif args.randomBoxNoise is True:
        print("random box noise")
        args.loss_kl_weight = d["randomBox"][env_name_para]["kl"]
        args.loss_nce_weight = d["randomBox"][env_name_para]["nce"]
    elif args.stickyAtari is True:
        print("sticky noise")
        args.loss_kl_weight = d["stickyAtari"][env_name_para]["kl"]
        args.loss_nce_weight = d["stickyAtari"][env_name_para]["nce"]
    else:
        print("standard atari")
        args.loss_kl_weight = d["standard"][env_name_para]["kl"]
        args.loss_nce_weight = d["standard"][env_name_para]["nce"]

    print("env_name:", env_name_para, "kl:", args.loss_kl_weight, ", nce:", args.loss_nce_weight)
    start_experiment(**args.__dict__)

