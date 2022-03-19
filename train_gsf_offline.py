import os
from collections import deque

import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from absl import app, flags
from flax.training.train_state import TrainState
from flax.training import checkpoints
from jax.random import PRNGKey

from algo import get_transition, select_action_critic, update, update_gsf, state_update, update_curl_jit, update_cluster_jit, random_crop
from buffer import Batch
from models import CriticCURL, CriticCTRL
from vec_env import ProcgenVecEnvCustom
import glob
import os

def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def load_shard(path, id):
    obs = np.load(os.path.join(path, 'obs_%d.npy' % id),
                  allow_pickle=False,
                  fix_imports=True)
    action = np.load(os.path.join(path, 'action_%d.npy' % id),
                     allow_pickle=False,
                     fix_imports=True)
    reward = np.load(os.path.join(path, 'reward_%d.npy' % id),
                     allow_pickle=False,
                     fix_imports=True)
    done = np.load(os.path.join(path, 'done_%d.npy' % id),
                   allow_pickle=False,
                   fix_imports=True)
    return obs, action, reward, done


FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("train_steps", 1_000_000, "Number of train frames.")
# CQL
flags.DEFINE_float("gamma", 0.99, "Gamma")
flags.DEFINE_integer("batch_size", 100, "Batch size.")
flags.DEFINE_float("lr", 3e-4, "PPO learning rate")
flags.DEFINE_float("cql_reg", 1, "CQL loss multiplier")
flags.DEFINE_float("tau_ema", 0.005, "EMA smoothing")
# Logging
flags.DEFINE_string("run_id", "gsf_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_string("wandb_mode", "disabled",
                    "W&B logging (disabled, online, offline)")
flags.DEFINE_string("wandb_entity", "ssl_rl",
                    "W&B entity (username or team name)")
flags.DEFINE_string("wandb_project", "jax_gsf_offline", "W&B project name")


def main(argv):
    # Report unclipped rewards for test
    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                      num_levels=200,
                                      mode='easy',
                                      start_level=0,
                                      paint_vel_info=False,
                                      num_envs=64,
                                      normalize_rewards=False)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                       num_levels=0,
                                       mode='easy',
                                       start_level=0,
                                       paint_vel_info=True,
                                       num_envs=64,
                                       normalize_rewards=False)

    group_name = "%s_%s" % (FLAGS.env_name, FLAGS.run_id)
    name = "%s_%s_%d" % (FLAGS.env_name, FLAGS.run_id,
                         np.random.randint(100000000))

    wandb.init(project=FLAGS.wandb_project,
               entity=FLAGS.wandb_entity,
               config=FLAGS,
               group=group_name,
               name=name,
               sync_tensorboard=False,
               mode=FLAGS.wandb_mode)

    np.random.seed(FLAGS.seed)
    key = PRNGKey(FLAGS.seed)

    if FLAGS.curl_reg > 0:
        model = CriticCURL(dims=[256, 256, 15])
        target = CriticCURL(dims=[256, 256, 15])
        fake_args_model = jnp.zeros((1, 68, 68, 3))
        params_model = model.init(key, fake_args_model)
        params_target = target.init(key, fake_args_model)
    elif FLAGS.ctrl_reg > 0:
        model = CriticCTRL(dims=(256, 256),
                           n_cluster=FLAGS.num_clusters,
                           n_actions=env_test_ID.action_space.n,
                           n_att_heads=FLAGS.n_att_heads,
                           embedding_type=FLAGS.embedding_type,
                           prefix_critic='vfunction',
                           prefix_actor="policy")
        target = CriticCTRL(dims=(256, 256),
                            n_cluster=FLAGS.num_clusters,
                            n_actions=env_test_ID.action_space.n,
                            n_att_heads=FLAGS.n_att_heads,
                            embedding_type=FLAGS.embedding_type,
                            prefix_critic='vfunction',
                            prefix_actor="policy")
        fake_args_cluster = (jnp.zeros((1, FLAGS.cluster_len, 68, 68, 3)),
                             jnp.zeros((1, FLAGS.cluster_len)))
        params_model = model.init(key,
                                  state=fake_args_cluster[0],
                                  action=fake_args_cluster[1],
                                  reward=fake_args_cluster[1])
        params_target = target.init(key,
                                    state=fake_args_cluster[0],
                                    action=fake_args_cluster[1],
                                    reward=fake_args_cluster[1])

    tx = optax.chain(optax.adam(FLAGS.lr, eps=1e-5))
    tx_target = optax.chain(optax.adam(FLAGS.lr, eps=1e-5))

    train_state = TrainState.create(apply_fn=model.apply,
                                    params=params_model,
                                    tx=tx)
    train_state_target = TrainState.create(apply_fn=target.apply,
                                           params=params_target,
                                           tx=tx_target)

    train_state_target = state_update(train_state,
                                      train_state_target,
                                      key='',
                                      tau=1.)

    state_id = env_test_ID.reset()
    state_ood = env_test_OOD.reset()

    epinfo_buf_id = deque(maxlen=100)
    epinfo_buf_ood = deque(maxlen=100)

    model_shards = list(
        map(
            lambda x: int(x.split('action_')[-1].split('.')[0]),
            sorted(
                glob.glob('/PATH/offline_procgen/%s/action_*.npy' %
                          FLAGS.env_name))))
    steps_in_shard = 0
    shard_id = np.random.choice(model_shards, 1).item()

    for step in tqdm.tqdm(range(FLAGS.train_steps)):
        if steps_in_shard == 0:
            shard_id = np.random.choice(model_shards, 1).item()
            batch = load_shard(
                '/PATH/offline_procgen/%s' % FLAGS.env_name,
                shard_id)
            steps_in_shard = len(batch[0])
            idxes = np.random.uniform(size=(len(batch[0]) - 1, )).argsort()
            obs_b, action_b, reward_b, done_b = batch
            obs_t, obs_tp1 = obs_b[:-1], obs_b[1:]
            action_t, action_tp1 = action_b[:-1], action_b[1:]
            reward_t, reward_tp1 = reward_b[:-1], reward_b[1:]
            done_t, done_tp1 = done_b[:-1], done_b[1:]
            print('Loaded shard %d' % shard_id)

        steps_in_shard -= FLAGS.batch_size
        obs, action, reward, next_obs, next_action, done = obs_t[
            idxes[steps_in_shard:steps_in_shard + FLAGS.batch_size]], action_t[
                idxes[steps_in_shard:steps_in_shard +
                      FLAGS.batch_size]], reward_t[
                          idxes[steps_in_shard:steps_in_shard +
                                FLAGS.batch_size]], obs_tp1[idxes[
                                    steps_in_shard:steps_in_shard +
                                    FLAGS.batch_size]], action_tp1[idxes[
                                        steps_in_shard:steps_in_shard +
                                        FLAGS.batch_size]], done_t[idxes[
                                            steps_in_shard:steps_in_shard +
                                            FLAGS.batch_size]]

        action_id, _, _, key = select_action_critic(train_state,
                                                    state_id,
                                                    key,
                                                    sample=True,
                                                    policy_fn=model.ac)
        state_id, _, _, infos_id = env_test_ID.step(action_id)

        action_ood, _, _, key = select_action_critic(train_state,
                                                     state_ood,
                                                     key,
                                                     sample=True,
                                                     policy_fn=model.ac)
        state_ood, _, _, infos_ood = env_test_OOD.step(action_ood)

        for info in infos_id:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_id.append(maybe_epinfo)

        for info in infos_ood:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_ood.append(maybe_epinfo)

        obs_1, obs_2 = random_crop(obs, n_augs=2)
        next_obs = random_crop(next_obs, n_augs=1)[0]
        
        # Use CURL/ PSE/ GSF here
        # update_X_jit(
        #     train_state, train_state_target, obs_1, obs_2, action, reward,
        #     next_obs, next_action, done, FLAGS.gamma, FLAGS.cql_reg,
        #     FLAGS.curl_reg, key)

        train_state_target = state_update(train_state,
                                          train_state_target,
                                          key='',
                                          tau=FLAGS.tau_ema)

        if (step % 1000) == 0:
            renamed_dict = {
                "%s/rl_loss" % FLAGS.env_name: critic_loss,
                "%s/cql_loss" % FLAGS.env_name: cql_loss,
            }

            wandb.log(renamed_dict, step=step)

            wandb.log(
                {
                    "%s/ep_return_200" % (FLAGS.env_name):
                    safe_mean([info['r'] for info in epinfo_buf_id])
                },
                step=step)
            wandb.log(
                {
                    "%s/ep_return_all" % (FLAGS.env_name):
                    safe_mean([info['r'] for info in epinfo_buf_ood])
                },
                step=step)


if __name__ == '__main__':
    app.run(main)
    quit()