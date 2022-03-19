import os
from collections import deque
from functools import partial

import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from flax.training.train_state import TrainState
from flax.training import checkpoints
from jax.random import PRNGKey

from models import TwinHeadModel
from vec_env import ProcgenVecEnvCustom
import jax

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("num_envs", 100, "Num of Procgen envs.")
# PPO
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
# Dataset
flags.DEFINE_integer("timesteps", int(100e3), "Number of transitions in dataset")
flags.DEFINE_integer("shards", 10, "Number of ways to split the dataset")
flags.DEFINE_string("dataset_dir", "/scratch/bmazoure/offline_procgen", "Dataset output dir")
flags.DEFINE_string("obs_type", "rgb", "Type of obs (rgb or ram)")
# Logging
flags.DEFINE_string("model_dir", "model_weights", "Model weights dir")

def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)

def main(argv):
    # Report unclipped rewards for test
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                                        num_levels=200,
                                        mode='easy',
                                        start_level=0,
                                        paint_vel_info=False,
                                        num_envs=FLAGS.num_envs,
                                        normalize_rewards=False)
                                        
    np.random.seed(FLAGS.seed)
    rng = PRNGKey(FLAGS.seed)

    model = TwinHeadModel(action_dim=env.action_space.n,
                            prefix_critic='vfunction',
                            prefix_actor="policy")
    fake_args_model = jnp.zeros((1, 64, 64, 3))
    params_model = model.init(rng, fake_args_model)
    
    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_grad_norm),
        optax.adam(FLAGS.lr, eps=1e-5)
    )
    
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params_model,
        tx=tx)
        
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.env_name)
    loaded_state = checkpoints.restore_checkpoint('./%s'%model_dir,target=train_state)

    def linear_scheduling(t):  # pylint: disable=unused-variable
        return 0.1 - 3.96e-9 * t

    @partial(jax.jit, static_argnums=0)
    def policy(apply_fn, params, state):
        value, logits = apply_fn(params, state)
        return value, logits

    def select_action(train_state, state, rng, t, greedy=False):
        state = state.astype(jnp.float32) / 255.
        value, pi = policy(train_state.apply_fn, train_state.params, state)
        rng, key = jax.random.split(rng)
        if greedy:
            action = pi.mode()
        else:
            eps = linear_scheduling(t)
            u = jax.random.uniform(key=key).item()
            rng, key = jax.random.split(rng)
            if 0 < eps < u:
                action = jax.random.randint(key=key,minval=0, maxval=env.action_space.n, shape=(len(state),))
            else:
                action = pi.sample(seed=key)
        return action, value[:, 0], rng

    epinfo_buf = deque(maxlen=100)

    state = env.reset()
    ram_state = [repr(x) for x in env.env.env.callmethod("get_state")]
    done = np.array([False]*FLAGS.num_envs)
    obs_acc, action_acc, reward_acc, done_acc = [], [], [], []
    dataset_path = os.path.join(FLAGS.dataset_dir, FLAGS.env_name)
    if not os.path.isdir(dataset_path):
        os.makedirs(dataset_path)

    # env.env.env.callmethod("set_state",[eval(x) for x in np.array(ram_acc)[0]])
    # ram_ep, action_ep, reward_ep = [], [], []
    shard_size = FLAGS.timesteps // FLAGS.shards
    shard_count = 0
    for t in range(FLAGS.num_envs, FLAGS.timesteps, FLAGS.num_envs):
        action, value, rng = select_action(loaded_state, state, rng, t, greedy=False)
        if FLAGS.obs_type == 'ram':
            obs_acc.append(ram_state)
        elif FLAGS.obs_type == 'rgb':
            obs_acc.append(state)
        
        action_acc.append(action.copy())

        state, reward, _, epinfo = env.step(action)
        reward_acc.append(reward.copy())
        done_acc.append(done.copy())

        ram_state = [repr(x) for x in env.env.env.callmethod("get_state")]
        
        for i, info in enumerate(epinfo):
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf.append(maybe_epinfo)
                done[i] = True
        if t > 0 and t % shard_size == 0:
            # column-by-column flatten to conserve time dependence
            if FLAGS.obs_type == 'ram':
                obs_acc = np.array(obs_acc).reshape(-1,order="F")
            elif FLAGS.obs_type == 'rgb':
                obs_acc = np.array(obs_acc).reshape((-1,64,64,3),order="F")
            action_acc = np.array(action_acc).reshape(-1,order="F")
            reward_acc = np.array(reward_acc).reshape(-1,order="F")
            done_acc = np.array(done_acc).reshape(-1,order="F")
            with open(os.path.join(dataset_path, "obs_%d.npy"%shard_count),"wb") as fh:
                np.save(fh, obs_acc, allow_pickle=False, fix_imports=True)
            with open(os.path.join(dataset_path, "action_%d.npy"%shard_count),"wb") as fh:
                np.save(fh, action_acc, allow_pickle=False, fix_imports=True)
            with open(os.path.join(dataset_path, "reward_%d.npy"%shard_count),"wb") as fh:
                np.save(fh, reward_acc, allow_pickle=False, fix_imports=True)
            with open(os.path.join(dataset_path, "done_%d.npy"%shard_count),"wb") as fh:
                np.save(fh, done_acc, allow_pickle=False, fix_imports=True)
            print("Saved shard %d"%(shard_count))
            shard_count += 1
            obs_acc, action_acc, reward_acc, done_acc = [], [], [], []
            
    returns = safe_mean([info['r'] for info in epinfo_buf])

    print('Returns [%s]: %.3f'% (FLAGS.env_name,returns))

if __name__ == '__main__':
    app.run(main)
    quit()