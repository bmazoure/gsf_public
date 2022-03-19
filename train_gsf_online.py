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

from algo import get_transition, select_action, update_ppo, update_ppo_and_gsf, update_ppo_and_pse, state_update, safe_mean, evaluate_policy
from buffer import Batch
from models import PPOModel, GSFPPOModel
from vec_env import ProcgenVecEnvCustom
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "bigfish", "Env name")
flags.DEFINE_integer("seed", 1, "Random seed.")
flags.DEFINE_integer("num_envs", 64, "Num of Procgen envs.")
flags.DEFINE_integer("train_steps", 25_000_000, "Number of train frames.")
# PPO
flags.DEFINE_float("max_grad_norm", 0.5, "Max grad norm")
flags.DEFINE_float("gamma", 0.999, "Gamma")
flags.DEFINE_integer("n_steps", 256, "GAE n-steps")
flags.DEFINE_integer("n_minibatch", 8, "Number of PPO minibatches")
flags.DEFINE_float("lr", 5e-4, "PPO learning rate")
flags.DEFINE_integer("epoch_ppo", 3, "Number of PPO epochs on a single batch")
flags.DEFINE_float("clip_eps", 0.2, "Clipping range")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda")
flags.DEFINE_float("entropy_coeff", 0.01, "Entropy loss coefficient")
flags.DEFINE_float("critic_coeff", 0.5, "Value loss coefficient")
# GSF
flags.DEFINE_integer("n_tasks_nce", 4, "Number of PPO values to load")
flags.DEFINE_integer("n_quantiles", 10, "Number of quantiles")
flags.DEFINE_float("temp", 0.1, "Temperature")
flags.DEFINE_float("gsf_coeff", 1.0, "GSF loss coefficient")
flags.DEFINE_string("gsf_type", "value", "Pre-trained choice of GSF")
# Logging
flags.DEFINE_integer("checkpoint_interval", 999424, "Checkpoint frequency (about 1M)")
flags.DEFINE_string("model_dir", "model_weights", "Model weights dir")
flags.DEFINE_string("run_id", "gsf_ppo",
                    "Run ID. Change that to change W&B name")
flags.DEFINE_string("wandb_mode", "disabled",
                    "W&B logging (disabled, online, offline)")
flags.DEFINE_string("wandb_entity", "ssl_rl", "W&B entity (username or team name)")
flags.DEFINE_string("wandb_project", "jax_gsf", "W&B project name")


def main(argv):
    if FLAGS.seed == -1:		
        seed = np.random.randint(100000000)		
    else:		
        seed = FLAGS.seed		
    np.random.seed(seed)		
    key = PRNGKey(seed)
    # Clip rewards for training
    env = ProcgenVecEnvCustom(FLAGS.env_name,
                              num_levels=200,
                              mode='easy',
                              start_level=0,
                              paint_vel_info=False,
                              num_envs=FLAGS.num_envs,
                              normalize_rewards=True)
    # Report unclipped rewards for test
    env_test_ID = ProcgenVecEnvCustom(FLAGS.env_name,
                                      num_levels=200,
                                      mode='easy',
                                      start_level=0,
                                      paint_vel_info=False,
                                      num_envs=FLAGS.num_envs,
                                      normalize_rewards=False)
    env_test_OOD = ProcgenVecEnvCustom(FLAGS.env_name,
                                       num_levels=0,
                                       mode='easy',
                                       start_level=0,
                                       paint_vel_info=False,
                                       num_envs=FLAGS.num_envs,
                                       normalize_rewards=False)

    group_name = "%s_%s" % (FLAGS.env_name, FLAGS.run_id)
    name = "%s_%s_%d" % (FLAGS.env_name, FLAGS.run_id, np.random.randint(100000000))

    wandb.init(project=FLAGS.wandb_project,
               entity=FLAGS.wandb_entity,
               config=FLAGS,
               group=group_name,
               name=name,
               sync_tensorboard=False,
               mode=FLAGS.wandb_mode)

    model = GSFPPOModel(action_dim=env.action_space.n,
                n_quantiles=FLAGS.n_quantiles,
                temp=FLAGS.temp)
    fake_args_model = jnp.zeros((1, 64, 64, 3))
    params_model = model.init(key, fake_args_model)

    tx = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_grad_norm),
        optax.adam(FLAGS.lr, eps=1e-5)
    )
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params_model,
        tx=tx)

    batch = Batch(
            discount=FLAGS.gamma,
            gae_lambda=FLAGS.gae_lambda,
            n_steps=FLAGS.n_steps+1,
            num_envs=FLAGS.num_envs,
            state_space=env.observation_space
        )
        
    state = env.reset()
    state_id = env_test_ID.reset()
    state_ood = env_test_OOD.reset()

    epinfo_buf_id = deque(maxlen=100)
    epinfo_buf_ood = deque(maxlen=100)

    """
    Load n_tasks pre-trained PPO policies/value functions
    """
    batches_nce = []
    train_states_nce = []
    states_nce = []
    envs_nce = []
    model_dir = os.path.join(FLAGS.model_dir, FLAGS.env_name)
    model_ckpts = glob.glob('/scratch/bmazoure/offline_procgen_ppo2/%s/**/'%FLAGS.env_name)
    model_files = np.random.choice(model_ckpts, size=FLAGS.n_tasks_nce)
    for m in range(FLAGS.n_tasks_nce):
        ckpt_m = checkpoints.latest_checkpoint(model_files[m])
        if ckpt_m is None:
            print('Skipping GVF')
            continue
        model_m = PPOModel(action_dim=env.action_space.n,
                          prefix_critic='vfunction',
                          prefix_actor="policy")
        params_model_m = model_m.init(key, fake_args_model)
        tx = optax.chain(
            optax.clip_by_global_norm(FLAGS.max_grad_norm),
            optax.adam(FLAGS.lr, eps=1e-5)
            )
        train_state_m = TrainState.create(
            apply_fn=model_m.apply,
            params=params_model_m,
            tx=tx)
        loaded_state_m = checkpoints.restore_checkpoint(ckpt_m, target=train_state_m)
        train_states_nce.append(loaded_state_m)
        batch_m = Batch(
            discount=FLAGS.gamma,
            gae_lambda=FLAGS.gae_lambda,
            n_steps=FLAGS.n_steps+1,
            num_envs=FLAGS.num_envs//FLAGS.n_tasks_nce,
            state_space=env.observation_space
        )
        batches_nce.append(batch_m)
        level_m = int(model_files[m].split('/')[-2])
        env_m =  ProcgenVecEnvCustom(FLAGS.env_name,
                              num_levels=1,
                              mode='easy',
                              start_level=level_m,
                              paint_vel_info=False,
                              num_envs=FLAGS.num_envs//FLAGS.n_tasks_nce,
                              normalize_rewards=False)
        state_m = env_m.reset()
        states_nce.append(state_m)
        envs_nce.append(env_m)

        returns_m = evaluate_policy(env_m, loaded_state_m, n_episodes=10)
        print('[level %d] Returns: %f'%(level_m, returns_m))

    for step in range(1, int(FLAGS.train_steps // FLAGS.num_envs + 1)):
        train_state, state, batch, key = get_transition(train_state, env, state, batch, key)

        for m in range(len(train_states_nce)):
            train_states_nce[m], states_nce[m], batches_nce[m], key = get_transition(train_state, envs_nce[m], states_nce[m], batches_nce[m], key, gvf_state=train_states_nce[m])

        action_id, _, _, key = select_action(train_state, state_id.astype(jnp.float32) / 255., key, sample=True)
        state_id, _, _, infos_id = env_test_ID.step(action_id)

        action_ood, _, _, key = select_action(train_state, state_ood.astype(jnp.float32) / 255., key, sample=True)
        state_ood, _, _, infos_ood = env_test_OOD.step(action_ood)

        for info in infos_id:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_id.append(maybe_epinfo)

        for info in infos_ood:
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf_ood.append(maybe_epinfo)

        if step % (FLAGS.n_steps + 1) == 0:
            data = batch.get()
            gvf_nce, obs_nce = [], []
            for b in batches_nce:
                data_m = b.get()
                if FLAGS.gsf_type == 'action':
                    gvf_m = data_m[1]
                elif FLAGS.gsf_type == 'value':
                    gvf_m = data_m[4]
                elif FLAGS.gsf_type == 'mc_return':
                    gvf_m = data_m[5]
                elif FLAGS.gsf_type == 'gae':
                    gvf_m = data_m[6]
                # MuZero transform
                # h=lambda x:jnp.sign(x)*(jnp.sqrt(jnp.abs(x)+1)-1+1e-3*x)
                # gvf_m = h(gvf_m)
                gvf_nce.append(gvf_m)
                obs_nce.append(data_m[0]) # save observations
            gvf_nce = jnp.stack(gvf_nce)
            obs_nce = jnp.stack(obs_nce)
            metric_dict, train_state, key = update_ppo_and_gsf(train_state,
                                            data,
                                            gvf_nce,
                                            obs_nce,
                                            FLAGS.num_envs,
                                            FLAGS.n_steps,
                                            FLAGS.n_minibatch,
                                            FLAGS.epoch_ppo,
                                            FLAGS.n_quantiles,
                                            FLAGS.temp,
                                            FLAGS.clip_eps,
                                            FLAGS.entropy_coeff,
                                            FLAGS.critic_coeff,
                                            FLAGS.gsf_coeff,
                                            key)

            renamed_dict = {}
            for k,v in metric_dict.items():
                renamed_dict["%s/%s"%(FLAGS.env_name,k)] = v
            wandb.log(
                renamed_dict,
                step=FLAGS.num_envs * step)

            batch.reset()
            for b in batches_nce:
                b.reset()

            renamed_dict = {}
            for k,v in metric_dict.items():
                renamed_dict["%s/%s"%(FLAGS.env_name,k)] = v
            wandb.log(
                renamed_dict,
                step=FLAGS.num_envs * step)

            wandb.log({
                "%s/ep_return_200" % (FLAGS.env_name): safe_mean([info['r'] for info in epinfo_buf_id]),
                "step": FLAGS.num_envs * step
            })
            wandb.log({
                "%s/ep_return_all" % (FLAGS.env_name): safe_mean([info['r'] for info in epinfo_buf_ood]),
                "step": FLAGS.num_envs * step
            })
            print('Eprew all: %f'% safe_mean([info['r'] for info in epinfo_buf_ood]))
            print(metric_dict['gsf_loss'])

if __name__ == '__main__':
    app.run(main)
    quit()