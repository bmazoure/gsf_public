from collections import defaultdict
from functools import partial
from typing import Any, Callable, Tuple, List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState
from jax.random import PRNGKey

from tensorflow_probability.substrates import jax as tfp

import dm_pix as pix

tfd = tfp.distributions
tfb = tfp.bijectors
"""
Inspired by code from Flax: https://github.com/google/flax/blob/main/examples/ppo/ppo_lib.py
"""

def calculate_action_cost_matrix(ac1, ac2):
  diff = jnp.expand_dims(ac1, axis=1).astype(jnp.float32) - jnp.expand_dims(ac2, axis=0).astype(jnp.float32)
  return jnp.abs(diff).mean(axis=-1)


def metric_fixed_point_fast(cost_matrix, gamma=0.99, eps=1e-7):
    """Dynamic prograaming for calculating PSM."""
    d = jnp.zeros_like(cost_matrix)

    def operator(d_cur, i):
        d_new = 1 * cost_matrix
        discounted_d_cur = gamma * d_cur
        d_new = d_new.at[:-1, :-1].set(d_new[:-1, :-1]+discounted_d_cur[1:, 1:])
        d_new = d_new.at[:-1, -1].set(d_new[:-1, -1]+discounted_d_cur[1:, -1])
        d_new = d_new.at[-1, :-1].set(d_new[-1, :-1]+discounted_d_cur[-1, 1:])
        return d_new, d_new

    d = jax.lax.scan(operator, d, jnp.arange(20))[0]
    return d


def safe_mean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def evaluate_policy(env, train_state, n_episodes=10):
    rng = PRNGKey(123)
    state = env.reset()
    epinfo_buf = []
    while n_episodes > 0:
        action, log_prob, value, rng = select_action(train_state, state.astype(jnp.float32) / 255., rng, sample=True)

        state, reward, _, epinfo = env.step(action)
        
        for i, info in enumerate(epinfo):
            maybe_epinfo = info.get('episode')
            if maybe_epinfo:
                epinfo_buf.append(maybe_epinfo)
                n_episodes -= 1
            
    returns = safe_mean([info['r'] for info in epinfo_buf])

    return returns


def compute_distance(A):
    similarity = jnp.dot(A, A.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = jnp.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    # inv_square_mag[jnp.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = jnp.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag


def cosine_similarity(A1, A2):
    similarity = jnp.dot(A1, A2.T)
    # squared magnitude of preference vectors (number of occurrences)
    square_mag = jnp.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    # inv_square_mag[jnp.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = jnp.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    return cosine.T * inv_mag


def state_update(online_state, target_state, key: str = '', tau: float = 1.):
    """ Update key weights as tau * online + (1-tau) * target
    """
    if key == '':
        p_o = online_state.params['params']
        p_t = target_state.params['params']
    else:
        p_o = online_state.params['params'][key]
        p_t = target_state.params['params'][key]
    new_weights = target_update(p_o, p_t, tau)
    if key != '':
        new_weights = target_state.params['params'].copy(
            add_or_replace={key: new_weights})
    new_params = target_state.params.copy(
        add_or_replace={'params': new_weights})

    target_state = target_state.replace(params=new_params)
    return target_state


def target_update(online, target, tau: float):
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), online, target)

    return new_target_params


def l2_normalize(A, axis=-1, eps=1e-8):
    return A * jax.lax.rsqrt((A * A).sum(axis=axis, keepdims=True) + eps)


def group_by(data, var):
    x, g = var
    x_grouped, group_cnts = data
    # append entries into specified group
    x_grouped = jax.ops.index_add(x_grouped, (g, group_cnts[g]), x)
    # track how many entries appended into each group
    group_cnts = jax.ops.index_add(group_cnts, g, 1)
    return (x_grouped, group_cnts), 0  # '0' is just a dummy value


def cos_loss(p, z):
    z = jax.lax.stop_gradient(z)
    p = l2_normalize(p, axis=1)
    z = l2_normalize(z, axis=1)
    dist = 2 - 2 * jnp.sum(p * z, axis=1)
    return dist


def loss_actor_and_critic(params_model: flax.core.frozen_dict.FrozenDict,
                          apply_fn: Callable[..., Any], state: jnp.ndarray,
                          target: jnp.ndarray, value_old: jnp.ndarray,
                          log_pi_old: jnp.ndarray, gae: jnp.ndarray,
                          action: jnp.ndarray, clip_eps: float,
                          critic_coeff: float,
                          entropy_coeff: float) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.

    (value_pred, pi), _ = apply_fn(params_model, state)
    value_pred = value_pred[:, 0]

    log_prob = pi.log_prob(action[:, 0])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy

    return total_loss, (value_loss, loss_actor, entropy, value_pred.mean(),
                        target.mean(), gae.mean())


def loss_actor_and_critic_and_gsf(params_model: flax.core.frozen_dict.FrozenDict,
                          apply_fn: Callable[..., Any], state: jnp.ndarray,
                          target: jnp.ndarray, value_old: jnp.ndarray,
                          log_pi_old: jnp.ndarray, gae: jnp.ndarray,
                          action: jnp.ndarray, gvf_nce: jnp.ndarray, obs_nce: jnp.ndarray, n_quantiles: int, clip_eps: float,
                          critic_coeff: float,
                          entropy_coeff: float,
                          gsf_coeff: float, 
                          rng: PRNGKey) -> jnp.ndarray:
    state = state.astype(jnp.float32) / 255.

    (value_pred, pi), _ = apply_fn(params_model, state)
    value_pred = value_pred[:, 0]

    log_prob = pi.log_prob(action[:, 0])

    value_pred_clipped = value_old + (value_pred - value_old).clip(
        -clip_eps, clip_eps)
    value_losses = jnp.square(value_pred - target)
    value_losses_clipped = jnp.square(value_pred_clipped - target)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    ratio = jnp.exp(log_prob - log_pi_old)
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
    loss_actor = loss_actor.mean()

    entropy = pi.entropy().mean()

    # GSF
    if gsf_coeff > 0.:
        aug_state = jnp.pad(obs_nce,[[0,0],[2,2],[2,2],[0,0]], 'edge')
        aug_state = pix.random_crop(key=rng, image=aug_state, crop_sizes=state.shape)
        aug_state = aug_state.astype(jnp.float32) / 255.

        if n_quantiles == 1:
            # MSE
            _, logits = apply_fn(params_model, aug_state, regression=True)
            gsf_loss = jnp.square(logits[:,0]-gvf_nce).mean()
        else:
            _, logits = apply_fn(params_model, aug_state, regression=False)
            quantile_edges = jnp.quantile(gvf_nce , # + eps
                                        q=jnp.linspace(0, 1, n_quantiles),
                                        axis=0)
            labels = tfp.stats.find_bins(gvf_nce, # + eps
                                        quantile_edges).transpose().reshape(-1, 1)
            gsf_loss = -jnp.take_along_axis(logits, labels, 1)[:, 0].mean()
    else:
        gsf_loss = 0.

    total_loss = loss_actor + critic_coeff * value_loss - entropy_coeff * entropy + gsf_coeff * gsf_loss

    return total_loss, (value_loss, loss_actor, entropy, value_pred.mean(),
                        target.mean(), gae.mean(), gsf_loss)


@partial(jax.jit, static_argnames=("sample"))
def select_action(
    train_state: TrainState,
    state: jnp.ndarray,
    rng: PRNGKey,
    sample: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKey]:
    (value, pi), _ = train_state.apply_fn(train_state.params, state)

    if sample:
        rng, key = jax.random.split(rng)
        action = pi.sample(seed=key)
    else:
        action = pi.mode()

    log_prob = pi.log_prob(action)
    return action, log_prob, value[:, 0], rng


@partial(jax.jit, static_argnames=("policy_fn", "sample"))
def select_action_critic(
    train_state: TrainState,
    state: np.ndarray,
    rng: PRNGKey,
    policy_fn=None,
    sample: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, PRNGKey]:
    paddings = jnp.array([[0, 0], [2, 2], [2, 2], [0, 0]], dtype=jnp.int32)
    state = jnp.pad(state.astype(jnp.int32), paddings, 'symmetric').astype(
        jnp.float32) / 255.
    (q1, q2), _ = train_state.apply_fn(train_state.params,
                                       state,
                                       method=policy_fn)
    q = jnp.minimum(q1, q2)
    if sample:
        rng, key = jax.random.split(rng)
        action = jax.random.categorical(key=key, logits=q)
    else:
        action = q.argmax(-1)

    return action, q1[:, 0], q2[:, 0], rng


def get_transition(
    train_state: TrainState,
    env,
    state,
    batch,
    rng: PRNGKey,
    gvf_state=None
):
    action, log_pi, value, new_key = select_action(train_state,
                                                   state.astype(jnp.float32) /
                                                   255.,
                                                   rng,
                                                   sample=True)
    if gvf_state is not None:
        _, _, value, new_key = select_action(gvf_state,
                                                   state.astype(jnp.float32) /
                                                   255.,
                                                   rng,
                                                   sample=True)
    next_state, reward, done, infos = env.step(action)
    level_seeds = [info['level_seed'] for info in infos]
    batch.append(state, action, reward, done, np.array(log_pi),
                 np.array(value), level_seeds)
    return train_state, next_state, batch, new_key


@partial(jax.jit)
def flatten_dims(x):
    return x.swapaxes(0, 1).reshape(x.shape[0] * x.shape[1], *x.shape[2:])


@partial(jax.jit,
         static_argnames=("num_envs", "n_steps", "n_minibatch", "epoch_ppo", "clip_eps", "entropy_coeff", "critic_coeff"))
def update_ppo(train_state: TrainState, batch: Tuple, num_envs: int,
               n_steps: int, n_minibatch: int, epoch_ppo: int, clip_eps: float,
               entropy_coeff: float, critic_coeff: float, rng: PRNGKey):

    state, action, reward, log_pi_old, value, target, gae, task_ids = batch

    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(num_envs * n_steps)
    idxes_policy = []
    for _ in range(epoch_ppo):
        rng, key = jax.random.split(rng)
        idxes = jax.random.permutation(rng, idxes)
        idxes_policy.append(idxes)
    idxes_policy = jnp.array(idxes_policy).reshape(-1, size_minibatch)

    avg_metrics_dict = defaultdict(int)

    state = flatten_dims(state)
    action = flatten_dims(action).reshape(-1, 1)
    log_pi_old = flatten_dims(log_pi_old)
    value = flatten_dims(value)
    target = flatten_dims(target)
    gae = flatten_dims(gae)

    def scan_policy(train_state, idx):
        grad_fn = jax.value_and_grad(loss_actor_and_critic, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    state=state[idx],
                                    target=target[idx],
                                    value_old=value[idx],
                                    log_pi_old=log_pi_old[idx],
                                    gae=gae[idx],
                                    action=action[idx],
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff)

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss
    train_state, total_loss = jax.lax.scan(scan_policy, train_state, idxes_policy)
    total_loss, (value_loss, loss_actor, ent, value_pred, target_val, gae_val) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['value_loss'] += value_loss.mean()
    avg_metrics_dict['loss_actor'] += loss_actor.mean()
    avg_metrics_dict['ent'] += ent.mean()
    avg_metrics_dict['value_pred'] += value_pred.mean()
    avg_metrics_dict['target_val'] += target_val.mean()
    avg_metrics_dict['gae_val'] += gae_val.mean()

    return avg_metrics_dict, train_state, rng


@partial(jax.jit,
         static_argnames=("num_envs", "n_steps", "n_minibatch", "epoch_ppo", "n_quantiles", "temp", "clip_eps", "entropy_coeff", "critic_coeff", "gsf_coeff"))
def update_ppo_and_gsf(train_state: TrainState, batch: Tuple, gvf_nce: jnp.array, obs_nce: jnp.array, num_envs: int,
               n_steps: int, n_minibatch: int, epoch_ppo: int, n_quantiles: int, temp: float, clip_eps: float,
               entropy_coeff: float, critic_coeff: float, gsf_coeff: float, rng: PRNGKey):

    state, action, reward, log_pi_old, value, target, gae, task_ids = batch

    size_batch = num_envs * n_steps
    size_minibatch = size_batch // n_minibatch

    idxes = jnp.arange(num_envs * n_steps)
    idxes_policy = []
    for _ in range(epoch_ppo):
        rng, key = jax.random.split(rng)
        idxes = jax.random.permutation(rng, idxes)
        idxes_policy.append(idxes)
    idxes_policy = jnp.array(idxes_policy).reshape(-1, size_minibatch)
    rng2, rng = jax.random.split(rng)
    avg_metrics_dict = defaultdict(int)

    state = flatten_dims(state)
    action = flatten_dims(action).reshape(-1, 1)
    log_pi_old = flatten_dims(log_pi_old)
    value = flatten_dims(value)
    target = flatten_dims(target)
    gae = flatten_dims(gae)
    gvf_nce = flatten_dims(flatten_dims(gvf_nce))
    obs_nce = flatten_dims(flatten_dims(obs_nce))
    def scan_policy(train_state, idx):
        key, rng = jax.random.split(rng2)
        grad_fn = jax.value_and_grad(loss_actor_and_critic_and_gsf, has_aux=True)
        total_loss, grads = grad_fn(train_state.params,
                                    train_state.apply_fn,
                                    state=state[idx],
                                    target=target[idx],
                                    value_old=value[idx],
                                    log_pi_old=log_pi_old[idx],
                                    gae=gae[idx],
                                    action=action[idx],
                                    gvf_nce=gvf_nce,
                                    obs_nce=obs_nce,
                                    n_quantiles=n_quantiles,
                                    clip_eps=clip_eps,
                                    critic_coeff=critic_coeff,
                                    entropy_coeff=entropy_coeff,
                                    gsf_coeff=gsf_coeff,
                                    rng=key)

        train_state = train_state.apply_gradients(grads=grads)
        return train_state, total_loss
    train_state, total_loss = jax.lax.scan(scan_policy, train_state, idxes_policy)
    total_loss, (value_loss, loss_actor, ent, value_pred, target_val, gae_val, gsf_loss) = total_loss

    avg_metrics_dict['total_loss'] += total_loss.mean()
    avg_metrics_dict['value_loss'] += value_loss.mean()
    avg_metrics_dict['loss_actor'] += loss_actor.mean()
    avg_metrics_dict['ent'] += ent.mean()
    avg_metrics_dict['value_pred'] += value_pred.mean()
    avg_metrics_dict['target_val'] += target_val.mean()
    avg_metrics_dict['gae_val'] += gae_val.mean()
    avg_metrics_dict['gsf_loss'] += gsf_loss.mean()

    return avg_metrics_dict, train_state, rng