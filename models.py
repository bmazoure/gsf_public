from typing import Any, Optional, Tuple, List, Sequence

import flax.linen as nn
import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


def default_conv_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.xavier_uniform()

def default_mlp_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)

def default_logits_init(scale: Optional[float] = 0.01):
    return nn.initializers.orthogonal(scale)


class ResidualBlock(nn.Module):
    """Residual block."""
    num_channels: int
    prefix: str

    @nn.compact
    def __call__(self, x):
        # Conv branch
        y = nn.relu(x)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.num_channels,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding='SAME',
                    kernel_init=default_conv_init(),
                    name=self.prefix + '/conv2d_2')(y)

        return y + x


class Impala(nn.Module):
    """IMPALA architecture."""
    prefix: str

    @nn.compact
    def __call__(self, x):
        out = x
        for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2),
                                                        (32, 2)]):
            conv = nn.Conv(num_channels,
                           kernel_size=[3, 3],
                           strides=(1, 1),
                           padding='SAME',
                           kernel_init=default_conv_init(),
                           name=self.prefix + '/conv2d_%d' % i)
            out = conv(out)

            out = nn.max_pool(out,
                              window_shape=(3, 3),
                              strides=(2, 2),
                              padding='SAME')
            for j in range(num_blocks):
                block = ResidualBlock(num_channels,
                                      prefix='residual_{}_{}'.format(i, j))
                out = block(out)

        out = out.reshape(out.shape[0], -1)
        out = nn.relu(out)
        out = nn.Dense(256, kernel_init=default_mlp_init(), name=self.prefix + '/representation')(out)
        out = nn.relu(out)
        return out


class PPOModel(nn.Module):
    """Critic+Actor for PPO."""
    action_dim: int
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"

    @nn.compact
    def __call__(self, x):
        z = Impala(prefix='shared_encoder')(x)
        # Linear critic
        v = nn.Dense(1, kernel_init=default_mlp_init(), name=self.prefix_critic + '/fc_v')(z)

        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name=self.prefix_actor + '/fc_pi')(z)

        pi = tfd.Categorical(logits=logits)
        
        return (v, pi), None


class GSFPPOModel(nn.Module):
    """GSF learner on top of an Impala convnet."""
    action_dim: int
    n_quantiles: int
    temp: float

    @nn.compact
    def __call__(self, x, regression=True):
        z = Impala(prefix='shared_encoder')(x)
        # GSF classifier
        z_gsf = nn.LayerNorm()(z)
        z_gsf = nn.Dense(256, kernel_init=default_mlp_init(), name='gsf_fc_1')(z_gsf)
        z_gsf = nn.LayerNorm()(z_gsf)
        z_gsf = nn.relu(z_gsf)
        z_gsf = nn.Dense(self.n_quantiles, kernel_init=default_logits_init(), name='gsf_fc_2')(z_gsf)
        
        if not regression:
            z_gsf = nn.log_softmax(z_gsf/self.temp, axis=1)

        v = nn.Dense(1, kernel_init=default_mlp_init(), name='fc_v')(z)

        logits = nn.Dense(self.action_dim,
                        kernel_init=default_logits_init(),
                        name='fc_pi')(z)

        pi = tfd.Categorical(logits=logits)
        return (v, pi), z_gsf


class CriticCURL(nn.Module):
    """CQL critic."""
    dims: List[int]
    
    @nn.compact
    def __call__(self, x):
        z = Impala(prefix='shared_encoder')(x)
        v1 = MLP(self.dims, prefix='v1')(z)
        v2 = MLP(self.dims, prefix='v2')(z)
        
        bilinear = self.param(
            'bilinear',
            default_mlp_init(),  # Initialization function
            (z.shape[-1], z.shape[-1]))
        return (v1, v2), (z, bilinear)


class MLP(nn.Module):
  dims: Sequence[int]
  prefix: str

  @nn.compact
  def __call__(self, x):
    for i, dim in enumerate(self.dims):
        x = nn.Dense(dim,
                    kernel_init=default_mlp_init(),
                    name=self.prefix+'/%d' % i)(x)
        if i < len(self.dims) - 1:
            x = nn.relu(x)
    return x


class CriticCTRL(nn.Module):
    """Critic+Actor+Cluster."""
    n_actions: int
    dims: Sequence[int]
    n_cluster: int
    embedding_type: str
    n_att_heads: int
    prefix_critic: str = "critic"
    prefix_actor: str = "policy"
    

    def setup(self):
        self.encoder = Impala(prefix='')

        self.fc_v1 = MLP(dims=list(self.dims)+[self.n_actions], prefix='v1')
        self.fc_v2 = MLP(dims=list(self.dims)+[self.n_actions], prefix='v2')

        self.action_mlp = MLP(dims=list(self.dims[:-1])+[self.dims[-1]*2], prefix='action_embedding')

        self.attn = nn.SelfAttention(num_heads=self.n_att_heads,
                                 qkv_features=self.dims[-1],
                                 out_features=self.dims[-1])

        self.v_clust_mlp = MLP(dims=self.dims, prefix='v_clust_mlp')
        self.w_clust_mlp = MLP(dims=self.dims, prefix='w_clust_mlp')

        self.v_pred_mlp = MLP(dims=self.dims, prefix='v_pred_mlp')
        self.w_pred_mlp = MLP(dims=self.dims, prefix='w_pred_mlp')

        self.protos = nn.Dense(self.n_cluster,
                          kernel_init=default_mlp_init(),
                          name='protos')

    @nn.compact
    def __call__(self, state, action=None, reward=None):
        v1, v2 = self.ac(state[0])
        if action is None and reward is None:
            return (v1, v2), (v_clust, w_clust, v_pred, w_pred)
        v_clust, w_clust, v_pred, w_pred = self.cluster(state, action, reward)
        Q = self.protos(v_clust)

        return (v1, v2), (v_clust, w_clust, v_pred, w_pred)

    def protos_fn(self, x):
        return self.protos(x)

    def ac(self, state):
        # Features
        z = self.encoder(state)
        # Linear critic
        v1 = self.fc_v1(z)
        v2 = self.fc_v2(z)
        return (v1, v2), None
    

    def cluster(self, state, action, reward):
        """
        state: n_batch x n_timesteps x H x W x C
        action: n_batch x n_timesteps
        reward: n_batch x n_timesteps
        """
        img_shape = state.shape[2:]
        batch_shape = state.shape[:2]

        # z_state: n_batch x n_timesteps x n_hidden
        z_state = self.encoder(state.reshape(-1,*img_shape)).reshape(*batch_shape, -1)

        # z_action: n_batch x n_timesteps x n_hidden
        z_action = jax.nn.one_hot(action.reshape(-1), self.n_actions)
        z_action = self.action_mlp(z_action)
        gamma_a, beta_a = z_action.reshape(*batch_shape, -1).split(2, axis=-1)

        if self.embedding_type == "concat":
            z = ((1 + gamma_a) * z_state + beta_a).reshape(state.shape[0], -1)
        elif self.embedding_type == "attention":

            z = ((1 + gamma_a) * z_state + beta_a) #+ 0.5 * ((1 + gamma_r) * z_state + beta_r)
            z = z.reshape(state.shape[0], -1)
            z = self.attn(z)

        v_clust = self.v_clust_mlp(z)
        w_clust = self.w_clust_mlp(z)

        v_pred = self.v_pred_mlp(z)
        w_pred = self.w_pred_mlp(z)

        return v_clust, w_clust, v_pred, w_pred