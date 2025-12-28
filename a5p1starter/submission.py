"""
submission.py — Starter code for "REINFORCE vs REINFORCE+Baseline on MountainCarContinuous"

Students: Complete ONLY the sections marked with
######## PUT YOUR CODE HERE ########
Do NOT rename classes/functions or this file.

You must support BOTH modes:
  - Pure REINFORCE (Monte-Carlo policy gradient)
  - REINFORCE with a learned value baseline (variance reduction)

Toggle via TrainConfig(use_baseline: bool).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as D
import gymnasium as gym

# Try to keep CPU usage predictable on autograder
try:
    torch.set_num_threads(1)
except Exception:
    pass


# ================================================================
# Utilities
# ================================================================

def set_seed(seed: int | None):
    """
    Set the random seed for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed. If None, no seeding is performed.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(seed: int | None = None, max_episode_steps: int | None = None):
    """
    Create a MountainCarContinuous-v0 Gymnasium environment, with optional
    seeding and custom max episode length.

    Args:
        seed: Random seed to apply to env, action space, and observation space.
        max_episode_steps: If not None, wrap env in TimeLimit with this horizon.

    Returns:
        A Gymnasium environment instance.
    """
    env = gym.make("MountainCarContinuous-v0")
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env.env, max_episode_steps=max_episode_steps)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


# ================================================================
# Networks
# ================================================================

class MLP(nn.Module):
    """
    Simple 2-layer MLP with Tanh activations, used by the policy and value nets.
    """

    def __init__(self, in_dim: int, hidden: Tuple[int, int], out_dim: int,
                 act: Callable[[], nn.Module] = nn.Tanh):
        """
        Initialize the MLP.

        Args:
            in_dim: Input feature dimension.
            hidden: Tuple of two hidden-layer sizes (h1, h2).
            out_dim: Output feature dimension.
            act: Activation constructor (default: nn.Tanh).
        """
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), act(),
            nn.Linear(h1, h2), act(),
            nn.Linear(h2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.

        Args:
            x: Input tensor of shape (B, in_dim).

        Returns:
            Output tensor of shape (B, out_dim).
        """
        return self.net(x)


class Policy(nn.Module):
    """
    Gaussian policy over an unsquashed action u; the environment action is a = tanh(u) in [-1, 1].

    # NOTE on the Policy function:
    In many continuous environments, the action space is not only continous
    but also bounded. This is also the case for Mountain Car Continuous, where
    the action (denoted as a) must be between [-1, 1].

    As such, we cannot simply represent the action as a Gaussian distribution;
    since, a Gaussian random variable can range from -infinity to +infinity.

    A common approach, which we also use here, is to introduce an intermediate
    unbounded variable (we denote this variable u, an unsquased action).
    Model the distribution of this unbounded random variable as a Gaussian.
    And, then squash the action between [-1, 1] using the tanh function.

    # HINT for computing log probabilities:
    The log_prob method of the Policy class provides log-probabilities of u.
    That is, log pi(u|s). Later while implementing the REINFORCE algorithm, you
    will also need to compute log-probabilites of a. That is, log pi(a|s).

    To compute these log-probabilites of a, simply taking the log probability
    of the Gaussian distribution is insufficent because of the tanh function.
    Instead, the following equation should be used:
            a = tanh(u)
            log pi(a|s) = log pi(u|s) - log(1 - a^2 + zeta)
    where zeta is a small positive value (e.g., 1e-6) added to avoid numerical
    errors when computing log probabilities.
    """

    def __init__(self, obs_dim: int = 2, hidden_sizes: Tuple[int, int] = (64, 64)):
        """
        Initialize the Gaussian policy network.

        Args:
            obs_dim: Dimension of the observation space.
            hidden_sizes: Two hidden-layer sizes used by the shared MLP.
        """
        super().__init__()
        self.feats = MLP(obs_dim, hidden_sizes, 64)
        # NOTE:
        # Policy encodes Gaussian distribution of the unsquased action, u.
        # Mean of the unsquased action u.
        self.mean_head = nn.Linear(64, 1)
        # Log of the standard deviation of the unsquased action u.
        # Here, log is used because learning standard deviation (std) directly
        # requires the optimizer to enforce the constraint that std >= 0.
        # In contrast, log_std can take any real value and does not require
        # the use of constrained optimization, thereby simplifying the
        # optimization and learning of standard deviation of the Gaussian.
        self.log_std = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))

        # Vincent's Note: Initialize log_std to small values for more stable initial training
        with torch.no_grad():
            self.mean_head.weight.mul_(0.01)
            self.mean_head.bias.zero_()


    def _preproc(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess observations before feeding into the network.

        For MountainCarContinuous, scaling the velocity helps with learning.

        Args:
            obs: Observation tensor of shape (B, obs_dim). 

        Returns:
            Preprocessed observation tensor of shape (B, obs_dim).
        """
        if obs.ndim == 2 and obs.shape[1] >= 2:
            obs = torch.stack([obs[:, 0], 10.0 * obs[:, 1]], dim=1)
        return obs

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and log-std of the Gaussian distribution over u.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            mean_u: Mean of the Gaussian over unsquashed actions u (B, 1).
            log_std: Log standard deviation (B, 1), broadcast from a global parameter.
        """
        # preproc will return obs of shape (B, obs_dim)
        obs = self._preproc(obs)
        # h means "hidden" not tanh
        h = self.feats(obs)
        mean_u = self.mean_head(h)
        # Expand log_std to match the batch size
        log_std = self.log_std.expand_as(mean_u)
        return mean_u, log_std

    def _dist(self, obs: torch.Tensor) -> D.Normal:
        """
        Build a Normal distribution over u for the given observations.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            A torch.distributions.Normal over unsquashed actions u.
        """
        mean_u, log_std = self.forward(obs)
        std = torch.exp(log_std).clamp_min(1e-6)
        return D.Normal(mean_u, std)

    def sample_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample unsquashed actions u ~ N(mean_u, std^2) with reparameterization.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            u: Sampled unsquashed actions (B, 1).
            logp_u: Log-probabilities log pi(u|s) (B, 1).
        """
        pi = self._dist(obs)
        u = pi.rsample()
        logp_u = pi.log_prob(u)
        return u, logp_u

    def log_prob(self, obs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute log pi(u|s) under the Gaussian policy.

        Args:
            obs: Observation tensor of shape (B, obs_dim).
            u: Unsquashed actions tensor of shape (B, 1).

        Returns:
            Log-probabilities log pi(u|s) of shape (B, 1).
        """
        return self._dist(obs).log_prob(u)


class ValueNet(nn.Module):
    """
    State-value function V(s), used as a learned baseline when cfg.use_baseline=True.
    """

    def __init__(self, obs_dim: int = 2, hidden_sizes: Tuple[int, int] = (64, 64)):
        """
        Initialize the value network.

        Args:
            obs_dim: Dimension of the observation space.
            hidden_sizes: Two hidden-layer sizes used by the shared MLP.
        """
        super().__init__()
        self.feats = MLP(obs_dim, hidden_sizes, 64)
        self.v = nn.Linear(64, 1)

    def _preproc(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Preprocess observations similarly to the policy network.

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            Preprocessed observation tensor of shape (B, obs_dim).
        """
        if obs.ndim == 2 and obs.shape[1] >= 2:
            obs = torch.stack([obs[:, 0], 10.0 * obs[:, 1]], dim=1)
        return obs

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the scalar state value V(s).

        Args:
            obs: Observation tensor of shape (B, obs_dim).

        Returns:
            State-value tensor of shape (B, 1).
        """
        obs = self._preproc(obs)
        h = self.feats(obs)
        return self.v(h)


# ================================================================
# BaseAgent
# ================================================================

class BaseAgent:
    """
    Base class that wraps a single environment instance and provides helpers
    for stepping, resetting, and running full episodes.
    """

    def __init__(self, env_make: Callable[[], Any]):
        """
        Initialize the agent and create one environment instance.

        Args:
            env_make: Zero-argument callable that returns a Gymnasium environment.
        """
        self._make = env_make
        self.env = env_make()
        self.obs, _ = self.env.reset()

    def env_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in the environment.

        Args:
            action: Action to apply to the environment (np.ndarray).

        Returns:
            obs: Next observation.
            reward: Scalar reward.
            done: Whether the episode has terminated or truncated.
            info: Environment info dict.
        """
        obs, r, term, trunc, info = self.env.step(action)
        done = term or trunc
        return obs, float(r), done, info

    def env_reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        Returns:
            obs: Initial observation after reset.
        """
        obs, _ = self.env.reset()
        return obs

    def run_episode(self, policy_func: Callable[[np.ndarray], np.ndarray]) -> Tuple[List[Dict], float]:
        """
        Run a single full episode using a provided policy function.

        Args:
            policy_func: Function mapping observation -> action (np.ndarray).

        Returns:
            episode: List of step dicts with keys:
                     {"obs", "act", "rew", "next_obs", "done"}.
            total_return: Cumulative reward over the episode.
        """
        episode = []
        obs = self.env_reset()
        done = False
        total = 0.0
        while not done:
            act = policy_func(obs)
            next_obs, r, done, _ = self.env_step(act)
            episode.append(
                {"obs": obs, "act": act, "rew": r, "next_obs": next_obs, "done": done}
            )
            obs = next_obs
            total += r
        return episode, total


# ================================================================
# REINFORCE Agent (supports baseline via cfg.use_baseline)
# ================================================================

@dataclass
class TrainConfig:
    """
    Configuration dataclass used by REINFORCEAgent.

    The autograder will construct TrainConfig instances directly,
    so do NOT rename, remove, or change the field names.
    """

    train_steps: int = 120            # Number of policy update iterations to run (Do not exceed 1000)
    min_batch_episodes: int = 8     # Episodes collected per update (Monte-Carlo batch size)
    gamma: float = 0.995            # Discount factor for computing returns
    lr: float = 0.0008             # Learning rate for both policy and (optional) value network
    seed: int = 42                  # Random seed for reproducibility
    use_baseline: bool = False      # If True, use a learned value function as baseline (advantage = G - V)


class REINFORCEAgent(BaseAgent):
    """
    REINFORCE agent with optional learned value baseline.

    If cfg.use_baseline is False:
        - Pure REINFORCE with returns G_t as advantages.
    If cfg.use_baseline is True:
        - Use ValueNet(s) to compute V(s) and advantages A_t = G_t - V(s_t).
    """

    def __init__(self, env_make: Callable[[], Any], cfg: TrainConfig):
        """
        Initialize the REINFORCE agent.

        Args:
            env_make: Callable that creates a Gymnasium environment.
            cfg: TrainConfig instance with learning hyperparameters.
        """
        super().__init__(env_make)
        self.cfg = cfg

        obs_dim = self.env.observation_space.shape[0]

        # Policy and (optional) value function
        self.policy = Policy(obs_dim=obs_dim, hidden_sizes=(64, 64))
        if cfg.use_baseline:
            self.value_fn = ValueNet(obs_dim=obs_dim, hidden_sizes=(64, 64))
            self.use_baseline = True
        else:
            self.value_fn = None
            self.use_baseline = False

        # Optimizers
        # 1
        self.pi_optim = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)
        self.vf_optim = (
            torch.optim.Adam(self.value_fn.parameters(), lr=cfg.lr)
            if self.value_fn is not None
            else None
        )

        # Action bounds
        self._low = self.env.action_space.low.astype(np.float32)
        self._high = self.env.action_space.high.astype(np.float32)

        # Policies
        self.behavior_policy = self.make_behavior_policy()
        self.eval_policy = self.make_policy()

        self._best_ep_ret = -1e9

        ######## PUT YOUR CODE HERE ########
        # (Optional) Add any additional initialization code here
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.pi_optim, T_max=cfg.train_steps, eta_min=1e-5
        # )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.pi_optim, step_size=1, gamma=0.999
        )
        self.logpa_old = None
        self.logpa_epsilon = 0.2
        self.entropy_coef_with_baseline = 1e-2
        self.entropy_coef_without_baseline = 5e-3
        ######## PUT YOUR CODE HERE ########

    # --------- Policies ---------

    def make_policy(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Deterministic evaluation policy: use the mean action (after tanh).

        Returns:
            A function mapping observation (np.ndarray) -> action (np.ndarray).
        """

        def policy_func(obs: np.ndarray) -> np.ndarray:
            """
            Evaluation-time policy: no exploration noise, just tanh(mean_u).

            Args:
                obs: Observation array of shape (obs_dim,).

            Returns:
                Action array of shape (1,) clipped to environment bounds.
            """
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean_u, _ = self.policy.forward(obs_t)
            a = torch.tanh(mean_u).detach().cpu().numpy().squeeze(0)
            a = np.clip(a, self._low, self._high)
            return a

        return policy_func

    def make_behavior_policy(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Stochastic behavior policy used during training:
        sample u ~ N(mean_u, std^2), then a = tanh(u).
        """

        def policy_func(obs: np.ndarray) -> np.ndarray:
            """
            Training-time policy: stochastic Gaussian over u, tanh-squashed.

            Args:
                obs: Observation array of shape (obs_dim,).

            Returns:
                Action array of shape (1,) clipped to environment bounds.
            """
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            u, _ = self.policy.sample_action(obs_t)   # unsquashed
            a_t = torch.tanh(u)                       # squashed
            a = a_t.detach().cpu().numpy().squeeze(0)
            a = np.clip(a, self._low, self._high)
            return a

        return policy_func

    # --------- Returns helper ---------

    @staticmethod
    def _discounted_returns(rewards: List[float], gamma: float) -> torch.Tensor:
        """
        Compute discounted returns G_t from a list of rewards.

        Args:
            rewards: List of rewards [r_0, r_1, ..., r_{T-1}].
            gamma: Discount factor in [0, 1).

        Returns:
            Tensor of shape (T, 1) with G_t for t = 0..T-1.
        """
        # G = 0.0
        # out = []
        # for r in reversed(rewards):
        #     G = r + gamma * G
        #     out.append(G)
        # out.reverse()
        # return torch.tensor(out, dtype=torch.float32).unsqueeze(-1)  # (T, 1)

        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        discounts = gamma ** torch.arange(len(rewards), dtype=torch.float32)
        out = torch.flip(torch.cumsum(torch.flip(rewards * discounts, dims=[0]), dim=0), dims=[0]) / discounts
        return out.unsqueeze(-1)

    # --------- Core update ---------

    def update(self, episodes: List[List[Dict]]):
        """
        Perform one REINFORCE (or REINFORCE+baseline) update from a batch
        of episodes.

        Implement both:
          - REINFORCE (no baseline)
          - REINFORCE with baseline
        """
        ######## PUT YOUR CODE HERE ########
        # (Optional) Common code for both variants of REINFORCE
        # obs_batch, act_batch, ret_batch = [], [], []
        # for ep in episodes:
        #     rewards = [step["rew"] for step in ep]
        #     returns = self._discounted_returns(rewards, self.cfg.gamma)  # (T, 1)
        #     for idx, step in enumerate(ep):
        #         obs_batch.append(step["obs"])
        #         act_batch.append(step["act"])
        #         ret_batch.append(returns[idx])
        # obs_batch = torch.tensor(np.array(obs_batch), dtype=torch.float32)  # (N, obs_dim)
        # # Ensure actions have shape (N, 1) to avoid accidental extra dimensions
        # act_batch = torch.tensor(np.array(act_batch), dtype=torch.float32).reshape(-1, 1)
        # act_batch = torch.clamp(act_batch, -0.999, 0.999)  # avoid numerical issues when computing log prob later
        # ret_batch = torch.cat(ret_batch, dim=0)  # shape (N,) or (N,1)

        obs_list = [torch.from_numpy(np.asarray([step["obs"] for step in ep], dtype=np.float32)) for ep in episodes]
        act_list = [torch.from_numpy(np.asarray([step["act"] for step in ep], dtype=np.float32).reshape(-1, 1)) for ep in episodes]
        ret_list = []

        for ep in episodes:
            rewards = [step["rew"] for step in ep]
            returns = self._discounted_returns(rewards, self.cfg.gamma)  # torch tensor (T,1)
            ret_list.append(returns)  # 直接 append tensor

        # 直接一次 stack / concatenate，再轉 torch tensor
        obs_batch = torch.cat(obs_list, dim=0)
        act_batch = torch.cat(act_list, dim=0).clamp_(-0.999, 0.999)
        ret_batch = torch.cat(ret_list, dim=0)  # (N, 1)

        # if ret_batch.ndim == 1:
        #     ret_batch = ret_batch.unsqueeze(-1)  # shape -> (N,1)

        # Policy loss
        logp_u = self.policy.log_prob(obs_batch, torch.atanh(act_batch))  # (N, 1)
        zeta = 1e-6
        logp_a = logp_u - torch.log(1 - act_batch ** 2 + zeta)  # (N, 1), when a is close to -1 or 1, 1 - a^2 can be close to 0 
        # Entropy bonus (optional, helps exploration)
        dist = self.policy._dist(obs_batch)
        entropy = dist.entropy().mean()

        ######## PUT YOUR CODE HERE ########
        if self.use_baseline:
            ######## PUT YOUR CODE HERE ########
            # REINFORCE with baseline
            # Compute value estimates
            V_s = self.value_fn(obs_batch)  # (N, 1)
            advantages = ret_batch - V_s.detach()  # (N, 1), At: advantages\

            # Normalize advantages to have mean 0 and std 1 for more stable training
            # if advantages.shape[0] > 1: # avoid division by zero when batch size is 1
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # if self.logpa_old is not None:
            #     # Clipped Surrogate Objective
            #     print("[REINFORCEAgent] Using clipped surrogate objective for policy loss.")
            #     ratio = torch.exp(logp_a - self.logpa_old)  # (N, 1)
            #     clipped_ratio = torch.clamp(ratio, 1 - self.logpa_epsilon, 1 + self.logpa_epsilon)
            #     unclipped = ratio * advantages
            #     clipped = clipped_ratio * advantages
            #     pi_loss = - torch.min(unclipped, clipped).mean()  # scalar
            # else:
            #     pi_loss = - (logp_a * advantages).mean()  # scalar



            # Value function loss
            vf_loss = nn.MSELoss()(V_s, ret_batch)  # scalar
            self.vf_optim.zero_grad()
            vf_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.value_fn.parameters(), max_norm=0.5)
            self.vf_optim.step()

            pi_loss = - (logp_a * advantages).mean() - self.entropy_coef_with_baseline * entropy  # scalar
            self.pi_optim.zero_grad()
            pi_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.pi_optim.step()

            # with torch.no_grad():
            #     self.logpa_old = logp_u - torch.log(1 - act_batch ** 2 + zeta)  # (N, 1)
            self.lr_scheduler.step()
            self.entropy_coef_with_baseline *= 0.995
        else:
            ######## PUT YOUR CODE HERE ########
            # REINFORCE (no baseline)
            # reward scaling for more stable training
            ret_std = ret_batch.std() + 1e-8
            ret_batch = ret_batch / ret_std
            advantages = ret_batch  # (N, 1), At: returns Gt

            pi_loss = - (logp_a * advantages).mean() - self.entropy_coef_without_baseline * entropy  # scalar
            # pi_loss = - (logp_a * advantages).mean() - self.entropy_coef * entropy  # scalar
            self.pi_optim.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.pi_optim.step()
            self.lr_scheduler.step()
            self.entropy_coef_without_baseline *= 0.995
            # Policy loss
            # logp_u = self.policy.log_prob(obs_batch, torch.atanh(act_batch))
            # zeta = 1e-6
            # logp_a = logp_u - torch.log(1 - act_batch ** 2 + zeta)  # (N, 1)
            # if self.logpa_old is not None:
            #     # Clipped Surrogate Objective
            #     ratio = torch.exp(logp_a - self.logpa_old)  # (N, 1)
            #     clipped_ratio = torch.clamp(ratio, 1 - self.logpa_epsilon, 1 + self.logpa_epsilon)
            #     unclipped = ratio * advantages
            #     clipped = clipped_ratio * advantages
            #     pi_loss = - torch.min(unclipped, clipped).mean() - self.entropy_coef * entropy  # scalar
            # else:
            #     pi_loss = - (logp_a * advantages).mean() - self.entropy_coef * entropy  # scalar


            # keep a raw copy for debugging
            # adv_raw = advantages.clone()

            # Only normalize if we have >1 sample and non-negligible std
            # if advantages.numel() > 1:
            #     adv_mean = advantages.mean()
            #     advantages = advantages - adv_mean

            # pi_loss = - (logp_a * advantages).mean()  # scalar, maximize expected return, use minus because optimizers minimize

            # adv_raw = adv_raw if 'adv_raw' in locals() else (ret_batch - V_s.detach() if self.use_baseline else ret_batch)
            # prod_raw = (logp_a * adv_raw)
            # prod_post = (logp_a * advantages)
            # print("[DEBUG] batch size:", logp_a.shape[0])
            # print("[DEBUG] logp_a mean/std:", logp_a.mean().item(), logp_a.std().item())
            # print("[DEBUG] adv raw mean/std:", adv_raw.mean().item(), adv_raw.std().item())
            # print("[DEBUG] adv post mean/std:", advantages.mean().item(), advantages.std().item())
            # print("[DEBUG] pi_loss raw (no centering):", -prod_raw.mean().item())
            # print("[DEBUG] pi_loss post (after centering/norm):", -prod_post.mean().item())

            # cov_raw = ((logp_a - logp_a.mean()) * (adv_raw - adv_raw.mean())).mean()
            # cov_post = ((logp_a - logp_a.mean()) * (advantages - advantages.mean())).mean()
            # print("[DEBUG] cov raw:", cov_raw.item(), "cov post:", cov_post.item())
            ######## PUT YOUR CODE HERE ########

        ######## PUT YOUR CODE HERE ########
        # (Optional) Common code for both variants of REINFORCE
        # pi_loss = - (logp_a * advantages).mean() - self.entropy_coef * entropy  # scalar
        # self.pi_optim.zero_grad()
        # pi_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        # self.pi_optim.step()

        # # with torch.no_grad():
        # #     self.logpa_old = logp_u - torch.log(1 - act_batch ** 2 + zeta)  # (N, 1)
        # self.lr_scheduler.step()
        # self.entropy_coef *= 0.995

        # print(f"[REINFORCEAgent] logp_a mean: {logp_a.mean().item():.4f}, std: {logp_a.std().item():.4f}")
        # print(f"[REINFORCEAgent] Average return in batch: {ret_batch.mean().item():.2f}")
        # print raw advantage stats if present
        # try:
        #     print(f"[REINFORCEAgent] adv raw mean: {adv_raw.mean().item():.2f}, std: {adv_raw.std().item():.2f}")
        # except Exception:
        #     pass
        # print(f"[REINFORCEAgent] Average advantage in batch (post-norm): {advantages.mean().item():.2f}, std: {advantages.std().item():.2f}")
        # print(f"[REINFORCEAgent] Policy loss: {pi_loss.item():.4f}")
        # print(f"[REINFORCEAgent] Log Std: {self.policy.log_std.item():.4f}")
        # print(f"[REINFORCEAgent] Learning Rate: {self.pi_optim.param_groups[0]['lr']:.6f}")
        
        ######## PUT YOUR CODE HERE ########

    # --------- Training loop ---------

    def train(self, num_updates: int):
        """
        Main training loop.

        Args:
            num_updates: Number of REINFORCE updates (outer loop iterations).
        """
        for update_idx in range(num_updates):
            batch_eps = []
            for _ in range(self.cfg.min_batch_episodes):
                ep, _ = self.run_episode(self.behavior_policy)
                batch_eps.append(ep)
            self.update(batch_eps)

    # --------- Eval helper ---------

    def evaluate(self, n_eval_episodes: int = 3) -> Tuple[float, float]:
        """
        Evaluate the current policy over multiple episodes.

        Args:
            n_eval_episodes: Number of evaluation episodes.

        Returns:
            mean_return: Average return across episodes.
            std_return: Standard deviation of returns.
        """
        rets = []
        for _ in range(n_eval_episodes):
            _, total = self.run_episode(self.eval_policy)
            rets.append(total)
        arr = np.array(rets, dtype=np.float32)
        return float(arr.mean()), float(arr.std())
