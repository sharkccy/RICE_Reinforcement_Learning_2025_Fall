import argparse
import torch 
import numpy as np
import time
from submission import TrainConfig, REINFORCEAgent, make_env, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train REINFORCE / REINFORCE+Baseline on MountainCarContinuous")
    
    parser.add_argument("--baseline", action="store_true", help="If set, use REINFORCE with a value baseline (use_baseline=True).")
    parser.add_argument("--train_steps", type=int, default=100, help="Number of policy updates (outer loops).")
    parser.add_argument("--min_batch_episodes", type=int, default=10, help="Number of episodes per update.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=0.001, help="Policy (and value) learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_episode_steps", type=int, default=700, help="Max steps per episode via TimeLimit wrapper.")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes to evaluate after training.")


    # parser.add_argument("--baseline", action="store_true", help="If set, use REINFORCE with a value baseline (use_baseline=True).")
    # parser.add_argument("--train_steps", type=int, default=900, help="Number of policy updates (outer loops).")
    # parser.add_argument("--min_batch_episodes", type=int, default=5, help="Number of episodes per update.")
    # parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    # parser.add_argument("--lr", type=float, default=0.00025, help="Policy (and value) learning rate.")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # parser.add_argument("--max_episode_steps", type=int, default=700, help="Max steps per episode via TimeLimit wrapper.")
    # parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes to evaluate after training.")
    
    # parser.add_argument("--baseline", action="store_true", help="If set, use REINFORCE with a value baseline (use_baseline=True).")
    # parser.add_argument("--train_steps", type=int, default=200, help="Number of policy updates (outer loops).")
    # parser.add_argument("--min_batch_episodes", type=int, default=2, help="Number of episodes per update.")
    # parser.add_argument("--gamma", type=float, default=0.996, help="Discount factor.")
    # parser.add_argument("--lr", type=float, default=3e-4, help="Policy (and value) learning rate.")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # parser.add_argument("--max_episode_steps", type=int, default=700, help="Max steps per episode via TimeLimit wrapper.")
    # parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes to evaluate after training.")
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()

    # Set global seeds
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Build environment factory with max_episode_steps
    env_make = lambda: make_env(seed=args.seed, max_episode_steps=args.max_episode_steps)

    # Set Config variables for training and evaluation
    cfg = TrainConfig(
        train_steps=args.train_steps,
        min_batch_episodes=args.min_batch_episodes,
        gamma=args.gamma,
        lr=args.lr,
        seed=args.seed,
        use_baseline=args.baseline,
    )

    # Create REINFORCE Agent 
    agent = REINFORCEAgent(env_make, cfg)

    # Simple Training Loop
    print(f"[main.py] Training started (baseline={args.baseline})...")
    agent.train(num_updates=cfg.train_steps)

    # Simple Evaluation
    avg, std = agent.evaluate(n_eval_episodes=args.eval_episodes)
    print(f"[main.py] Eval — mean: {avg:.2f}, std: {std:.2f}, baseline={args.baseline}")
    print(f"[main.py] Total time elapsed: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
