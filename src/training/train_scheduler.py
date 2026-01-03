import torch
from tqdm import tqdm
from .utils import ppo_update
from src.evaluation.evaluate import evaluate

def train_scheduler(agent, env, episodes=500):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        agent.optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    rewards = []

    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        done = False
        total = 0

        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)

            ppo_update(agent, state, action, log_prob, reward, next_state)

            total += reward
            state = next_state

        rewards.append(total)

        if (ep + 1) % 50 == 0:
            eval_r = evaluate(agent, env, episodes=5)
            scheduler.step(eval_r)
            print(f"Eval @ {ep+1}: {eval_r:.3f}")

    return rewards
