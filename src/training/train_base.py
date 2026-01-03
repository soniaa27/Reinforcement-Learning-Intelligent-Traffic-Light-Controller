from tqdm import tqdm
from .utils import ppo_update

def train_base(agent, env, episodes=1000):
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
        print(f"Episode {ep+1} | Reward: {total:.2f}")

    return rewards
