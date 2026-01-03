import numpy as np

data = pd.read_csv("trainrl_data.csv")

if 'timestamp' in data.columns:
    data = data.drop(columns=['timestamp'])

data = data.apply(pd.to_numeric, errors='coerce').fillna(0)

print("Shape:", data.shape)
data.head()

class TrafficEnv(gym.Env):
    def __init__(self, data):
        super(TrafficEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.max_index = len(data) - 1

        # Define spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(data.columns)-1,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 4 signal phases

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Start from a random point in dataset
        self.current_step = np.random.randint(0, self.max_index - 50)
        obs = self.data.iloc[self.current_step, :-1].values.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_index

        row = self.data.iloc[self.current_step]
        wait_time = row.get('waiting_time', 0)
        queue_len = row.get('queue_length', 0)
        emergency = row.get('emergency_detected', 0) if 'emergency_detected' in row else 0

        reward = self.compute_reward(wait_time, queue_len, emergency, action)
        next_obs = row[:-1].values.astype(np.float32)
        truncated = False
        info = {}

        return next_obs, reward, done, truncated, info

    def compute_reward(self, wait_time, queue_len, emergency_detected, action):
        reward = - (0.7 * wait_time + 0.3 * queue_len)
        if emergency_detected and action == 0:  # example: NS-green helps emergency
            reward += 20
        return reward

    def render(self):
        pass
