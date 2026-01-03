🚦 Reinforcement Learning–Based Intelligent Traffic Light Controller
Overview:

This project implements an Intelligent Traffic Signal Control System using Reinforcement Learning (RL).
A Proximal Policy Optimization (PPO) agent is trained to dynamically control traffic signals based on real-world traffic data, with the goal of reducing waiting time, minimizing queue lengths, and prioritizing emergency vehicles.

Unlike traditional fixed-time traffic signals, this system learns optimal signal policies from data and adapts to changing traffic conditions.

Objectives:

Minimize average vehicle waiting time

Reduce traffic congestion (queue length)

Prioritize emergency vehicles when detected

Learn stable traffic signal policies using RL

Evaluate performance on unseen (test) data

Approach:

Reinforcement Learning Algorithm: Proximal Policy Optimization (PPO)

State Space: Traffic features extracted from real-world datasets

Action Space: Traffic signal phases (e.g., NS green, EW green, etc.)

Reward Function: Penalizes congestion and waiting time, rewards emergency handling

Training Strategy: Multi-phase training with fine-tuning and learning-rate scheduling

🔧 Environment & Libraries Used:

Python 3.10+

PyTorch – Neural network & PPO implementation

NumPy / Pandas – Data processing

Gymnasium-style environment (custom) – RL interface

Matplotlib – Training & evaluation plots

tqdm – Training progress visualization

🧪 State & Action Design
🔹 State Features (example)

Vehicle waiting time

Queue length

Traffic density

Emergency vehicle indicator

Time-step information

etc 
🔹 Action Space

Discrete actions representing traffic light phases:

North-South green

East-West green

Turning phases (if enabled)

🏆 Reward Function

The reward is designed to balance efficiency and safety:

wait_time /= 100
queue_len /= 50
reward = -(0.6 * wait_time + 0.4 * queue_len)

if emergency_detected:
    reward += 10

Interpretation:

Negative reward: High congestion or waiting time

Higher (less negative) reward: Smoother traffic flow

Emergency bonus: Encourages clearing emergency paths

🔁 Training Strategy:

Training is performed in multiple phases:

Base Training

PPO trained from scratch

Fixed learning rate

Fine-Tuning

Continue training from saved checkpoint

Lower learning rate for stability

Scheduler-Based Training

Learning rate reduced automatically when performance plateaus

Improves convergence and reduces reward noise

📈 Results & Observations:

Rewards initially show high variance (expected in RL)

Over training, rewards stabilize and improve

Reduced fluctuations after lowering learning rate

Final model demonstrates consistent performance on unseen test data

Note: In RL, absolute reward values are less important than trend and stability.

Testing & Evaluation:

Evaluation is performed without exploration

The trained policy selects actions greedily

Performance is measured as average episode reward on unseen data

How to Run (High-Level):

Preprocess dataset using scripts in preprocessing/

Initialize environment and PPO agent

Run base training → fine-tuning → scheduler training

Evaluate trained model on test data

(Optional) Visualize traffic signals using UI module

Key Takeaways:

Reinforcement Learning can effectively optimize traffic control

Reward normalization is crucial for stable training

Learning rate tuning significantly impacts convergence

Modular project structure improves scalability and clarity

Future Improvements:

Multi-intersection control

Multi-agent reinforcement learning (MARL)

Integration with SUMO or real-time simulators

Advanced reward shaping
