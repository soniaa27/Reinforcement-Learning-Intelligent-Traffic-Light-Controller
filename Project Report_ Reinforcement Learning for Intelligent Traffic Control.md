# **Project Report: Reinforcement Learning for Intelligent Traffic Control**

**Date:** 16-Nov-2025 **Project:** Intelligent Traffic Controller using PPO and Pygame

[https://github.com/PSaanviBhat/Reinforcement-Learning-Intelligent-Traffic-Light-Controller](https://github.com/PSaanviBhat/Reinforcement-Learning-Intelligent-Traffic-Light-Controller)

### **1\. Project Objective**

The primary objective of this project was to develop, train, and visualize an intelligent traffic light controller using Reinforcement Learning (RL). The goal was to create an agent that could learn an optimal policy for managing a four-way intersection, with the primary metrics for success being the minimization of vehicle waiting time and queue length. The agent was trained on a static, preprocessed dataset.

### **2\. Data Pipeline & Feature Engineering**

The foundation of the RL environment was a data-processing pipeline.

* **Raw Data:** The initial data (*smart\_traffic\_management\_dataset.csv*) contained raw, unscaled values for traffic flow, vehicle counts, and categorical descriptions (e.g., "Cloudy," "Red").  
* **Preprocessing Script (preprocess\_pipeline.py):** A dedicated Python script was used to clean this data. Its key functions included:  
  1. **Synthetic Feature Generation:** This was the most critical step. As detailed in *PREPROCESSING\_DOCUMENTATION.txt*, key metrics essential for the RL reward signal were **synthetically generated**. This includes the target variables waiting\_time\_seconds and queue\_length, which were calculated based on factors like signal status, traffic volume, and vehicle density.  
  2. **Feature Generation:** Additional informative features were created from the raw data, such as vehicle\_density, congestion\_index, and heavy\_vehicle\_ratio.  
  3. **Categorical Encoding:** Converting non-numeric features into a machine-readable format using **One-Hot Encoding**. For example, weather\_condition="Cloudy" became weather\_Cloudy=1, and signal\_status="Red" became signal\_Red=1.  
  4. **Normalization:** Scaling all numeric features to a uniform range (e.g., 0 to 1\) using a **Min-Max Scaler**. This is essential for stable neural network training.  
* **Processed Data:** The pipeline outputted the full\_preprocessed\_data.csv, which was then split into *train\_data.csv* and *test\_data.csv* for model training and validation.

### **3\. Reinforcement Learning Environment (Static)**

A custom RL environment was built, conforming to the gymnasium.Env interface, as detailed in *RL\_model\_TRAINING.ipynb*.

* **Environment Design:** Instead of a live simulation, the environment was data-driven. The step() function iterated row-by-row through the *train\_data.csv*. Each row represented a single "state" (a snapshot in time) at the intersection.  
* **State Space (state\_dim=30):** The state was defined as a vector of 30 features. This vector included all the preprocessed data points from a single row of *train\_data.csv*, such as scaled traffic volume, vehicle counts, weather conditions, and current signal status.  
* **Action Space (action\_dim=4):** A discrete action space with four possible actions was used:  
  * 0: Set North-South Green / East-West Red  
  * 1: Set North-South Yellow / East-West Red  
  * 2: Set East-West Green / North-South Red  
  * 3: Set East-West Yellow / North-South Red  
* **Reward Function:** The reward was engineered to be negative, framing the problem as minimizing a penalty. The reward signal was calculated based on the **synthetically generated** objectives: reward \= \- ( (w1 \* waiting\_time\_seconds) \+ (w2 \* queue\_length) ) This asked the agent to take actions that led to states (data rows) with the lowest possible (synthetic) waiting times and queues.

### **4\. Model Architecture & Training**

Proximal Policy Optimization (PPO) agent.

* **Algorithm:** **PPO** was chosen as it is a modern, highly stable, and sample-efficient **Actor-Critic** algorithm. It balances exploration (trying new actions) and exploitation (using known-good actions) effectively.  
* **Model (PPOAgent):** The agent's architecture consists of two simple Multi-Layer Perceptrons (MLPs):  
  * **Actor Network:** Takes the 30-feature state as input and outputs a probability distribution over the 4 possible actions.  
    * *Architecture:* Input(30) \-\> Linear(64) \-\> ReLU \-\> Linear(4) \-\> Softmax  
  * **Critic Network:** Takes the 30-feature state as input and outputs a single scalar value, estimating the "goodness" (V-value) of that state.  
    * *Architecture:* Input(30) \-\> Linear(64) \-\> ReLU \-\> Linear(1)  
* **Training:**  
  * The agent was trained by iterating through the *train\_data.csv* dataset for **4,500 episodes**.  
  * The final trained network weights were saved as *ppo\_traffic\_model\_4500.pt.*  
  * **Learned Behavior:** Analysis of the agent's decisions in the simulation revealed a learned bias. 

### **5\. Dynamic Visualization (Pygame)**

To test the model in a live, dynamic environment, a custom simulation was built using Pygame (*pygame\_ui.py)*.

* **Model Deployment:** The PPOAgent class was re-defined in the Pygame script, and the trained weights from *ppo\_traffic\_model\_4500.pt* were loaded into it.  
* **Simulation Loop:** The Pygame simulation runs an independent, event-based world. The "AI-Simulation" loop functions as follows:  
  * **Observe (Pygame):** The simulation observes its *own* state (e.g., counts the 10 cars in the N-S queue).  
  * **Create State (Pygame):** It builds a 30-feature "mock" state array by populating it with this live data, matching the format of the training data.  
  * **Act (AI Model):** This live state array is fed into the loaded PPO model (model.act(state)).  
  * **Execute (Pygame):** The model's returned action (0-3) is used to update the traffic light colors in the Pygame window.  
* **Simulation Features:**  
  * **Roads:** Wide, multi-lane (2 per direction) roads with right-hand drive logic and clear stop lines.  
  * **Signals:** Pole-mounted, 3-color signals with "N-S", "S-N", "E-W", and "W-E" labels for clarity.  
  * **Physics:** Stable vehicle logic for lane-specific collision avoidance and precise stopping at stop lines.  
  * **Emergency Vehicles (EVs):** A special vehicle type (white, flashing "EV") spawns randomly, moves faster, and ignores red lights. 

### **6\. Conclusions & Future Work**

This project successfully demonstrated the full lifecycle of an RL project, from data-driven training to dynamic deployment. The PPO model learned a clear policy from the static data.

The primary limitation is that the model's "worldview" is restricted to the patterns in the *train\_data.csv*, which are themselves based on synthetic wait times. To improve performance and fairness, the following steps are recommended:

1. **Re-train with a New Reward Function:** Modify the reward function to more heavily penalize high waiting times in *any* direction, encouraging the model to learn a more "fair" and balanced policy.  
2. **Add Emergency Vehicles to State:** Add a new feature to the state (e.g., ev\_approaching) and re-train the model. A new reward (e.g., a large bonus for clearing the path) would teach the AI to react to them.  
3. **Migrate to Online Training:** The most significant improvement would be to train the model *inside* the live Pygame simulation (or a tool like SUMO), allowing it to learn from its own real-time actions and consequences rather than from a fixed, synthetic dataset.

