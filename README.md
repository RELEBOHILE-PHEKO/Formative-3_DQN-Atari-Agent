#  DQN Atari Agent
 video game :
 https://youtu.be/edyTBbGMEag

##  Project Overview
This project implements a **Deep Q-Network (DQN)** agent to play Atari games using **Deep Reinforcement Learning**. The agent learns optimal actions by interacting with the environment and improving its performance over time through reward-based learning.

DQN combines **Q-Learning** with deep neural networks, allowing agents to achieve strong performance in complex environments like Atari games.



##  Features
- Deep Q-Network (DQN) implementation
- Atari game environment support
- Experience Replay for stable training
- Target Network for improved learning
- Training performance visualization
- Model evaluation



##  How It Works
1. The agent observes the current game state  
2. Selects an action using an ε-greedy policy  
3. Receives a reward from the environment  
4. Stores experiences in replay memory  
5. Learns by sampling past experiences  



##  Tech Stack
- Python  
- PyTorch / TensorFlow  
- OpenAI Gym / Gymnasium  
- NumPy  
- Matplotlib  



##  Project Structure
```

Formative-3_DQN-Atari-Agent/
│── models/               # Saved models
│── notebooks/            # Experiments & training notebooks
│── src/
│   ├── agent.py          # DQN agent
│   ├── network.py        # Neural network
│   ├── replay_buffer.py  # Memory buffer
│   └── train.py          # Training script
│── results/              # Outputs & graphs
│── requirements.txt
│── README.md
│──train_Fidel.py
│──train_Limpho.py
│──train_Rele.py



````



##  Installation

### Clone the repository
```bash
git clone https://github.com/RELEBOHILE-PHEKO/Formative-3_DQN-Atari-Agent.git
cd Formative-3_DQN-Atari-Agent
````

### Create a virtual environment

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Linux / Mac:

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```



##  Usage

### Train the agent

```bash
python src/train.py
```

### Evaluate the agent

```bash
python src/evaluate.py
```



##  Results

* Agent improves over time through training
* Learns strategies for Atari gameplay
* Performance can be visualized using graphs



##  Example Hyperparameters

* Learning Rate: `0.0001`
* Batch Size: `32`
* Replay Buffer Size: `100000`
* Gamma: `0.99`
* Target Update Frequency: `1000`



##  Future Improvements

* Double DQN
* Dueling Networks
* Prioritized Experience Replay
* Rainbow DQN
