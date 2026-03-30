# DQN Atari Agent
Video Game Demo: [YouTube Link](https://youtu.be/edyTBbGMEag)

## Project Overview
This project implements a **Deep Q-Network (DQN)** agent to play Atari games using **Deep Reinforcement Learning**. The agent learns optimal actions by interacting with the environment and improving its performance over time through reward-based learning.

DQN combines **Q-Learning** with deep neural networks, allowing agents to achieve strong performance in complex environments like Atari games.

---

## Features
- Deep Q-Network (DQN) implementation
- Atari game environment support
- Experience Replay for stable training
- Target Network for improved learning
- Training performance visualization
- Model evaluation

---

## How It Works
1. The agent observes the current game state  
2. Selects an action using an ε-greedy policy  
3. Receives a reward from the environment  
4. Stores experiences in replay memory  
5. Learns by sampling past experiences  

---

## Tech Stack
- Python  
- PyTorch / TensorFlow  
- OpenAI Gym / Gymnasium  
- NumPy  
- Matplotlib  

---

## Project Structure

```
FORMATIVE-3_DQN-ATARI-AGENT/
│── logs/
│   ├── limpho/                # Limpho's training logs
│   └── rele/                  # Rele's training logs
│── models/                    # Saved models
│── .gitattributes
│── compare.py                 # Script to compare experiment results
│── play.py                    # Script to play using a trained model
│── README.md
│── train_Fidel.py             # Fidel's experiment training script
│── train_Limpho.py            # Limpho's experiment training script
│── train_Rele.py              # Rele's experiment training script
```

---

## Installation

### Clone the repository
```bash
git clone https://github.com/RELEBOHILE-PHEKO/Formative-3_DQN-Atari-Agent.git
cd Formative-3_DQN-Atari-Agent
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

- **Linux / Mac:**
  ```bash
  source venv/bin/activate
  ```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Limpho's experiments

```bash
python train_Limpho.py
```

### Run Rele's experiments

```bash
python train_Rele.py
```

### Compare experiment results

```bash
python compare.py
```

### Play using a trained model

```bash
python play.py
```

---

## Example Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | `0.0001` |
| Batch Size | `32` |
| Replay Buffer Size | `100000` |
| Gamma | `0.99` |
| Target Update Frequency | `1000` |

---

## Experiments

Both training scripts use `ALE/BankHeist-v5` as the environment, with `TOTAL_TIMESTEPS = 200,000`. Set `RUN_ALL = True` to run all experiments sequentially, or `RUN_ALL = False` to run a single experiment by index.

Logs are saved under `logs/limpho/` and `logs/rele/` respectively. Models are saved under `models/`.

---

### Limpho Experiments

| Experiment | Policy | Learning Rate | Gamma | Batch Size | Exploration Initial | Exploration Final | Buffer Size | Notes |
|---|---|---|---|---|---|---|---|---|
| Exp 1 | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.05 | 100000 | Standard CNN moderate |
| Exp 2 | CnnPolicy | 1e-3 | 0.99 | 16 | 1.0 | 0.05 | 100000 | High LR, small batch |
| Exp 3 | CnnPolicy | 1e-4 | 0.80 | 32 | 1.0 | 0.05 | 100000 | Low gamma (myopic) |
| Exp 4 | CnnPolicy | 1e-4 | 0.999 | 32 | 1.0 | 0.05 | 100000 | High gamma (farsighted) |
| Exp 5 | CnnPolicy | 1e-4 | 0.99 | 32 | 0.5 | 0.05 | 100000 | Lower initial exploration |
| Exp 6 | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.001 | 100000 | Near zero final exploration |
| Exp 7 | CnnPolicy | 3e-4 | 0.99 | 64 | 1.0 | 0.02 | 100000 | Balanced configuration |
| Exp 8 | CnnPolicy | 5e-4 | 0.995 | 128 | 1.0 | 0.05 | 100000 | High LR, gamma, batch |
| Exp 9 | MlpPolicy | 3e-4 | 0.99 | 64 | 1.0 | 0.05 | 100000 | MLP instead of CNN |
| Exp 10 | CnnPolicy | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 100000 | Limpho best guess |

---

### Relebohile Experiments

| Experiment | Policy | Learning Rate | Gamma | Batch Size | Exploration Initial | Exploration Final | Buffer Size | Notes |
|---|---|---|---|---|---|---|---|---|
| Exp 1 | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Baseline |
| Exp 2 | CnnPolicy | 5e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | High LR |
| Exp 3 | CnnPolicy | 1e-5 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Low LR |
| Exp 4 | CnnPolicy | 1e-4 | 0.90 | 32 | 1.0 | 0.05 | 100000 | Low gamma |
| Exp 5 | CnnPolicy | 1e-4 | 0.99 | 128 | 1.0 | 0.05 | 100000 | Large batch |
| Exp 6 | CnnPolicy | 1e-4 | 0.99 | 16 | 1.0 | 0.05 | 100000 | Small batch |
| Exp 7 | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 100000 | Long exploration |
| Exp 8 | CnnPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.2 | 100000 | High final epsilon |
| Exp 9 | MlpPolicy | 1e-4 | 0.99 | 32 | 1.0 | 0.05 | 10000 | MLP policy, smaller buffer |
| Exp 10 | CnnPolicy | 2.5e-4 | 0.995 | 64 | 1.0 | 0.01 | 100000 | Tuned combination |

---

## Results

- Agent improves over time through training
- Learns strategies for Atari gameplay
- Use `compare.py` to visualize and compare performance across experiments

---

## Future Improvements

- Double DQN
- Dueling Networks
- Prioritized Experience Replay
- Rainbow DQN
