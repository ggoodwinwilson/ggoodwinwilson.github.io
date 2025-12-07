Transformers in Reinforcement Learning: A Deep Dive into PPO
December 2025

Introduction
The goal of this project is to train an agent to solve Flappy Bird (gymnasium FlappyBird-v0) using LIDAR observations. My target is to first surpass my personal high score of 120 and ultimately attempt to beat the world record of 2607.
While the majority of introductory Reinforcement Learning (RL) projects rely on simple Multi-Layer Perceptrons (MLPs) or Convolutional Neural Networks (CNNs), the recent success of Large Language Models has sparked intense interest in applying Transformer architectures to other applications and modalities. This post analyzes the implementation of a standard MLP-based agent versus a Transformer-based agent, exploring how self-attention mechanisms can act as a temporal feature extractor for state-based control.

Put image here: high-score.gif
Context: Flappy Bird gameplay using an autonomous agent

The agent is trained via RL and built from scratch using PyTorch. I implemented Proximal Policy Optimization (PPO), featuring a modular architecture that supports both MLP and Transformer-Encoder backbones. PPO, introduced by OpenAI in 2017, is famously known for solving the OpenAI Rubik’s Cube robotics challenge and defeating Dota 2 professionals.
I chose PPO for this environment for several reasons:
•	Stability: Policy improvement clipping prevents drastic, destructive updates.
•	Robustness: It is generally less sensitive to hyperparameter tuning than off-policy methods.
•	Sample Efficiency: For an on-policy algorithm, it reuses training data effectively across multiple epochs.

There are downsides, though they are less relevant for this specific task. As an on-policy method, the logistics of training can be challenging, and it is sometimes outperformed by other algorithms on continuous output tasks. However, since Flappy Bird is a lightweight environment with discrete outputs (flap or don't flap), PPO is a good choice.

The Naïve Model Approach: MLP
I initially hypothesized that this environment could be solved using a simple MLP for both the agent model and value function. If we can solve it with a simple architecture, there is no need to overcomplicate things; if not, it serves as a solid baseline. To test this theory, I implemented 128 and 512-dimension, 2-layer networks.

Put image here: mlp-diagram.webp
Subtext: Simple MLP model architecture

The Training Process
The core of this system is the Agent class and the trajectory rollout. The algorithm works as follows:
1.	Play the game using the current policy and collect a fixed length of samples (a “trajectory”).
2.	Save all PPO-relevant variables in memory: observation, action, action probability, value, reward, and done flags.
3.	Calculate the loss objectives and backpropagate them through the policy and value models.
Using PPO to train the model turned out to be straightforward. The agent improved quickly, achieving scores up to 100 within an hour. However, training eventually plateaued, taking a full day to achieve a score of 1000. Note that I trained this on a laptop RTX 4060 GPU—significantly slower than state-of-the-art GPU hardware.
As the agent gains experience, we start to see diminishing returns per training cycle. The error signals become sparse once the agent reaches higher scores; essentially, if the agent rarely fails, it rarely learns from its mistakes.

MLP Results & The "Partial Observability" Problem
The MLP achieved surprisingly good results, but its performance seems to be bound by the observation limits. In specific scenarios, the agent fails because it cannot properly observe its surroundings.

If the bird is close to the corner of a pipe as it passes, the LIDAR signals may not "see" the bottom corner, leading to a collision. Consider the following case: the bird’s "vision" has passed the pipe, but the bird's hitbox has not. This results in failure with no apparent correlation in the observation space regarding what went wrong.

Put image here: bird-collision.webp
Subtext: Hitbox edge colliding with pipe, despite no obstacles in observation space.

The fundamental problem is that the environment is only a Partially Observable Markov Decision Process in this state. We cannot see the true state, only incomplete observations. Because the bird can only see in front of it, not around it, the Markov property, which assumes the current state contains all necessary information to decide an action, is broken.
To fix this, we need to add previous states to our observation. This history should allow the agent to remember the pipe location and act before a collision occurs.

We have a few options to inject past states into the model:
1.	Expand the MLP Input: Concatenate 5-10 past frames. With a fully connected net, this drastically increases the parameter count.
2.	Recurrent Networks (RNN/LSTM): Use hidden state vectors to implicitly store temporal information.
3.	Transformers: Implement a Transformer with past observations as input "tokens" and use self-attention to relate them.

For an agent this simple, an RNN or expanded MLP is likely the optimal engineering solution due to inference speed. Despite this, I chose the "overkill" Transformer solution to explore the topology in a control context.

Implementing the Transformer
1. Input Embedding & Positional Encoding
The environment returns a LiDAR-based state vector ($d_{in}=180$). To process this via a Transformer, I first project this feature vector into a higher-dimensional embedding space ($d_{model}=512$) using a learnable linear layer followed by LayerNorm.
Because the Transformer is permutation-invariant (it sees the sequence as a "bag of states" rather than a timeline), we must inject temporal order. I implemented Absolute Sinusoidal Positional Encodings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

This allows the model to generalize to sequence lengths potentially unseen during training and provides a distinct geometric relationship between timestamps $t$ and $t-k$.

2. The Attention Mechanism
The heart of the policy is the Multi-Head Attention block. This mechanism allows different temporal observations to influence one another. The model computes Query ($Q$), Key ($K$), and Value ($V$) matrices from the input sequence, and attention weights are calculated via a scaled dot-product:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In the context of Flappy Bird, this allows the agent to "attend" to specific past states that are most relevant to the current decision. For example, the observation of the bird 3 frames ago might be more critical for determining a jump action than the observation 10 frames ago.

Put gif here: part-trained-attention.gif
Subtext: Animation of the attention weights for each head and layer during gameplay.

Interpreting these results is a challenge, but it appears the Transformer is learning distinct features of the temporal data. I noticed that as model training progressed, the attention map became much sharper and more certain about which timesteps mattered. 
Attention weights with a fully trained Transformer.

Put gif here: fully-trained-attention.gif
Subtext: Fully trained transformer attention visualization

3. Value Head vs. Policy Head
The Transformer backbone is shared between the policy and value heads. Since they both rely on the same state history to derive their output and they’re trained simultaneously, sharing the backbone is computationally efficient. The final token's embedding ($x[:, -1, :]$) is passed to two separate linear heads:
•	Policy Head: Outputs logits for 2 actions (Flap / Do Nothing).
•	Value Head: Outputs a scalar estimate of the state value.

Training Stability: Using Entropy to Encourage Exploration
I encountered a significant issue during training where the model got stuck in a permanent "flap" action loop, causing the agent to hit the ceiling and lose immediately.

My hypothesis is that because the sparse reward signal is difficult to find, the agent discovers a local minimum (flapping constantly) and decides it is the "safest" policy.

Put image here: failure-path.gif
Subtext: Agent always takes the same "flap" path, ending in premature failure.

To rectify this, I utilized the entropy bonus (ent_coef=0.02) in the loss function, a standard PPO feature. This penalizes the model for being too confident too early, effectively preventing the probability distribution from collapsing into a deterministic policy. This forces the agent to keep its options open, widening the experience pool by rewarding the exploration of alternative actions.

Put image here: entropy-ablation.webp
Subtext: A comparison of training progress with and without the entropy bonus, using a simple MLP.

Conclusion & Future Work
This project demonstrates that Transformers can successfully solve low-dimensional continuous control tasks when treated as sequence modeling problems. By replacing the Markov assumption (current state is all that matters) with a sequence history processed via self-attention, the agent learns robust policies that account for previous observations even when the current sensor data is incomplete.

Future iterations of this project could explore the use of RNNs or expanded-input MLPs to compare the inference/training efficiency against the Transformer approach. It’s also worth exploring vectorized environments to unlock large speedups in training. I did not look into GPU optimization for this project, and it’s possible that much higher performance is possible with the same hardware setup.

Appendix

Training Algorithm Breakdown
The agent is trained using Proximal Policy Optimization (PPO), an on-policy gradient method that strikes a balance between ease of implementation and sample efficiency.

Generalized Advantage Estimation (GAE)
To reduce the variance of the gradient estimates, I implemented GAE. Instead of using raw rewards, we calculate the Advantage $A_t$, which measures how much better an action was compared to the baseline value function $V(s)$. The code computes the Temporal Difference (TD) error $\delta_t$ first:

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Then, it computes the exponentially weighted average of these errors:

$$A_t = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \delta_{t+k}$$

The Clipped Surrogate Objective
To prevent catastrophic forgetting (where a large gradient update destroys the policy), PPO constrains the policy update. We calculate the probability ratio between the new policy and the old policy: 

$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

If this probability changes too drastically, we clip the update. The loss function is defined as:

$$L^{CLIP}(\theta) = \mathbb{E}_t [ \min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) ]$$

Gradient Clipping
To prevent exploding gradients—common in Transformers due to their depth—torch.nn.utils.clip_grad_norm_ is applied before the optimizer step, capped at 0.5.

Supplemental Model Specifications

MLP Based Model:
•	Policy & Value Network: Separate
•	Input Dimension: 180 (observation dim)
•	Hidden Dimension: 128 / 512
•	Output Dimension: 2 (probabilities)
•	Layers: 2 (Fully Connected)
•	Number of Parameters: [Insert Count]

Transformer Based Model:
•	Policy & Value Network: Combined (Shared Backbone)
•	Embedding Dimension ($d_{model}$): 128 / 512
•	Attention Heads: 4
•	Layers: 4
•	Sequence Length: 10 (Sliding window)
•	Feed Forward Dim: 512 / 2048 ($4 \times d_{model}$)
•	Activation: GELU
•	Normalization: LayerNorm (Pre-norm)
•	Dropout: 0.1
•	Number of Parameters: [Insert Count]

PPO Hyperparameters
•	Optimizer: Adam (lr=3e-4)
•	Gamma ($\gamma$): 0.99 (Discount factor)
•	Lambda ($\lambda$): 0.95 (GAE smoothing)
•	Clip Range ($\epsilon$): 0.2
•	Rollout Buffer: 256 steps
•	Minibatch Size: 32
•	Epochs per Rollout: 3
