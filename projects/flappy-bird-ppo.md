Transformers in Reinforcement Learning: A Deep Dive into PPO
December 2025

Introduction
While the majority of introductory Reinforcement Learning (RL) projects rely on simple Multi-Layer Perceptrons (MLPs) or Convolutional Neural Networks (CNNs), the recent success of Large Language Models has sparked intense interest in applying Transformer architectures to sequential decision-making. In this project, I engineered a custom reinforcement learning agent from scratch using PyTorch to solve the FlappyBird-v0 environment. Unlike standard implementations that rely on "black box" libraries like Stable Baselines 3, I implement Proximal Policy Optimization (PPO) manually, featuring a modular architecture that supports both MLP and Transformer-Encoder backbones. This writeup analyzes the implementation of the MLP-based agent versus a Transformer-based agent, exploring how self-attention mechanisms can act as a temporal feature extractor for state-based control.

Reinforcement Learning
The agent learns using Proximal Policy Optimization, a reinforcement learning technique introduce by OpenAI in 2017. PPO is famously known for solving the OpenAI Rubiks Cube robotics challenge and defeating Dota 2 professionals. It's a popular choice for reinforcement learning because of various reasons:
-Stable training thanks to policy improvement clipping
-Simple and easy to implement
-Works well across many tasks
-Robust and predictable, fewer hyperparameters to tune and less sensitive to them
-It's sample efficient for an on-policy algorithm, reusing the training data for mulitple epochs

With all the upsides, it's also worth mentioning the downsides and why they're not as relevant for this environment
-On-policy method, which can make the logistics of training a challenge
-Since this environment is cheap to capture lots of samples and doesn't take a long time to train, this is ok.

-Policy clipping can restrict training with less impactful backpropagation
-Total training time is low so taking longer is ok in exchange for more stability

-Can be out-performed by other algorithms on continuous output tasks
-Flappy bird is a categorical learning task with two discrete outputs, so continuous considerations don't apply

I would bet that many reinforcement learning techniques would be suitable for this environment, but I focused on PPO for it's clear advantages and popularity.

The Model
I first took the naive approach of utilizing a simple MLP for both the agent model, and the value function. I anticipated achieving acceptable performance from this architecture because of the simplicity of the environment, and if not, it would serve as a good baseline to compare against. 
-other options? 

System Architecture
The core of this system is the Agent class, which orchestrates data collection via the PPOMemory buffer and policy updates via the learn() loop. The architecture separates the "Experience" phase from the "Learning" phase. During the Experience Phase, the agent interacts with the environment, maintaining a sliding window of the past 10 observations (seq_len=10). This sequence is fed into the model to generate a stochastic action distribution. During the Learning Phase, the agent utilizes Generalized Advantage Estimation (GAE) to compute learning targets and updates the policy using the clipped surrogate objective.

Training
Using PPO to train the model turned out to be straight-forward. I found that the agent improved quickly in this environment with common PPO hyperparameters. The agent was able to achieve high scores up to 100 within an hour, however training did plateau and achieving a score of one thousand took a whole day. I trained on a laptop RTX4060 GPU which is not designed for AI and very slow compared to state-of-art GPU hardware.

I hypothesize that  took so long to reach large high scores was  but takes so long to perfect is that the error signals are so sparse once the agent gets good enough. It takes long for rollouts to lose, so there's not much error signal. Perhaps increasing the entropy reward coefficient would be a good way to encourage more exploration and forced error.

MLP Results

The Problem: partially observable markov chain

The Transformer Backbone
The most distinct feature of this implementation is the replacement of the standard fully connected policy network with a Transformer Encoder. In typical RL, handling temporal dependencies (like velocity or acceleration inferred from position) requires Frame Stacking or Recurrent Neural Networks (LSTMs). Here, I utilize Self-Attention.

1. Input Embedding & Positional Encoding
The environment returns a LiDAR-based state vector ($d_{in}=180$). To process this via a transformer, we first project this feature vector into a higher-dimensional embedding space ($d_{model}=512$) using a learnable linear layer followed by LayerNorm. Because the Transformer is permutation-invariant (it sees the sequence as a "bag of states"), we must inject temporal order. I implemented Sinusoidal Positional Encodings rather than learnable embeddings:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

This allows the model to generalize to sequence lengths potentially unseen during training and provides a distinct geometric relationship between timestamps $t$ and $t-k$.2. 

The Attention Mechanism
The heart of the policy is the Multi-Head Attention block. The model computes Query ($Q$), Key ($K$), and Value ($V$) matrices from the input sequence. The attention weights are calculated via scaled dot-product:

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

In the context of Flappy Bird, this mechanism allows the agent to "attend" to specific past states that are most relevant to the current decision. For example, the vertical velocity of the bird 3 frames ago might be more critical for determining a jump action than the exact pipe position 10 frames ago. 

The Training Algorithm
The agent is trained using Proximal Policy Optimization (PPO), an on-policy gradient method that strikes a balance between ease of implementation and sample efficiency. 

Generalized Advantage Estimation (GAE) 
To reduce the variance of the gradient estimates, I implemented GAE. Instead of using raw rewards, we calculate the Advantage $A_t$, which measures how much better an action was compared to the baseline value function $V(s)$. The code computes the Temporal Difference (TD) error 

$\delta_t$ first:$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

Then, it computes the exponentially weighted average of these errors:

$$A_t = \sum_{k=0}^{T-t-1} (\gamma\lambda)^k \delta_{t+k}$$

The Clipped Surrogate Objective
To prevent catastrophic forgetting (where a large gradient update destroys the policy), PPO constrains the policy update. We calculate the probability ratio between the new policy and the old policy: 

$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

The loss function is defined as:

$$L^{CLIP}(\theta) = \mathbb{E}_t [ \min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) ]$$

Note the inclusion of entropy. I added an entropy bonus (ent_coef=0.02) to the loss function. This penalizes the model for being too confident too early, effectively encouraging exploration by preventing the probability distribution from collapsing into a deterministic policy prematurely. 

Technical Implementation Details 
Observation Processing: The Flappy Bird environment provides distance readings from the bird to the surrounding environment, which serve as the observation it doesn't "know" anything else about the environment. I created a FIFO buffer of size 10 to store observations from previous timesteps. This effectively converts the Partially Observable Markov Decision Process (POMDP) into an MDP by providing the agent with memory of the past.

Adding Temporal History
I originally trained this agent with a simple multi-layer perceptron (MLP) model, which which takes the observation and outputs two   

Gradient Clipping: To prevent exploding gradients, particularly common in Transformers due to their depth, torch.nn.utils.clip_grad_norm_ is applied before the optimizer step, capped at 0.5.

Value Head vs Policy Head: The Transformer backbone is shared. The final token's embedding ($x[:, -1, :]$) is passed to two separate linear heads: 
    Policy Head: Outputs logits for 2 actions (Flap / Do Nothing).Value Head: Outputs a scalar estimate of the state value.
    
Supplemental Model Specifications
Transformer 
Policy Network
Embedding Dimension ($d_{model}$): 512
Attention Heads: 4
Layers: 4
Sequence Length: 10 (Sliding window of past observations)
Feed Forward Dim: 2048 ($4 \times d_{model}$)
Activation: GELU
Normalization: LayerNorm (Pre-norm configuration)
Dropout: 0.1 (Disabled during specific training phases for PPO stability)
PPO Hyperparameters
Optimizer: Adam (lr=3e-4)
Gamma ($\gamma$): 0.99 (Discount factor)
Lambda ($\lambda$): 0.95 (GAE smoothing)
Clip Range ($\epsilon$): 0.2
Rollout Buffer: 256 steps
Minibatch Size: 32
Epochs per Rollout: 3

Conclusion & Future Work
This project demonstrates that Transformers can successfully solve low-dimensional continuous control tasks when treated as sequence modeling problems. By replacing the Markov assumption (current state is all that matters) with a sequence history processed via self-attention, the agent learns more robust policies that account for velocity and acceleration implicitly. Future iterations of this project will explore Decision Transformers, where the target return is passed as a token to condition the generation of actions, moving from online PPO training to offline sequence modeling.