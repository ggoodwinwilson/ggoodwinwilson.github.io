 ---
  title: Understanding the PI* 0.6 VLA Model
  slug: pi-star-06-analysis
  date: November 29th, 2025
  summary: I went down a rabbithole of trying to understand Physical Intelligence VLA models and thought this writeup may help others doing the same
  tags: [rl, vision, robotics, VLA, multi-modal]
  Disclaimer: This is an unofficial analysis of the $\pi_{0.6}$ architecture based on the public papers and open-source code released by Physical Intelligence. All interpretations are my own and do not represent the authors. For the official implementation, please refer to the following sources:
  https://www.pi.website/blog
  https://github.com/Physical-Intelligence/openpi
  ---

Introduction
In this writeup I analyze the Physical Intelligence PI* 0.6 model, a state-of-the-art model for generalized robotics AI. Physical Intelligence has not open-sourced PI0.6 code yet, but they have released a technical white-paper on the model. We can use the open sourced PI0 model as a framework, and then add details from the whitepaper to infer what the PI0.6 architecture is. 

codex, put PI0.6 Model.svg here (located in /assets)
Subtext: Diagram of the PI* 0.6 VLA model

The high-level idea of the PI VLA models made sense to me, but a lot of the small implementation details were hazy when I tried to explain them to myself. If you're in a similar situation, maybe this write-up can help us understand.

Overall Architecture
The key to understanding PI0.6 is to understand how vision, language, and action models are stitched together. I like to think of the vision & language model (VLM) as completely separate from the "action expert" model. The only thing which is shared between the models is the attention mechanism. The action model concatenates it's data vectors with the VLM and therefore it's actions are influenced by the VLMs understanding of the surroundings and objective. Mathematically, multiple dot product operations are being computed in attention between vision, language, and action information. 

Gemma3, the Vision and Language Backbone

codex, put Gemma3 Model.svg here (located in /assets)
Subtext: The Gemma3 Vision Language Model Diagram

PI0.6 uses the 4B parameter version of Google Gemma3 to process robot camera feeds and language. Internally, Gemma 3 consists of two main parts: 1, a vision transformer encoder called SigLIP with a modified projection "connector" layer at the end to match the Gemma3 embedding dimension and 2, a small but typical LLM. SigLIP breaks up the robot camera feeds (4x) into patches and encodes them into the LLMs token space. Now the model has a semantic understanding of what the robot cameras see.

What Order is the Model Trained?
Firstly, Gemma3 is first pre-trained by Google on web-scale data, where it learns to understand what things are. Next, it's fine-tuned by PI on robot datasets. Training on robot datasets is crucial so that the VLM gains an understanding of what actions "mean", called "action representations". This is all before the action expert is even introduced.

So how is the VLM trained on a bunch of robot data without an action expert? How are action tokens represented? PI's solution is to compress all of the robot movement data (joint angles, gripper position, etc.) with the Discrete Cosine Transform (DCT), a compression method used in image compression. They call these special tokens "FAST" tokens.
    1. The whole robot training dataset is compressed into these FAST tokens using the compression algorithm
    2. The VLM is trained to replicate these action tokens, one by one, simply by minimizing cross-entropy loss. For example, given this image, task text, and robot state, how close were we to the real robot action that was taken in the dataset?
This strategy is good enough to complete a lot of tasks without even adding a special dedicated action expert. The model can just generate FAST tokens, de-compress them, and send the commands to the robot joints. Unfortunately, inference is far too slow and actions are not smooth. For the robot to generate 1 second of movement, it needs to generate ~60 FAST tokens.

The Token Space
Imagine two separate parts which make up the token space. The "Prefix", and the "Suffix". Fine-tuning simply tries to guess the suffix given some prefix.

codex, put PI0.6 Tokens - VLM Training.svg here (located in /assets)
Subtext: The VLM token space during FAST token fine tuning

The prefix is made up of 3 categories - Vision, language, and physical data called the "proprioceptive" state. Basically, the current joint angles of the robot and gripper positions are binned into 0-255 integers so the LLM can easily treat them as singular tokens.

A fine-tuning example may look like this:
1.Given the Prefix: <image tokens here> how should the arm move to pick up the red block? State: 25, 97, 190 ... 11
2. Predict the Suffix: <FAST action tokens here>
3. Compare with the real suffix: <more FAST action tokens>
4. Calculate the cross-entropy loss between prediction and real suffix, and backpropogate.

After training, the model has internalized all of these action representations from robot-specific fine tuning. It's time to introduce the dedicated action model to operate smoothly. The action expert has a simple objective - given the VLM's attention matrix, predict the next 50 actions. These future actions can be predicted simultaneously with a "flow-matching" model. Imagine an AI image generator, but instead of image pixels, the model is generating a matrix of future joint positions for the robot. It works by starting with an action matrix of pure gaussian noise, and generates a matrix of velocity vectors which guide the action matrix to de-noise itself. The action matrix joint positions move in this learned direction a bit, and repeats this de-noising process 10 times to fully generate the actions. The training dataset for the action expert is labelled demonstrations of the open-sourced "ALOHA" robot architecture, which consists of:
    - Two 6-DOF arms with grippers
    - A front-facing camera
    - A camera on each gripper

Fine Tuning the Action Expert
This time we aren't training the VLM at all, and only focusing on the flow-matching action expert instead. During this training phase, the model knows the prefix and the suffix (true actions), but noises the true actions a random amount and tries to predict the velocity matrix to de-noise them:

codex, put PI0.6 Tokens - VLM Training.svg here (located in /assets)
Subtext: The VLA token space during action expert fine tuning

Here's the action expert training recipe:
    1. Sample Data: Sample a real robot trajectory from the dataset ($x_1$) and a random  for the action matrix ($x_0$).
    2. Sample Time: Pick a random time to represent "how de-noised" the matrix is, $t \in [0, 1]$.
    3. Interpolate: Create a "noisy" intermediate sample $x_t$ which is just a weighted average: $t \cdot x_1 + (1-t) \cdot x_0$.
    4. Train: Ask the neural network: "Given this noisy $x_t$ and time $t$, what vector points directly to $x_1$?"
        ○ The answer is always mathematically simple: $x_1 - x_0$.
    5. Result: The network learns to look at any noisy garbage and tell you which direction points to a valid robot motion.

Fine-tuning vs. Pre-training
I was confused about the difference for a while. From my current understanding:
    - The objective in both is: minimize how far away are we from the right answer
    - Pre-training is next word prediction across the whole internet
    - Fine tuning is next word prediction given some specific input
This is how model behaviors and "personalities" are trained into the weights. We can make the models act a certain way given some input that we choose. In thise case, we're just making the Gemma VLM model really good at understanding actions. Another note - fine tuning doesn't need to be reinforcement learning, although it can be.

Combining the VLM and Action Expert

codex, put PI0.6 Attention.svg here (located in /assets)
Subtext: Combining attention matrices for VLM and action models

This is section is more low level, but it's the key to implementing a multi-modal like this. How does the action expert "pay attention" to the VLM's attention matrix? The trick is to dimension the flow-matching transformer model to match the dimensions of Gemma3 in the attention block only. The following diagram shows how the VLM and action models come together during the attention part of the transformer, and then separate again during MLP. The MLP dimension of the VLM and action expert don't need to match! This is key to unlocking a 4-5x reduction in parameter count for the action expert and greatly speeding up inference. 

During inference, the VLM is always run first and pays no attention to the action expert. Since it was trained without the action expert, it has nothing to gain from its tokens. The action expert does pay attention to the VLMs tokens though. This is done by attention masking, which is simply setting parts of the attention matrix to 0 that shouldn't attend to each other.

Reinforcement Learning
The major innovation in PI*0.6 is reinforcement learning. Until now, the models have just been trained using immitation learning - the model compares responses with reference datasets, and backpropagates the error. RL is a completely different training paradigm. The key is to reward the model when it does a good job, and train a "Value Model" to tell the VLA how close it is to achieving a reward over time. Now the model can now teach itself by watching replays, and trying out actions.

First, an entire new model is introduced to judge the value.  

Supplemental Model Info

PI0 Hyperparameters

SigLIP ViT (So400m) Architecture in Gemma 3
    • Input format: The model takes a normalized 448×448 RGB image. This is processed into a grid of 32×32 patches (not 16×16).
    • Patch embedding: Each 14×14×3 patch is linearly projected via a stride-14 conv into a 1152‑dim embedding.
    • Token count: Because the input is 448×448 and patches are 14×14, the model produces 1024 patch tokens (32×32), not 256.
    • Positional encoding: SigLIP uses learned absolute 2D positional embeddings. Note: The native pre-trained weights are usually 224×224 (256 tokens); for 448×448 inputs, Gemma 3 interpolates these embeddings to match the 1024 token sequence.
    • Class/global token: No [CLS] token is used (pool_type="none"), so the sequence length is preserved as 1024 tokens.
    • Transformer depth: The encoder is a stack of 27 transformer blocks (matching the So400m specification).
    • Self-attention specifics: Each block uses 16 attention heads. With a model width of 1152, this results in 72‑dim per head ($1152/16 = 72$) and a 1152‑dim concatenated output.
    • Attention mechanism: Each token forms Q/K/V linearly and performs full global self-attention over all 1024 tokens.
    • MLP block: After attention, each token goes through a 2-layer GELU MLP with a hidden size of 4304 (So400m specific width, approx 4× embedding dim).
    • Normalization scheme: Pre-Norm architecture. Every block applies LayerNorm → Attention → Residual, then LayerNorm → MLP → Residual.
    • Final pooling: No pooling is applied; the model outputs the full grid of patch embeddings.
    • Projection head: There is no extra projection head on the encoder output itself (though Gemma 3 may use a multimodal linear connector/projector after this stage to map to the LLM dimension).
    • Output vector: The output is a 1024×1152 tensor of patch features.
    • Training loss: (Contextual Note) While the original SigLIP was trained with sigmoid loss, Gemma 3 freezes or fine-tunes this encoder; the original loss function is not active during Gemma 3 inference.

Gemma 3 4B (Text Decoder) Architecture
    • Input format: The model takes a sequence of tokenized text IDs (using a SentencePiece tokenizer with a vocabulary of ~262,208 tokens). The context window is 128,000 tokens.
    • Token embedding: Input tokens are looked up in an embedding matrix of size 262,208×2560, producing a sequence of 2560-dimensional vectors.
    • Positional encoding: Uses Rotary Positional Embeddings (RoPE). A hybrid frequency strategy is used: a base frequency of 10,000 for local layers and 1,000,000 for global layers to support the 128k context.
    • Class/global token: No [CLS] token is used; the model is a causal decoder processing the full sequence for next-token prediction.
    • Transformer depth: The decoder is a stack of 34 transformer blocks.
    • Self-attention specifics: Each block uses 8 attention heads with a head dimension of 256. It employs Grouped-Query Attention (GQA) with 4 Key-Value heads (2:1 query-to-KV ratio). Note that the concatenated attention output (8×256 = 2048) is projected to the model's hidden size of 2560.
    • Attention mechanism: A hybrid "5:1" sliding window approach. The model alternates 5 layers of Local Sliding Window Attention (window size 1024) followed by 1 layer of Global Attention (full context).
    • MLP block: After attention, tokens pass through a Gated MLP (GeGLU) with an intermediate size of 10,240 ($\approx$4× the hidden dimension).
    • Normalization scheme: Uses RMSNorm for pre-normalization and post-normalization steps. It uniquely employs QK-Norm (normalizing Queries and Keys) instead of the soft-capping used in Gemma 2.
    • Final pooling: No pooling is applied; the model maintains the sequence of hidden states for the final prediction.
    • Projection head: The final 2560-dimensional hidden states are projected back to the vocabulary size (262,208) via a linear layer (often tied to the input embeddings).
    • Output vector: The output is a [Sequence Length × 262,208] tensor of logits representing the probability distribution for the next token.
