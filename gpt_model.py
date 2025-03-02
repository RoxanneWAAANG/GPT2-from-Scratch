import numpy as np

# ----------------------------
# Helper Classes and Functions
# ----------------------------

class LayerNorm:
    """
    A simple layer normalization module.
    Normalizes the input across the last dimension.
    """
    def __init__(self, model_dim, eps=1e-5):
        self.eps = eps
        # Gamma (scale) is initialized to ones and beta (shift) to zeros.
        self.gamma = np.ones((model_dim,))
        self.beta = np.zeros((model_dim,))

    def forward(self, x):
        """
        Forward pass for layer normalization.
        x: numpy array of shape (batch, seq_length, model_dim)
        """
        # Compute mean and variance across the model dimension.
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        # Normalize the input.
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        # Scale and shift.
        return self.gamma * x_norm + self.beta

class FeedForward:
    """
    A simple two-layer feed-forward network with a ReLU activation.
    """
    def __init__(self, model_dim, hidden_dim):
        # Weight initialization with scaling
        self.W1 = np.random.randn(model_dim, hidden_dim) / np.sqrt(model_dim)
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(hidden_dim, model_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((model_dim,))

    def forward(self, x):
        """
        x: numpy array of shape (batch, seq_length, model_dim)
        """
        # First linear layer: use tensordot to operate on the last axis.
        hidden = np.tensordot(x, self.W1, axes=([-1],[0])) + self.b1
        # Apply ReLU activation.
        hidden = np.maximum(0, hidden)
        # Second linear layer.
        output = np.tensordot(hidden, self.W2, axes=([-1],[0])) + self.b2
        return output

class MultiHeadSelfAttention:
    """
    Implements multi-head self-attention with causal (masked) attention.
    """
    def __init__(self, model_dim, num_heads):
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        # Initialize projection weights for Q, K, V.
        self.Wq = np.random.randn(model_dim, model_dim) / np.sqrt(model_dim)
        self.bq = np.zeros((model_dim,))
        self.Wk = np.random.randn(model_dim, model_dim) / np.sqrt(model_dim)
        self.bk = np.zeros((model_dim,))
        self.Wv = np.random.randn(model_dim, model_dim) / np.sqrt(model_dim)
        self.bv = np.zeros((model_dim,))
        # Output projection weight.
        self.Wo = np.random.randn(model_dim, model_dim) / np.sqrt(model_dim)
        self.bo = np.zeros((model_dim,))

    def forward(self, x):
        """
        x: numpy array of shape (batch, seq_length, model_dim)
        Returns: output of same shape as x.
        """
        batch, seq_length, _ = x.shape

        # Linear projections to get Q, K, V.
        Q = np.dot(x, self.Wq) + self.bq  # (batch, seq_length, model_dim)
        K = np.dot(x, self.Wk) + self.bk
        V = np.dot(x, self.Wv) + self.bv

        # Reshape and transpose to separate heads.
        # New shape: (batch, seq_length, num_heads, head_dim)
        Q = Q.reshape(batch, seq_length, self.num_heads, self.head_dim)
        K = K.reshape(batch, seq_length, self.num_heads, self.head_dim)
        V = V.reshape(batch, seq_length, self.num_heads, self.head_dim)
        # Transpose to (batch, num_heads, seq_length, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Calculate attention scores.
        # scores shape: (batch, num_heads, seq_length, seq_length)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Create a causal mask so that position i can only attend to positions <= i.
        # mask: a boolean array of shape (seq_length, seq_length)
        mask = np.triu(np.ones((seq_length, seq_length), dtype=bool), k=1)
        # Expand mask dimensions to match scores shape and apply it.
        scores = np.where(mask[None, None, :, :], -1e9, scores)

        # Compute softmax over the last axis (the key dimension).
        # For numerical stability, subtract the max from scores.
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Compute the weighted sum of V.
        context = np.matmul(attention_weights, V)  # (batch, num_heads, seq_length, head_dim)
        # Reassemble heads: transpose and reshape back to (batch, seq_length, model_dim)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_length, self.model_dim)

        # Final linear projection.
        output = np.dot(context, self.Wo) + self.bo
        return output

# ----------------------------
# GPT Block and Model Classes
# ----------------------------

class GPTBlock:
    """
    A single Transformer block used in GPT.
    Consists of:
      1. LayerNorm
      2. Masked Multi-Head Self-Attention with residual connection
      3. Another LayerNorm
      4. Feed-Forward network with residual connection
    """
    def __init__(self, model_dim, num_heads, ff_hidden_dim):
        self.ln1 = LayerNorm(model_dim)
        self.mha = MultiHeadSelfAttention(model_dim, num_heads)
        self.ln2 = LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, ff_hidden_dim)

    def forward(self, x):
        # First, apply layer norm then self-attention, and add the residual.
        attn_output = self.mha.forward(self.ln1.forward(x))
        x = x + attn_output

        # Next, apply layer norm then feed-forward network, and add the residual.
        ff_output = self.ff.forward(self.ln2.forward(x))
        x = x + ff_output

        return x

class GPT:
    """
    A GPT (decoder-only) Transformer model.
    """
    def __init__(self, vocab_size, max_seq_length, model_dim, num_heads, ff_hidden_dim, num_layers):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.model_dim = model_dim

        # Token embeddings: shape (vocab_size, model_dim)
        self.token_embedding = np.random.randn(vocab_size, model_dim) / np.sqrt(model_dim)
        # Positional embeddings: shape (max_seq_length, model_dim)
        self.position_embedding = np.random.randn(max_seq_length, model_dim) / np.sqrt(model_dim)

        # Build a list of GPT blocks.
        self.layers = [GPTBlock(model_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)]
        # Final layer normalization.
        self.ln_f = LayerNorm(model_dim)
        # For generating logits, we tie the token embedding weights with the output projection.

    def forward(self, input_ids):
        """
        Performs a forward pass of the model.
        input_ids: numpy array of shape (batch, seq_length) with integer token IDs.
        Returns: logits of shape (batch, seq_length, vocab_size)
        """
        batch, seq_length = input_ids.shape
        if seq_length > self.max_seq_length:
            raise ValueError("Sequence length exceeds the model's maximum sequence length.")

        # Look up token embeddings.
        # x: (batch, seq_length, model_dim)
        x = self.token_embedding[input_ids]
        # Add positional embeddings (broadcasted over batch).
        x = x + self.position_embedding[:seq_length, :]

        # Pass through each Transformer block.
        for layer in self.layers:
            x = layer.forward(x)

        # Apply final layer normalization.
        x = self.ln_f.forward(x)
        # Compute logits by a linear projection using the (tied) token embeddings.
        # logits shape: (batch, seq_length, vocab_size)
        logits = np.dot(x, self.token_embedding.T)
        return logits

# ----------------------------
# Example Usage
# ----------------------------

if __name__ == "__main__":
    # Define hyperparameters.
    vocab_size = 10000       # Vocabulary size.
    max_seq_length = 128     # Maximum sequence length.
    model_dim = 512          # Model (hidden) dimensionality.
    num_heads = 8            # Number of attention heads.
    ff_hidden_dim = 2048     # Hidden dimensionality in feed-forward network.
    num_layers = 6           # Number of Transformer blocks.

    # Create an instance of the GPT model.
    model = GPT(vocab_size, max_seq_length, model_dim, num_heads, ff_hidden_dim, num_layers)

    # Create a dummy input (batch_size x sequence_length) with random token IDs.
    batch_size = 2
    seq_length = 20
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_length))

    # Run the forward pass to obtain logits.
    logits = model.forward(input_ids)

    # Logits shape should be (batch_size, seq_length, vocab_size)
    print("Logits shape:", logits.shape)
