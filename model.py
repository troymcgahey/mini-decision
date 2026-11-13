import mlx.core as mx
import mlx.nn as nn

class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, model_dim, max_length=64):
        super().__init__()
        self.table = nn.Embedding(vocab_size, model_dim)
        self.position_table = nn.Embedding(max_length, model_dim)
        self.max_length = max_length

    def __call__(self, token_ids):
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.max_length:
            raise ValueError(f"Input exceeds {self.max_length} tokens")

        position_ids = mx.arange(sequence_length)
        token_vectors = self.table(token_ids)
        position_vectors = self.position_table(position_ids)

        return token_vectors + position_vectors

class SelfAttention(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.scale = model_dim ** -0.5

    def __call__(self, hidden):
        queries = self.query(hidden)
        keys = self.key(hidden)
        values = self.value(hidden)

        scores = (queries @ keys.T) * self.scale
        weights = mx.softmax(scores, axis=1)
        
        return weights @ values
