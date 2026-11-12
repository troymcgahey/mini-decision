import mlx.nn as nn

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size, model_dim):
        super().__init__()
        self.table = nn.Embedding(vocab_size, model_dim)

    def __call__(self, token_ids):
        return self.table(token_ids)
