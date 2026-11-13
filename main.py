from data import build_vocab, encode_tokens, pack_example
from model import SelfAttention, TokenEmbedding

import mlx.core as mx

from model import TokenEmbedding

def main():
    message = "Where is my parcel"
    options = ["shipping", "billing", "account access"]

    tokens, decision_positions = pack_example(message, options)

    for position, token in enumerate(tokens):
        print(f"{position:2d}   {token}")

    print("Decision positions:", decision_positions)

    vocab = build_vocab([tokens])
    token_ids = encode_tokens(tokens, vocab)

    embeddings = TokenEmbedding(vocab_size=len(vocab), model_dim=8)
    hidden = embeddings(mx.array(token_ids))

    attention = SelfAttention(model_dim=8)
    contextual = attention(hidden)

    print("Attention output shape:", contextual.shape)
    for marker_position in decision_positions:
        print(f"Contextual marker at {marker_position}:",
            contextual[marker_position].tolist())

    print("Embedding shape:", hidden.shape)

    for marker_position in decision_positions:
        vector = hidden[marker_position]
        print(f"Marker at {marker_position}:", vector.tolist())

    print("Token IDs:", token_ids)
    print("IDs at decision positions:", [token_ids[i] for i in decision_positions])
    print("Unknown word:", encode_tokens(["unseen"], vocab))

if __name__ == "__main__":
    main()
