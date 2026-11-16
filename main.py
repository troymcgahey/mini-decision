from data import build_vocab, encode_tokens, pack_example
from model import DecisionHead, SelfAttention, TokenEmbedding
import mlx.nn as nn

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

    print("Token IDs:", token_ids)
    print("IDs at decision positions", [token_ids[i] for i in decision_positions])
    print("Unknown word:", encode_tokens(["unseen"], vocab))

    embedding = TokenEmbedding(vocab_size=len(vocab), model_dim=8)
    hidden = embedding(mx.array(token_ids))

    print("Embedding shape:", hidden.shape)
    for marker_position in decision_positions:
        vector = hidden[marker_position]
        print(f"Marker at {marker_position}:", vector.tolist())

    attention = SelfAttention(model_dim=8)
    contextual = attention(hidden)

    head = DecisionHead(model_dim=8)
    marker_indices = mx.array(decision_positions)
    logits = head(contextual, marker_indices)
    probabilities = mx.softmax(logits, axis=-1)

    correct_option = "shipping"
    target_index = options.index(correct_option)
    target = mx.array([target_index])

    loss = nn.losses.cross_entropy(
        logits[None, :],
        target,
        reduction="mean",
    )

    print(f"Target: {correct_option} (index {target_index})")
    print(f"Loss: {loss.item():.4f}")

    print("Logits shape:", logits.shape)
    for option, logit, probability in zip(
        options, logits.tolist(), probabilities.tolist()
    ):
        print(f"{option:14s} logit={logit:+.4f} probability={probability:.4f}")

    print("Attention output shape:", contextual.shape)
    for marker_position in decision_positions:
        print(f"Contextual marker at {marker_position}:",
            contextual[marker_position].tolist())

    first_marker = decision_positions[0]
    edited_ids = token_ids.copy()
    edited_ids[first_marker + 1] = vocab["billing"]

    edited_hidden = embedding(mx.array(edited_ids))
    edited_contextual = attention(edited_hidden)

    before = hidden[first_marker] - edited_hidden[first_marker]
    after = contextual[first_marker] - edited_contextual[first_marker]

    print("Marker change before attention:", mx.max(mx.abs(before)).item())
    print("Marker change after attention:", mx.max(mx.abs(after)).item())

if __name__ == "__main__":
    main()
