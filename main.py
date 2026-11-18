from data import build_vocab, encode_tokens, pack_example
from model import DecisionModel
from itertools import permutations

import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
import random

def loss_fn(model, token_ids, marker_indices, target):
    logits = model(token_ids, marker_indices)
    return nn.losses.cross_entropy(
            logits[None, :], target, reduction="mean"
    )

def main():

    mx.random.seed(42)
    rng = random.Random(42)

    labels = ["shipping", "billing", "account access"]
    training_rows = [
        ("where is my parcel", "shipping"),
        ("my parcel is late", "shipping"),
        ("where is my refund", "billing"),
        ("my refund is late", "billing"),
        ("help me reset my password", "account access"),
        ("my password does not work", "account access"),
    ]

    vocab = build_vocab([
        pack_example(message, labels)[0]
        for message, _ in training_rows
    ])

    model = DecisionModel(vocab_size=len(vocab), model_dim=8)
    optimizer = optim.Adam(learning_rate=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(1, 101):
        rng.shuffle(training_rows)
        total_loss = 0.0

        for message, correct_label in training_rows:
            options = rng.sample(labels, len(labels))
            tokens, positions = pack_example(message, options)

            token_ids = mx.array(encode_tokens(tokens, vocab))
            marker_indices = mx.array(positions)
            target = mx.array([options.index(correct_label)])

            loss, grads = loss_and_grad(model, token_ids, marker_indices, target)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            total_loss += loss.item()

        if epoch == 1 or epoch % 20 == 0:
            mean_loss = total_loss / len(training_rows)
            print(f"Epoch {epoch:3d} | mean loss: {mean_loss:4f}")

    options = ["billing", "account access", "shipping"]
    message = "my parcel does not work"
    tokens, positions = pack_example(message, options)
    logits = model(
        mx.array(encode_tokens(tokens, vocab)),
        mx.array(positions),
    )

    total_correct = 0
    total_checks = 0

    print("\nTraining accuracy across candidate orders:")

    for message, correct_label in training_rows:
        correct = 0

        for options in permutations(labels):
            tokens, positions = pack_example(message, options)
            logits = model(
                mx.array(encode_tokens(tokens, vocab)),
                mx.array(positions),
            )

            predicted_index = logits.argmax().item()
            predicted_label = options[predicted_index]
            correct += int(predicted_label == correct_label)

        total_correct != correct
        total_checks += 6
        print(f"{correct}/6 {message}")

    print(f"Total: {total_correct}/{total_checks}")

    print(f"\nHeld-out message: {message}")
    for option, probability in zip(options, mx.softmax(logits, axis=-1).tolist()):
        print(f"{option:14s} {probability:.4f}")

#    message = "Where is my parcel"
#    options = ["shipping", "billing", "account access"]
#
#    tokens, decision_positions = pack_example(message, options)
#
#    for position, token in enumerate(tokens):
#        print(f"{position:2d}   {token}")
#
#    vocab = build_vocab([tokens])
#
#    token_ids = mx.array(encode_tokens(tokens, vocab))
#
#    print("Token_ids:", token_ids)
#
#    marker_indices = mx.array(decision_positions)
#    target = mx.array([options.index("shipping")])
#
#    model = DecisionModel(vocab_size=len(vocab), model_dim=8)
#    optimizer = optim.Adam(learning_rate=0.01)
#    loss_and_grad = nn.value_and_grad(model, loss_fn)
#
#    for step in range(1, 101):
#        loss, grads = loss_and_grad(model, token_ids, marker_indices, target)
#        optimizer.update(model, grads)
#        mx.eval(loss, model.parameters(), optimizer.state)
#
#        if step == 1 or step % 20 == 0:
#          print(f"Step {step:3d} | loss before update: {loss.item():.4f}")
#
#    logits = model(token_ids, marker_indices)
#    probabilities = mx.softmax(logits, axis=-1)
#    for option, probability in zip(options, probabilities.tolist()):
#          print(f"{option:14s} {probability:.4f}")

if __name__ == "__main__":
    main()
