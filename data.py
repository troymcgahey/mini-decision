def pack_example(message, options):
    tokens = message.lower().split()
    tokens.append("[SEP]")

    decision_positions = []

    for option in options:
        decision_positions.append(len(tokens))
        tokens.append("[DECIDE]")
        tokens.extend(option.lower().split())
        tokens.append("[SEP]")

    return tokens, decision_positions

def build_vocab(token_sequences):
    vocab = {"[PAD]": 0, "[UNK]":1, "[SEP]": 2, "[DECIDE]": 3}

    for tokens in token_sequences:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab

def encode_tokens(tokens, vocab):
    unknown_id = vocab["[UNK]"]
    return [vocab.get(token, unknown_id) for token in tokens]
