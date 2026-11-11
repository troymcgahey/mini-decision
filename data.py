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
