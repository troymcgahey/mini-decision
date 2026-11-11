from data import build_vocab, encode_tokens, pack_example

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
    print("IDs at decision positions:", [token_ids[i] for i in decision_positions])
    print("Unknown word:", encode_tokens(["unseen"], vocab))

if __name__ == "__main__":
    main()
