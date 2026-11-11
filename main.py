from data import pack_example

def main():
    message = "Where is my parcel"
    options = ["shipping", "billing", "account access"]

    tokens, decision_positions = pack_example(message, options)

    for position, token in enumerate(tokens):
        print(f"{position:2d}   {token}")

    print("Decision positions:", decision_positions)

if __name__ == "__main__":
    main()
