import mlx.core as mx

def main():
    scores = mx.array([1.0, 2.0, 3.0])
    probabilities = mx.softmax(scores, axis=0)
    print(probabilities.tolist())


if __name__ == "__main__":
    main()
