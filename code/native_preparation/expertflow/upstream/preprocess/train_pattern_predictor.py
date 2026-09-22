"""Generic ExpertFlow routing-path predictor trainer."""

try:
    from predictor_training import train
except ImportError:
    from preprocess.predictor_training import train


if __name__ == "__main__":
    train()
