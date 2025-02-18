import argparse

def argument_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--model_fn", required=True, help="Model_File_Name")

    config = p.parse_args()

    return config

def main(config):
    # Load Data

    # Define Model

    # Train

    # Save Model
    pass

if __name__ == "__main__":
    config = argument_parser()
    main(config)
