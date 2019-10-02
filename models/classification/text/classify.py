import argparse

parser = argparse.ArgumentParser()
parser.add_argument("train", help="training dataset in TSV format {label, text}")
parser.add_argument("test", help="training dataset in TSV format {label, text}")
args = parser.parse_args()

def evaluate(model_context, train, test):
    """
    this function is used to generate 
    """
    pass

if __name__ == "__main__":
    evaluate({}, args.train, args.test)