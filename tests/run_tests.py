import os
import argparse

def main():
    os.chdir("i2b2_evaluation_scripts/")
    os.system("python evaluate.py phi ../../../evaluation_data/" + args.model + "/ ../../../data/raw/testing-PHI-Gold-fixed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type = str, help="pick model", default='baseline')
    args = parser.parse_args()
    main()