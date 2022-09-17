import os
import argparse

def main():
    os.chdir("i2b2_evaluation_scripts/")
    os.system("python evaluate.py phi ../../../evaluation_data/" + args.model + "/ ../../../de-ID_data/raw/testing-PHI-Gold-fixed/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type = str, help="pick model", default='bi-lstm-crf')
    args = parser.parse_args()
    main()