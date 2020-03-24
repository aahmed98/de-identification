import os

def main():
    os.chdir("i2b2_evaluation_scripts/")
    # os.system("python evaluate.py phi ../../../evaluation_data/baseline-rnn/ ../../../data/raw/testing-PHI-Gold-fixed/")
    os.system("python evaluate.py phi -v ../../../evaluation_data/bi-lstm/ ../../../data/raw/testing-PHI-Gold-fixed/")
    # os.system("python -m i2b2_evaluation_scripts.evaluate phi evaluation_data/system/ evaluation_data/gold/")

if __name__ == "__main__":
    main()