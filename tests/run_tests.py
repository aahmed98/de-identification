import os

def main():
    os.chdir("i2b2_evaluation_scripts/")
    os.system("python evaluate.py phi ../../evaluation_data/baseline-rnn/ ../../evaluation_data/gold/")
    # os.system("python -m i2b2_evaluation_scripts.evaluate phi evaluation_data/system/ evaluation_data/gold/")

if __name__ == "__main__":
    main()