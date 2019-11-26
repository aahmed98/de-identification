import os

def main():
    os.chdir("../i2b2_evaluation_scripts-master")
    os.system("python evaluate.py phi ../evaluation_data/system/ ../evaluation_data/gold/")

if __name__ == "__main__":
    main()