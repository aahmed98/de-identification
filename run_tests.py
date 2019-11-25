import os

if __name__ == "__main__":
    os.chdir("i2b2_evaluation_scripts-master")
    os.system("python evaluate.py phi 110-01_test.xml 110-01.xml")