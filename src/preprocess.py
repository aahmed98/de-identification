import tarfile
from zipfile import ZipFile
import os
import xml.etree.ElementTree as ET

def extract_file(file_path:str):
    if (file_path.endswith('.tar.gz') or file_path.endswith('.tgz')):
        try:
            tar = tarfile.open(file_path)
            tar.extractall()
            tar.close()
        except:
            print("Error in extracting tar file")
    elif (file_path.endswith('.zip')):
        try:
            with ZipFile(file_path,'r') as zipObj:
                zipObj.extractall()
        except:
            print("Error in extracting zip file")

def get_data(dir_name: str):
    print("Fetching data...")
    for filename in os.listdir(dir_name):
        tree = ET.parse(filename)

def process_data():
    pass

if __name__ == "__main__":
    print("preprocessing...")

