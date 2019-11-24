import tarfile
from zipfile import ZipFile

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

def get_data(file_name):
    pass

def process_data():
    pass

if __name__ == "__main__":
    pass

