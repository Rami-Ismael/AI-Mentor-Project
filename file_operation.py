import shutil
import os

## create a folder
os.makedirs(os.path.join(os.getcwd(), "nlp_dataset_json") , exist_ok=True)

## get all file and folder in the current directory
files = os.listdir(os.getcwd())

for x in files:
    if ".json" in x:
        try:
            shutil.move(x , os.path.join(os.getcwd() , "nlp_dataset_json"))
        except Exception as e:
            os.remove(os.path.join(os.getcwd(), x))
            print(e)
            
        


