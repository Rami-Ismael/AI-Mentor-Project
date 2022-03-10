import json
from xmlrpc import client
import requests
import sys
import os
from dotenv import load_dotenv
import time

load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
## create a method that will get the number of issue a respository have
number_of_permittted_404_error = 25
for x in range(1 , 20000):
    try:
        time.sleep(.7)
        url = f"https://api.github.com/repos/PyTorchLightning/pytorch-lightning/issues/{x}"
        response = requests.get(url , auth=(client_id , client_secret) , timeout=20)
        if response.status_code == 200:
            ## convert the response to json
            response = response.json()
            ## save the json file a specfici file format
            with open(f"data_{x}.json" , "w") as outfile:
                json.dump(response , outfile)
            print(f"At {x} success api call")
        elif response.status_code == 404:
            print(f"At {x} the response is 404 and the status code is {response.status_code}")
            if number_of_permittted_404_error == 0:
                print("Exceeded the number of permittted 404 error")
                break
            else:
                number_of_permittted_404_error -= 1
        else:
            print(f"At {x} the response is negative and the status code is {response.status_code}")
    except:
        print(f"At {x} fault api call")