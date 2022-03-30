import json
from traceback import print_tb
from urllib import request
from xmlrpc import client
import requests
import sys
import os
from dotenv import load_dotenv
import time

load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")
organization_name = os.getenv("ORGANIZATION")
respository_name = os.getenv("RESPOSITORY_NAME")
print(organization_name)
print(respository_name)

## check if the organization exist on github
request = requests.get(f"https://api.github.com/orgs/{organization_name}" , auth=(client_id , client_secret) , timeout=20)
if request.status_code == 200:
    print("Organization exist")
else:
    ValueError("Organization does not exist")
time.sleep(.7)
## check if the repository exists on github organization
request = requests.get(f"https://api.github.com/orgs/{organization_name}/{respository_name}" , auth=(client_id , client_secret) , timeout=100)
if request.status_code == 200:
    print("Repository exist")
else:
    print("Repository does not exist")
## create a method that will get the number of issue a respository have
number_of_permittted_404_error = 2000
for x in range(12000 , 20000):
    try:
        time.sleep(2)
        
        #url = f"https://api.github.com/repos/{orgaorganization_name}/{respository_name}/issues/{x}"
        url = f"https://api.github.com/repos/PytorchLightning/pytorch-lightning/issues/{x}"
        response = requests.get(url , auth=(client_id , client_secret) , timeout=400)
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
                print(f"At {x} the response is 404 and the status code is {response.status_code}")
                number_of_permittted_404_error -= 1
        else:
            print(f"At {x} the response is negative and the status code is {response.status_code}")
    except:
        print(f"At {x} fault api call")
