import json
from sgqlc.endpoint.http import HTTPEndpoint
import time
from dotenv import load_dotenv
import os
load_dotenv()
'''https://github.com/profusion/sgqlc'''

url = 'https://api.github.com/graphql'
github_access_key = os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
print (github_access_key)
headers = {'Authorization': 'bearer ' + github_access_key}
print(headers)
time.sleep(2)
query = """
query ($name_of_repository: String = "PyTorchLightning", $name: String = "pytorch-lightning") {
  repository(owner: $name_of_repository, name: $name) {
    discussions(first: 100) {
      totalCount
      nodes {
        number
      }
    }
  }
}
"""
variables = {
"name_of_repository":"PyTorchLightning",
  "name":"pytorch-lightning"
}
print(query)
endpoint = HTTPEndpoint(url, headers)
print("The endpoint was made with the following parameters:")
time.sleep(2)
data = endpoint(query , variables)
print(data["data"]["repository"]['discussions']['nodes'])
max_number = 0
for x in data["data"]["repository"]['discussions']['nodes']:
    max_number = max(max_number, x['number'])
## get the min number
query = """
query ($name_of_repository: String = "PyTorchLightning", $name: String = "pytorch-lightning") {
  repository(owner: $name_of_repository, name: $name) {
    discussions(last: 100) {
      totalCount
      nodes {
        number
      }
    }
  }
}
"""
endpoint = HTTPEndpoint(url, headers)
print("The endpoint was made with the following parameters:")
time.sleep(2)
data = endpoint(query , variables)
print(data["data"]["repository"]['discussions']['nodes'])
min_number = 2<<17
for x in data["data"]["repository"]['discussions']['nodes']:
    min_number = min(min_number, x['number'])

print(f"The min number is {min_number}")
print(f"The max number is {max_number}")
query = """
query ($name_of_repository: String = "PyTorchLightning", $name: String = "pytorch-lightning", $number: Int = 5) {
  repository(owner: $name_of_repository, name: $name) {
    discussion(number: $number) {
      id
      title
      bodyText
      createdAt
      databaseId
      number
      publishedAt
      answer {
        bodyText
        isAnswer
        lastEditedAt
        upvoteCount
        url
      }
    }
  }
}
"""
## get the dicussion in the processes
for x in range(min_number , max_number+1):
  try:
    variables = {
    "name_of_repository":"PyTorchLightning",
    "name":"pytorch-lightning",
    "number":x
    }
    time.sleep(2)
    #print("The endpoint was made with the following parameters:")
    data = endpoint(query , variables)
    print(data["data"]["repository"]["discussion"])
    os.makedirs('discussion_dataset', exist_ok=True)
    if data["data"]["repository"]["discussion"] is None:
      continue
    else:
      with open(f"discussion_dataset/dicussion_data{x}.json" , "w") as outfile:
        json.dump(data["data"]["repository"]["discussion"] , outfile)
  except:
    print("The endpoint was made with the following {x} does not work")
  
  
  