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
    discussions(orderBy: {field: CREATED_AT, direction: DESC}, first: 10) {
      totalCount
      nodes {
        url
        answer {
          bodyText
        }
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
os.makedirs('discussion_dataset', exist_ok=True)
for idx , element in enumerate( data['data']['repository']['discussions']['nodes']):
  with open(f"discussion_dataset/dicussion_data{idx}.json" , "w") as outfile:
    json.dump(element , outfile)