import time
import json
from sgqlc.endpoint.http import HTTPEndpoint
'''https://github.com/profusion/sgqlc'''

url = 'https://api.github.com/graphql'
headers = {'Authorization': 'bearer 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'}

query = """
query ($name_of_repository: String = "PyTorchLightning", $name: String = "pytorch-lightning") {
  repository(owner: $name_of_repository, name: $name) {
    discussions(orderBy: {field: CREATED_AT, direction: DESC}, first: 100) {
      totalCount
      edges {
        node {
          id
          bodyText
          url
          answer {
            id
            bodyText
          }
        }
      }
    }
  }
}
"""
time.sleep(1)
variables = {
"name_of_repository":"PyTorchLightning",
  "name":"pytorch-lightning"
}
print(query)
endpoint = HTTPEndpoint(url, headers)
print("The endpoint was made with the following parameters:")
data = endpoint(query , variables)
print(data)
print(data.keys())
nodes = data["data"]["repository"]["discussions"]["edges"]["nodes"]
for x in range(len(nodes)):
  with open(f"discussion_answer_{x}.json" , "w") as outfile:
    json.dump(nodes[x] , outfile)
