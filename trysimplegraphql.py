from sgqlc.endpoint.http import HTTPEndpoint
'''https://github.com/profusion/sgqlc'''

url = 'https://api.github.com/graphql'
headers = {'Authorization': 'bearer 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'}

query = "query{repository(owner:\"PyTorchLightning\", name:\"pytorch-lightning\"){discussions(last:5){totalCount}}}"
#variables = {'varName': 'value'}
print(query)
endpoint = HTTPEndpoint(url, headers)
print("The endpoint was made with the following parameters:")
data = endpoint(query)
print(data)
