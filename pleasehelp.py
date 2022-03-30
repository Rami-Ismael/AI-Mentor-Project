'''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'''
'''05d239c02385adfda446f53cee763986feb9fadb'''

from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport


# Select your transport with a defined url endpoint
transport = AIOHTTPTransport(url="https://countries.trevorblades.com/")

# Create a GraphQL client using the defined transport
client = Client(transport=transport, fetch_schema_from_transport=True)

# Provide a GraphQL query
query = gql(
    """
    query getContinents {
      continents {
        code
        name
      }
    }
"""
)

# Execute the query on the transport
result = client.execute(query)
print(result)

from sgqlc.endpoint.http import HTTPEndpoint

url = 'http://server.com/graphql'
headers = {'Authorization': 'bearer TOKEN'}

query = 'query { ... }'
variables = {'varName': 'value'}

endpoint = HTTPEndpoint(url, headers)
data = endpoint(query, variables)