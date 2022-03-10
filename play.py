import os
from plistlib import load
from dotenv import load_dotenv
import requests
load_dotenv()

secret_key = os.getenv("CLIENT_SECRET")

print(secret_key)