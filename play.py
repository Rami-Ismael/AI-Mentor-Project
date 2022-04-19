import pickle
name = {"name":"rami"}
import json
print(name)
print(name["name"])
print(type(name))
tf = open("myDictionary.json", "r")
new_dict = json.load(tf)
print(new_dict)