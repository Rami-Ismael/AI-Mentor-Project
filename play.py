import pickle
name = {"name":"rami"}
print(name)
print(name["name"])
print(type(name))
with open("myDictionary.pkl", "wb") as tf:
    new_dict = pickle.load(tf)

print(new_dict)