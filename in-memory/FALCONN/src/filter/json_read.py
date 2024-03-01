import json

with open("/media/mydrive/ann-codes/in-memory/FALCONN/axequaltoy.json", "r") as fin:
    a = json.load(fin)
print(a["number of filters"])