import json


f = open('train_annotations/000.json')

data = json.load(f)


for i in data['sequence']:
    print(i['TgtXPos_LeftUp'])


f.close()