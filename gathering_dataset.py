import os
import re
import json
import pandas as pd

from ryu.lib import mrtlib

### Dataset 1 - Dialogue summary dataset

data_path = 'data/Dialogue_summary_dataset/'
#data_path = 'C://Users//chait//Desktop//Casestudy//data//Dialogue_summary_dataset//'

test_filepath = os.path.join(data_path, 'test.json')
#print(test_filepath)
train_filepath = os.path.join(data_path, 'train.json')
val_filepath = os.path.join(data_path, 'val.json')

with open(test_filepath, encoding="utf8") as filepath:
    df_test = pd.DataFrame(json.load(filepath))

with open(train_filepath, encoding="utf8") as filepath:
    df_train = pd.DataFrame(json.load(filepath))
    
with open(val_filepath,encoding="utf8") as filepath:
    df_val = pd.DataFrame(json.load(filepath))
    
df1 = pd.concat([df_test, df_train, df_val]).rename(columns={'summary': 'sentences'})
df1['type'] = 'dataset1'

print(df1.head())


### Dataset 2 - ICSI Dataset

data_path = 'data/ICSI_dataset/transcripts'

regex = re.compile(".*?\<(.*?)\>")
sentences = []

for filepath in os.listdir(data_path):
    with open(os.path.join(data_path, filepath), 'rb') as filepath:
        for x in mrtlib.Reader(filepath):
            data = x  
    
    for dialogue in str(data).split('<Segment')[1:]:
        dialogue = dialogue.split('</Segment>')[0].split('    ')[1].replace('\n', '')
        for i in re.findall(regex, dialogue):
            dialogue = dialogue.replace("<" + i + ">", '')
        dialogue = dialogue[2:-2]
        
        if dialogue != '':
            sentences.append(dialogue)

df2 = pd.DataFrame(sentences).rename(columns={0: 'sentences'})         
df2['type'] = 'dataset2'

print ("Total size of Dataset2:", len(df2))
print(df2.head())

## Combine and save all dialogues data

df = pd.concat([df1, df2])
df['id'] = [i for i in range(len(df))]

print ("Total size of Dataset:", len(df))

df.type.value_counts()

df.to_csv("all_dialogues.csv", index=False)