import json,numpy as np
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
import threading
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pandas as pd
authenticator = IAMAuthenticator('dFS-HvfZ_m-ZYupgDyDbfcZrxEuiG0zv2eZgTT8gPQVG')
service = SpeechToTextV1(authenticator=authenticator)
service.set_service_url('https://api.eu-de.speech-to-text.watson.cloud.ibm.com/instances/a7ca155d-bf1b-497d-a8b7-fc5eda16e5e1')

models = service.list_models().get_result()
#print(json.dumps(models, indent=2))

model = service.get_model('en-US_BroadbandModel').get_result()
#print(json.dumps(model, indent=2))

with open(join(dirname('__file__'), 'C:/Users/chait/Desktop/Casestudy/Audio/audio.wav'),
          'rb') as audio_file:

#    print(json.dumps(
    output = service.recognize(
    audio=audio_file,
    speaker_labels=True,
    content_type='audio/wav',
    #timestamps=True,
    #word_confidence=True,
    model='en-US_NarrowbandModel',
    continuous=True).get_result(),
    indent=3
l1=[] 

for i in output[0]['speaker_labels']:
    l1.append(i['speaker'])

v = l1[0]  

for idx,value in enumerate(l1[1:],1):   # we do all the others starting at index 1
    if l1[idx] == v:
        #print(idx)
        l1[idx] = 9
        #l1.pop(idx)                   # replace list element
    else:
        v = l1[idx]                     # or replace v if different

    #l = int(''.join(l1))               # only create one string & convert to int
lis1 = np.array(l1)

print(lis1[lis1<8])
#print(list_1)


#print(i[0]['speaker_labels'][0])  
#print(type(output[0]['speaker_labels'])) 

df = pd.DataFrame([i for elts in output for alts in elts['results'] for i in alts['alternatives']])
print(df)
#for i in df['timestamps']:
#    print(i[0][1])

#print(df['time_updated_start'])


#print(df['timestamps'][0][0])
#print((df['timestamps'][0][0][1]))
#print((df['timestamps'][0][0][len(df['timestamps'][0][0])-1]))

#print(len(df['timestamps'][0][0]))
#df1 = pd.DataFrame([i for elts in output for i in elts['speaker_labels']])


#print(df)
#print(df1)