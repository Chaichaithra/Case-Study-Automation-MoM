import pandas as pd
import os
import spacy
from pycaret.nlp import *

with open("/Users/chaithras/Desktop/projects/Case_Study/Output/SRH Hochschule Heidelberg.txt") as text_file:
     sentences = text_file.readlines()

data= pd.DataFrame(sentences)
data=data.rename(columns={0:'discussion'})
print(data)

nlp1=setup(data,target='discussion')

lda=create_model('lda')

print(lda)

df_lda=assign_model(lda)

print(df_lda)

plot_model(lda,plot='wordcloud',topic_num='Topic 1')

# plot_model(lda,plot='topic_model')