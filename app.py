import os
from flask import Flask, render_template, send_file, request
import gcp
import summary
import prac
import prediction_pipeline
#import datab

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analysis',methods=['POST'])
def getAudio():
    f = request.files['audio_file']
    print('File uploaded successfully')

    with open('Audio/SRH Hochschule Heidelberg 1.wav','wb') as audio:
        f.save(audio)

    f = open("minutesOM/SRH Hochschule Heidelberg 1Text.txt","r")        
    originalText = f.read()
    f.close()

    topic = prac.topics() 
    abssummary = summary.absSummary()
    text = gcp.getOriginalText()
    prediction = prediction_pipeline.get_predicted_text()
    
    #datab.data()  

    return render_template('summary.html',original_text = text,topic = topic, summary = abssummary,prediction = prediction )

if __name__ == '__main__':
    app.run(debug= True)