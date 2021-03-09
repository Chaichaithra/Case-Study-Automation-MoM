filepath = '/Users/chaithras/Desktop/projects/Case_Study/Audio_input/'
output_filepath = '/Users/chaithras/Desktop/projects/Case_Study/Output/'

bucketname = "sapstudy"

#import libraries
from pydub import AudioSegment
import io
import os
#from google.cloud import speech_v1, speech
from google.cloud import speech_v1p1beta1 as speech
#from google.cloud.speech_v1p1beta1 import enums #Changed
#from google.cloud.speech_v1p1beta1 import types #Changed
#from google.cloud import speech_v1, speech
import wave
from google.cloud import storage

def mp3_to_wav(audio_file_name):
    if audio_file_name.split('.')[1] == 'mp3':    
        sound = AudioSegment.from_mp3(audio_file_name)
        audio_file_name = audio_file_name.split('.')[0] + '.wav'
        sound.export(audio_file_name, format="wav")

def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")


def frame_rate_channel(audio_file_name):
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        return frame_rate,channels

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()


def google_transcribe(audio_file_name):
    
    file_name = filepath + audio_file_name
    
    # The name of the audio file to transcribe
    
    frame_rate, channels = frame_rate_channel(file_name)
    
    if channels > 1:
        stereo_to_mono(file_name)
    
    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name
    
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''
    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)

    config = speech.RecognitionConfig(
    encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz = frame_rate,
    language_code = 'en-US',
    enable_speaker_diarization=True,
    diarization_speaker_count = 3)

    # Detects speech in the audio file
    operation = client.long_running_recognize(config = config,audio = audio)
    response = operation.result(timeout=1000000000)
    result = response.results[-1] #Changed
    words_info = result.alternatives[0].words #Changed
    
    tag=1 #Changed
    speaker="" #Changed

    for word_info in words_info: #Changed
        if word_info.speaker_tag==tag: #Changed
            speaker=speaker+" "+word_info.word #Changed
        else: #Changed
            transcript += "speaker {}: {}".format(tag,speaker) + '\n' #Changed
            tag=word_info.speaker_tag #Changed
            speaker=""+word_info.word #Changed
 
    transcript += "speaker {}: {}".format(tag,speaker) #Changed
    
    delete_blob(bucket_name, destination_blob_name)
    return transcript

def write_transcripts(transcript_filename,transcript):
    f= open(output_filepath + transcript_filename,"w+")
    f.write(transcript)
    f.close()

if __name__ == "__main__":
    for audio_file_name in os.listdir(filepath):
        #print(audio_file_name)
        transcript = google_transcribe(audio_file_name)
        transcript_filename = audio_file_name.split('.')[0] + '.txt'
        write_transcripts(transcript_filename,transcript)