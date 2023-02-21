from flask import Flask, jsonify, request
from utils import predict, predict_pipeline
import torch
from os.path import join, dirname
from dotenv import load_dotenv
from src.model import SphereNeuralNetwork
import os


app = Flask(__name__)

dotenv_path = join(dirname(dirname(__file__)), '.env')
load_dotenv(dotenv_path)



model = SphereNeuralNetwork()
model.load_state_dict(torch.load(join(os.getenv("S3_BINARY_FILES_DIRECTORY"), f"{os.getenv('MODEL_NAME')}.pt") ))
model.eval()


@app.post('/predict')
def predict():
    data = request.json
    try:
        sample = data['coords']
    except KeyError:
        return jsonify({'error': 'No coordinates sent'})
    
    sample = (torch.Tensor([sample]))
    predictions = predict_pipeline(model, sample)
    try:
        result = jsonify(predictions)
    except TypeError as e:
        result = jsonify({'error':str(e)})
    return result
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug= True)