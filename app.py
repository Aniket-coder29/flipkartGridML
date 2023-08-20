from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import io
import os

# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans


#
# from requests_toolbelt.multipart import decoder
import logging
from logging.handlers import RotatingFileHandler


logger = logging.getLogger()

logFormatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

fileLogHandler = RotatingFileHandler("logs.log",backupCount=800,maxBytes=10240)
fileLogHandler.setFormatter(logFormatter)
logger.addHandler(fileLogHandler)
#

app = Flask(__name__)

def print_cluster(i,order_centroids,terms):
    recommendations = []
    # print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        # print(' %s' % terms[ind])
        recommendations.append(terms[ind])
    # print
    return recommendations

def show_recommendations(product,model,vectorizer,order_centroids,terms):
    #print("Cluster ID:")
    Y = vectorizer.transform([product])
    prediction = model.predict(Y)
    #print(prediction)
    return print_cluster(prediction[0],order_centroids,terms)

@app.route("/getRecommendation")
def give_recommendation():

    product= request.args.get('search')
    modelfiles = []
    for file in os.listdir(f'./models/'):
        isFile = os.path.isfile(f'./models/{file}')
        if isFile:
            print("cnfrm h ki file h")
            print(file)
            modelfiles.append(file)
    
    for model_file in modelfiles:
        if model_file == 'vectorizer.pkl':
            with io.open(f'./models/{model_file}','rb') as f:
                vectorizer =  pickle.load(f)
        elif model_file == 'recommender.pkl':
            with io.open(f'./models/{model_file}','rb') as f:
                model =  pickle.load(f)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    return show_recommendations(product,model,vectorizer,order_centroids,terms)

@app.route("/")
def index():
    return "hello world"

if __name__ == '__main__':
    app.run(debug=True, port=8080)