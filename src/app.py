import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from flask import Flask,request,jsonify
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from flask.json import JSONEncoder
from datetime import datetime
import joblib
from flask_cors import CORS
import random

def on_click(event, x, y, flags, param):
    # Check if the mouse was actually clicked
    if event == cv.EVENT_LBUTTONDOWN:
        print(f"{x} x, {y} y")

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, datetime):
                return obj.isoformat()
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return list(iterable)
        return JSONEncoder.default(self, obj)


app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
CORS(app)

@app.route("/api/conduct",methods=["GET","POST"])
def api():
	body = request.form
	start = body.get("startDate")
	end = body.get("endDate")
	model, scaler = load_model(body.get("boya"),"conduc")

	dates = list(pd.date_range(start=start,end=end))
	df = pd.DataFrame({"date":dates,"value":""})
	df['date_number'] = df.date.astype(np.int64) // 10**9
	print(df)
	labels = df.iloc[:, [2]].values
	print(labels)
	predictions = aleatoriedad(scaler.inverse_transform(model.predict(labels)),0.077)
	print(predictions)
	df.loc[:, 'value'] = predictions
	return jsonify(df.iloc[:,:2].to_dict('r'))

@app.route("/api/pressure",methods=["GET","POST"])
def pressure():
	body = request.form
	start = body.get("startDate")
	end = body.get("endDate")
	model, scaler = load_model(body.get("boya"),"presion")

	dates = list(pd.date_range(start=start,end=end))
	df = pd.DataFrame({"date":dates,"value":""})
	df['date_number'] = df.date.astype(np.int64) // 10**9
	print(df)
	labels = df.iloc[:, [2]].values
	print(labels)
	predictions = aleatoriedad(scaler.inverse_transform(model.predict(labels)),0.253)
	print(predictions)
	df.loc[:, 'value'] = predictions
	return jsonify(df.iloc[:,:2].to_dict('r'))

@app.route("/api/temp",methods=["GET","POST"])
def temp():
	body = request.form
	start = body.get("startDate")
	end = body.get("endDate")
	model, scaler = load_model(body.get("boya"),"temperatura")

	dates = list(pd.date_range(start=start,end=end))
	df = pd.DataFrame({"date":dates,"value":""})
	df['date_number'] = df.date.astype(np.int64) // 10**9
	print(df)
	labels = df.iloc[:, [2]].values
	print(labels)
	predictions = aleatoriedad(scaler.inverse_transform(model.predict(labels)),2)
	df.loc[:, 'value'] = predictions
	return jsonify(df.iloc[:,:2].to_dict('r'))

def aleatoriedad(array,max):
	for i in range(len(array)):
		sign = random.randint(0,1)
		if sign == 0:
			array[i,0] = array[i,0] + (random.random()*max)
		else:
			array[i,0] = array[i,0] - (random.random()*max)
	return array

def load_model(boya,variable):
	if boya == "3":
		return keras.models.load_model(f"./models/escenario_1_{variable}.h5"), joblib.load(f"./scales/escala_escenario_1_{variable}.scale")
	elif boya == "7":
		return keras.models.load_model(f"./models/escenario_3_{variable}.h5"), joblib.load(f"./scales/escala_escenario_3_{variable}.scale")
	else:
		return None

if __name__ == '__main__':
	app.run(host="0.0.0.0",port=5000,debug=True)