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
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

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

@app.route("/api/get",methods=["GET","POST"])
def api():
	body = request.form
	start = body.get("startDate")
	end = body.get("endDate")
	model, scaler = load_model(body.get("scenario"))
	dates = list(pd.date_range(start=start,end=end))
	df = pd.DataFrame({"date":dates,"value":""})
	df['date_number'] = df.date.astype(np.int64) // 10**9
	labels = df.iloc[:, 2]
	predictions = scaler.inverse_transform(model.predict(labels))
	df.loc[:, 'value'] = predictions
	return jsonify(df.iloc[:,:2].to_dict('r'))

def load_model(model):
	if model == "1":
		return keras.models.load_model("./models/escenario_1.h5"), joblib.load("./scales/escala_escenario_1.scale")
	elif model == "2":
		return keras.models.load_model("./models/escenario_2.h5"), joblib.load("./scales/escala_escenario_2.scale")
	elif model == "3":
		return keras.models.load_model("./models/escenario_3.h5"), joblib.load("./scales/escala_escenario_3.scale")
	elif model == "4":
		return keras.models.load_model("./models/escenario_4.h5"), joblib.load("./scales/escala_escenario_4.scale")
	else:
		return None

if __name__ == '__main__':
	app.run(host="0.0.0.0",port=5000,debug=True)