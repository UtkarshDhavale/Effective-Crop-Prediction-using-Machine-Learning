from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np


app = Flask(__name__)

model=pickle.load(open('SVM_model.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    Dictionary = {  0: 'Adzuki Beans',
                    1: 'Black gram',
                    2: 'Chickpea',
                    3: 'Coconut',
                    4: 'Coffee',
                    5: 'Cotton',
                    6: 'Ground Nut',
                    7: 'Jute',
                    8: 'Kidney Beans',
                    9: 'Lentil',
                    10: 'Moth Beans',
                    11: 'Mung Bean',
                    12: 'Peas',
                    13: 'Pigeon Peas',
                    14: 'Rubber',
                    15: 'Sugarcane',
                    16: 'Tea',
                    17: 'Tobacco',
                    18: 'Apple',
                    19: 'Banana',
                    20: 'Grapes',
                    21: 'Maize',
                    22: 'Mango',
                    23: 'Millet',
                    24: 'Muskmelon',
                    25: 'Orange',
                    26: 'Papaya',
                    27: 'Pomegranate',
                    28: 'Rice',
                    29: 'Watermelon',
                    30: 'Wheat'}
    features=[float(x) for x in request.form.values()]
    final=np.array(features)
    print(final)
    final = final.reshape(1,-1)
    prediction=(int)(model.predict(final))
    return render_template('index.html',pred = "Cultivation of {} is Suitable for your Land".format(Dictionary[prediction]))


if __name__ == '__main__':
    app.run(debug=True)
