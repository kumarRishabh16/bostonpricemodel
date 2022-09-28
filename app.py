import json
import pickle

from flask import Flask,request,app,jsonify, url_for, render_template
import numpy as np
import pandas as pd


app = Flask(__name__)

# Load the model
regmodel=pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


'''The above code is the app.py file. This is the main file that will run the application. The first thing we do is import the necessary libraries. We then load the model that we saved in the previous step. We then create a route for the home page. The home page is the page that will be displayed when the application is run. We then create a route for the prediction page. This is the page that will be displayed when the user clicks the predict button. We then get the values from the form and store them in a variable. We then pass the values to the model and get the prediction. We then display the prediction on the home page. We then run the application.

'''

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)


