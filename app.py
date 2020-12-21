from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd


app = Flask(__name__)


model = pickle.load(open("Diabetes.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    pregnancy = float(request.form['1'])
    glucose = float(request.form['2'])
    blood_pressure = float(request.form['3'])
    skin_thickness = float(request.form['4'])
    insulin = float(request.form['5'])
    bmi = float(request.form['6'])
    diabetes_pedigree_function = float(request.form['7'])
    age = float(request.form['8'])

    row_df = pd.DataFrame([pd.Series([pregnancy,glucose,blood_pressure,skin_thickness,insulin,bmi,diabetes_pedigree_function,age])])
    print(row_df)
    
    prediction = model.predict_proba(row_df)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)

    if output > str(0.5):
        return render_template('result.html', pred='You might have chance of having diabetes.\nProbability of having Diabetes is {}'.format(output))
    else:
        return render_template('result.html', pred='You are safe.\n Probability of having diabetes is {}'.format(output))



if __name__ == '__main__':
    app.run(debug=True)
