from flask import Flask, request, url_for, redirect, render_template
import pickle
import pandas as pd

app = Flask(__name__)

data = pd.read_csv("diabetes.csv")
attributes = data.drop("Outcome", axis=1).columns
model = pickle.load(open("Diabetes.pkl", "rb"))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    row_dict = {}
    for attribute in attributes:
        temp = float(request.form[attribute])
        row_dict[attribute] = (
            temp - data[attribute].mean()) / data[attribute].std()

    # print(row_df)
    #row_df = pd.DataFrame(row_dict)
    # = pd.DataFrame(row_dict)
    row_df = pd.Series(row_dict).to_frame()
    print(row_df)
    prediction = model.predict_proba(row_df)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)

    if output > str(0.5):
        return render_template('result.html', pred='You might have chance of having diabetes.\n Probability of having Diabetes is {}'.format(output))
    else:
        return render_template('result.html', pred='You are safe.\n Probability of having diabetes is {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
