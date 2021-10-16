from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)

model = pickle.load(open('Scaler.pkl', 'rb'))
model2 = pickle.load(open('Random_Forest.pkl', 'rb'))


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        raise Exception(f"(home) - Could not find the index.html Page \n" + str(e))


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():

    if request.method == "POST":
        try:
            CRIM = float(request.form['CRIM'])
            ZN = float(request.form['ZN'])
            INDUS = float(request.form['INDUS'])
            CHAS = float(request.form['CHAS'])
            NOX = float(request.form['NOX'])
            RM = float(request.form['RM'])
            AGE = float(request.form['AGE'])
            DIS = float(request.form['DIS'])
            RAD = float(request.form['RAD'])
            TAX = float(request.form['TAX'])
            PTRATIO = float(request.form['PTRATIO'])
            B = float(request.form['B'])
            LSTAT = float(request.form['LSTAT'])

            value = model.transform([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

            prediction = model.predict(value)

            return render_template('index.html', prediction_text="The Boston Housing Price IS {}".format(prediction))

        except Exception as e:
            raise Exception(f"(predict) - Their Is Something Wrong About Predict \n" + str(e))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
