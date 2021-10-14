from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle

app = Flask(__name__)

model = pickle.load(open('Scaler.pkl', 'rb'))
model2 = pickle.load(open('Titanic.pkl', 'rb'))


@app.route("/", methods=['GET', 'POST'])
@cross_origin()
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        raise Exception(f"(home) - Could Not Find The index.html Page \n" + str(e))


@app.route("/report", methods=['GET', 'POST'])
@cross_origin()
def report():
    try:
        return render_template('Report.html')
    except Exception as e:
        raise Exception(f"(report) - Could Not Find The affaris_report.html Page \n" + str(e))


@app.route("/predict", methods=['POST'])
@cross_origin()
def predict():

    if request.method == "POST":

        try:
            Pclass = request.form['Pclass']

            Age = int(request.form['Age'])

            SibSp = request.form['SibSp']

            Parch = request.form['Parch']

            Fare = int(request.form['Fare'])

            Sex_Type = request.form['Sex_Type']

            value = model.transform([[Pclass, Age, SibSp, Parch, Fare, Sex_Type]])

            prediction = model2.predict(value)

            if prediction[0] == 0:
                result = 'No One Is Survived'
            else:
                result = 'Only One Is Survived'
            
            return render_template('index.html', prediction_text=result)

        except Exception as e:
            raise Exception(f"(predict) - Their Is Something Wrong About Predict \n" + str(e))

    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
