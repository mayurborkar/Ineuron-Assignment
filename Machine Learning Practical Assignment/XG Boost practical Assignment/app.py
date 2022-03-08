from flask import Flask, request, render_template
from wsgiref import simple_server
from flask_cors import cross_origin
import pickle
import os

app = Flask(__name__)

model = pickle.load(open('Scaler_Census_Data.pkl', 'rb'))
model2 = pickle.load(open('XGModel_Census_Data.pkl', 'rb'))


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
            age = int(request.form['age'])

            final_weight = int(request.form['final_weight'])

            education_num = int(request.form['education_num'])

            hours_per_week = int(request.form['hours_per_week'])

            capital_gain = int(request.form['capital_gain'])

            education = request.form['education']

            work_class = request.form['work_class']

            marital_status = request.form['marital_status']

            occupation = request.form['occupation']

            relationship = request.form['relationship']

            value = [[age, final_weight, education_num, hours_per_week, capital_gain, education, work_class,
                      marital_status, occupation, relationship]]

            scaling = model.transform(value)

            prediction = model2.predict(scaling)

            if prediction == 0:
                label = 'Less Than 50k'
            else:
                label = 'Greater Than 50k'

            return render_template('index.html', prediction_text='The Person Makes {} Per Year'.format(label))

        except Exception as e:
            raise Exception(f"(predict) - Their Is Something Wrong About Predict \n" + str(e))

    else:
        return render_template('index.html')


port = int(os.getenv("PORT", 5001))

if __name__ == "__main__":
    host = '0.0.0.0'
    # app.run()
    httpd = simple_server.make_server(host=host, port=port, app=app)
    # httpd = simple_server.make_server(host='127.0.0.1', port=5000, app=app)
    print("Serving on %s %d" % (host, port))
    httpd.serve_forever()
