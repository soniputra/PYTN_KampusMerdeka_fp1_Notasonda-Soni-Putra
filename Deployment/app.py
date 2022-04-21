from flask import Flask, render_template, request, url_for
import numpy as np
import pickle

# Decision Tree Pickle
model1 = pickle.load(open("model/model_dt.pkl", "rb"))
# Linear Regression Pickle
model2 = pickle.load(open("model/model_lr.pkl", "rb"))
# Random Forest Pickle
model3 = pickle.load(open("model/model_rf.pkl", "rb"))


app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


# Decision Tree
@app.route('/predict_1', methods=['POST'])
def predict_1():
    '''
    For rendering results on HTML GUI
    '''
    features_1 = [x for x in request.form.values()]
    final_features_1 = [np.array(features_1)]
    prediction_1 = model1.predict(final_features_1)

    output_1 = round(prediction_1[0], 2)

    return render_template('index.html', prediction_text_1='Prediksi Tarif Decision Tree yaitu : $ {} dengan tingkat akurasi 95%'.format(output_1))

# Linear regression


@app.route('/predict_2', methods=['POST'])
def predict_2():
    '''
    For rendering results on HTML GUI
    '''
    features_2 = [y for y in request.form.values()]
    final_features_2 = [np.array(features_2)]
    prediction_2 = model2.predict(final_features_2)

    output_2 = round(prediction_2[0], 2)

    return render_template('index.html', prediction_text_2='Prediksi Tarif Linear Regression yaitu : $ {} dengan tingkat akurasi 52%'.format(output_2))

# Random Forest


@app.route('/predict_3', methods=['POST'])
def predict_3():
    '''
    For rendering results on HTML GUI
    '''
    features_3 = [z for z in request.form.values()]
    final_features_3 = [np.array(features_3)]
    prediction_3 = model3.predict(final_features_3)

    output_3 = round(prediction_3[0], 2)

    return render_template('index.html', prediction_text_3='Prediksi Tarif Random Forest yaitu : $ {} dengan tingkat akurasi 96%'.format(output_3))


if __name__ == '__main__':
    app.run(debug=True)
