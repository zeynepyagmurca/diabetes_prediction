from flask import Flask, request, render_template
import numpy as np
import joblib

from src.new_data_preprocessing import new_data_prep

app = Flask(__name__)

model = joblib.load("models/voting_clf.pkl")

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Formdan gelen verileri al
            features = [
                float(request.form['PREGNANCIES']),
                float(request.form['GLUCOSE']),
                float(request.form['BLOODPRESSURE']),
                float(request.form['SKINTHICKNESS']),
                float(request.form['INSULIN']),
                float(request.form['BMI']),
                float(request.form['DIABETESPEDIGREEFUNCTION']),
                float(request.form['AGE'])
            ]

            df_processed = new_data_prep(features)
            prediction = model.predict(df_processed)[0]
            return render_template('predict.html', prediction=prediction)
        except Exception as e:
            return render_template('predict.html', prediction=f"Hata: {str(e)}")
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Portu değiştirebilirsin, örn: 5001