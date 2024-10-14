from flask import Flask, request, render_template
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained model
model = load('ufc_predictor_model.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect input data from the form
            data = {
                'RedWinsByDecisionMajority': int(request.form['RedWinsByDecisionMajority']),
                'RedWinsByTKODoctorStoppage': int(request.form['RedWinsByTKODoctorStoppage']),
                'WinDif': float(request.form['WinDif']),
                'KODif': float(request.form['KODif']),
                'RedTotalRoundsFought': int(request.form['RedTotalRoundsFought']),
                'RedWins': int(request.form['RedWins']),
                'RWelterweightRank': int(request.form['RWelterweightRank']),
                'RedAvgTDPct': float(request.form['RedAvgTDPct']),
                'BSubOdds': float(request.form['BSubOdds']),
                'BlueWinsByKO': int(request.form['BlueWinsByKO']),
                'BlueWinsByDecisionMajority': int(request.form['BlueWinsByDecisionMajority']),
                'BlueWins': int(request.form['BlueWins']),
                'BlueLosses': int(request.form['BlueLosses']),
                'RedAge': int(request.form['RedAge']),
                'BlueAge': int(request.form['BlueAge']),
                'NumberOfRounds': int(request.form['NumberOfRounds']),
                # Add more fields as necessary based on your model
            }

            # Create DataFrame for prediction
            input_data = pd.DataFrame([data])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)