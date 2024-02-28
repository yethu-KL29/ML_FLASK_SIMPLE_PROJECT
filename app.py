from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

# Flask app instance
app = Flask(__name__)

# Load models
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))


# Home route
@app.route("/")
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    print(data)

    try:
        # Check data structure
        if isinstance(data, dict) and 'data' in data:
            inner_data = data['data']  # Access inner dictionary
            if isinstance(inner_data, dict):
                # Extract and convert values (assuming they're valid numbers)
                mol_logp = float(inner_data['MolLogP'])
                mol_wt = float(inner_data['MolWt'])
                num_rotatable_bonds = float(inner_data['NumRotatableBonds'])
                aromatic_proportion = float(inner_data['AromaticProportion'])

                # Prepare data for prediction
                new_data = scalar.transform(np.array([mol_logp, mol_wt, num_rotatable_bonds, aromatic_proportion]).reshape(1, -1))
                out = regmodel.predict(new_data)
                print(out)

                return jsonify(out[0])
            else:
                # Handle unexpected inner data structure
                return jsonify({'error': 'Invalid data format'}), 400
        else:
            # Handle unexpected main data structure
            return jsonify({'error': 'Invalid data format'}), 400

    except ValueError:
        # Handle conversion errors
        print("Error: Some values cannot be converted to floats.")
        return jsonify({'error': 'Invalid data format'}), 400

    except Exception as e:  # Catch other unexpected errors
        print(f"Unexpected error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run()
