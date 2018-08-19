import numpy as numpy
from flask import Flask, abort, jsonify, request
import _pickle as pickle


nb_pkl = pickle.load(open('Hackathon_NB_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/api', methods=['POST'])

def make_predict():
	#get json from POST
	data = request.get_json(force=True)

	#convert json to np array
	predict_request = [data['dia_morosidad'],
						data['edad'],
						data['sexo'],
						data['missing_score'],
						data['missing_salario'],
						data['missing_apc'],
						data['missing_corregimiento'],
						data['rare_province'],
						data['prepaid_suscription'],
						data['plan_voz'],
						data['line_number'],
						data['weekly_minutes_call'],
						data['arpu']
						]
	# cast variables into np.array
	predict_request = np.array(predict_request)

	# np array goes into nb_pkl, prediction comes out
	y_hat = nb_pkl(predict_request)

	# return prediction / return json
	output = [y_hat[0]]
	return jsonify(results=output)

if __name__ == '__main__':
    app.run(port=9000, debug=True)


