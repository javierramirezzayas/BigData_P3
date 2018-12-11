from flask import Flask,jsonify,request
from flask import render_template
import ast
import json

app = Flask(__name__)
labels_model1 = []
values_model1 = []
labels_model2 = []
values_model2 = []


@app.route("/")
def get_chart_page():
	global labels_model1, values_model1, labels_model2, values_model2
	labels_model1 = []
	values_model1 = []
	labels_model2 = []
	values_model2 = []
	return render_template('chart.html', values_model1=values_model1, labels_model1=labels_model1, values_model2=values_model2, labels_model2=labels_model2)


@app.route('/refreshData')
def refresh_graph_data():
	global labels_model1, values_model1, labels_model2, values_model2
	if labels_model1 == [] or labels_model2 == []:
		with open('graph_values/models_results.json', "r") as f:
			data = json.load(f)
		labels_model1 = ast.literal_eval(data['label_model1'])
		labels_model2 = ast.literal_eval(data['label_model2'])
		values_model1 = ast.literal_eval(data['data_model1'])
		values_model2 = ast.literal_eval(data['data_model2'])
	return jsonify(m1Label=labels_model1, m1Data=values_model1, m2Label=labels_model2, m2Data=values_model2)


@app.route('/updateData', methods=['POST'])
def update_data():
	global labels_model1, values_model1, labels_model2, values_model2
	if not request.form or 'data_model1' not in request.form:
		return "error", 400
	labels_model1 = ast.literal_eval(request.form['label_model1'])
	values_model1 = ast.literal_eval(request.form['data_model1'])
	labels_model2 = ast.literal_eval(request.form['label_model2'])
	values_model2 = ast.literal_eval(request.form['data_model2'])
	return "success", 201


if __name__ == "__main__":
	app.run(host='localhost', port=5002)

