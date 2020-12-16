from model import Predict

from flask import Flask
app = Flask(__name__)

@app.route('/predict/<int:input_data>')
def hello_world(input_data):
    return Predict(input_data)

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')