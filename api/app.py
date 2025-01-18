from flask import Flask, render_template
import os

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/dashboard", methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000)