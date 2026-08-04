from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>AI Exam Proctoring System</h1>
    <p>My website is working!</p>
    """

if __name__ == "__main__":
    app.run(debug=True)