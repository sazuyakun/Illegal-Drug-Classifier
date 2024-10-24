from flask import Flask, jsonify, request
from DrugTextAnalyzer import DrugTextAnalyzer

model = DrugTextAnalyzer()

app = Flask(__name__)

@app.post("/text-predict")
def drugClassification():
    data = request.json
    postText = data["user"]
    result = model.process_input(str(postText))
    return jsonify({
        "classification": result
    })
    
if __name__ == "__main__":
    app.run(port=8080, debug=True)