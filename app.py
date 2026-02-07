from flask import Flask, request, jsonify, send_from_directory
import torch
from torch import nn
import os

app = Flask(__name__)

# Re-define model class (must match the one used during training)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

# Load model
MODEL_PATH = "model.pth"
model = LinearRegressionModel()
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print("Model loaded successfully")
else:
    print("Model file not found! Please run pytorch_workflow.py first.")

@app.route("/")
def index():
    return send_from_directory('static', 'index.html')

@app.route("/static/<path:path>")
def send_static(path):
    return send_from_directory('static', path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        x_val = float(data.get("x", 0))
        
        x_tensor = torch.tensor([[x_val]], dtype=torch.float)
        with torch.inference_mode():
            prediction = model(x_tensor).item()
            
        return jsonify({
            "status": "success",
            "x": x_val,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

@app.route("/model-info")
def model_info():
    return jsonify({
        "weights": model.weights.item(),
        "bias": model.bias.item()
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
