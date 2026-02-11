from flask import Flask, request, jsonify, send_from_directory
import torch
from torch import nn
import os

app = Flask(__name__)

# Re-define model class (must match the one used during training)
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Load model
MODEL_PATH = "models/01_putting_all_together_0.pth"
model = LinearRegressionModelV2()
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
        "weights": model.linear_layer.weight.item(),
        "bias": model.linear_layer.bias.item()
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
