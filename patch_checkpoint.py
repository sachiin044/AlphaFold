import torch

# Path to your trained model
MODEL_PATH = "models/protein_classifier.pt"

# Load existing checkpoint
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

# Add missing metadata
ckpt["model_type"] = "mlp"   # Change to "conv" if your model was CNN
ckpt["input_dim"] = 512      # Change to your actual embedding/output size

# Save it back
torch.save(ckpt, MODEL_PATH)
print(f"✅ Patched {MODEL_PATH} successfully!")
