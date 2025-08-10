# fix_input_dim.py

import torch

ckpt_path = "models/protein_classifier.pt"

# Load original checkpoint safely
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

# Detect correct input dimension from first Linear layer
first_weight = checkpoint["model_state"]["net.0.weight"]
real_input_dim = first_weight.shape[1]
print(f"Detected input_dim = {real_input_dim}")

# Update checkpoint
checkpoint["input_dim"] = real_input_dim
checkpoint["model_type"] = "mlp"

# Save back
torch.save(checkpoint, ckpt_path)
print(f"✅ Updated checkpoint with input_dim={real_input_dim}")
