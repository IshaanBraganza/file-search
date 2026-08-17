from sentence_transformers import SentenceTransformer

# Choose your model (e.g., 'all-MiniLM-L6-v2')
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Save the model files to a specific folder
local_model_path = "sentence_transformer_local_folder"
model.save(local_model_path)
print(f"Model saved successfully to {local_model_path}")
