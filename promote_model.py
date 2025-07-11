from mlflow.tracking import MlflowClient

# Replace with the run ID printed from your training script
run_id = "29db458357f04feeb6bea503fe908218"  # Example: "abc123def456"

client = MlflowClient()
model_name = "Best_MedicalCostModel"

# Try to create the model name (will skip if it already exists)
try:
    client.create_registered_model(model_name)
except:
    print("ℹ️ Model name already exists — continuing.")

# Register the model version
model_version = client.create_model_version(
    name=model_name,
    source=f"runs:/{run_id}/model",
    run_id=run_id
)

# Promote it to Production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print(f"✅ Model '{model_name}' version {model_version.version} promoted to Production.")