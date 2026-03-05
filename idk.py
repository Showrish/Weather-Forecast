import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODELS_DIR):
    print("Models folder not found.")
    exit()

model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

if not model_files:
    print("No .pkl files found inside models folder.")
    exit()

print("Found model files:", model_files)
print("=" * 60)

for filename in model_files:
    print(f"\nInspecting: {filename}")
    print("-" * 60)

    model_path = os.path.join(MODELS_DIR, filename)

    try:
        obj = joblib.load(model_path)
    except Exception as e:
        print("Error loading model:", e)
        continue

    print("Object type:", type(obj))

    # If saved as dictionary
    if isinstance(obj, dict):
        print("Keys inside object:")
        for key in obj.keys():
            print(" -", key)

        if "features" in obj:
            features = obj["features"]
            print("\nNumber of features:", len(features))
            print("Features:")
            for f in features:
                print(" ", f)

        if "model" in obj:
            model = obj["model"]
            print("\nInner model type:", type(model))

    else:
        model = obj

        if hasattr(model, "feature_names_in_"):
            print("\nFeature names:")
            for f in model.feature_names_in_:
                print(" ", f)

        if hasattr(model, "named_steps"):
            print("\nPipeline steps:")
            for name, step in model.named_steps.items():
                print(f" {name}: {type(step)}")

    print("=" * 60)

print("\nInspection complete.")