import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO validation script.")
    
    parser.add_argument("weights", type=str, help="Path to the model weights file (e.g., .pt file)")
    parser.add_argument("data_yaml", type=str, help="Path to the dataset YAML file")
    
    args = parser.parse_args()
    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data_yaml, 
        iou=0.4
    )
    print("Validation completed successfully!")