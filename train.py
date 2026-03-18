from ultralytics import YOLO
import os
import argparse
if __name__ == "__main__":
    # Initialize the command-line argument parser
    parser = argparse.ArgumentParser(description="YOLO training script.")
    
    # Add the two required arguments
    parser.add_argument("model_yaml", type=str, help="Path to the model YAML file")
    parser.add_argument("data_yaml", type=str, help="Path to the dataset YAML file")
    
    # Parse the arguments
    args = parser.parse_args()

    print(f"Loading model from: {args.model_yaml}")
    # Build a new model from the provided YAML path
    model = YOLO(args.model_yaml)

    print(f"Starting training with dataset: {args.data_yaml} ...")
    # Train the model using the parsed data path and your fixed parameters
    results = model.train(
        data=args.data_yaml,
        project="",
        epochs=400,
        batch=16,
        optimizer='SGD',
        pretrained=False,
        imgsz=640,
        # workers=12,
        iou=0.4
    )

# # Load a model
# model = YOLO("/opt/data/private/UOD/ultralytics-mm/train_runs/mm_frequency/3_28_mm_v2_duov2/train2/weights/best.pt")  # build a new model from YAML
# metrics = model.val(iou=0.3)
# print(model.info(detailed=True))

# # # Train the model
# results = model.train(resume=True, epochs=400, batch=16, optimizer='SGD', imgsz=640, device=0, name='test_cuda11_8_89_s_duo')
# results = model.train(data='/opt/data/private/UOD/DUO/duo.yaml', epochs=400, batch=16, optimizer='SGD', pretrained=False, imgsz=640, device=0, name='a6000_brk_2_2_duo')