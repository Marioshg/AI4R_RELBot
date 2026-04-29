from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8n.pt")
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    
    print(f"Starting training using dataset config: {yaml_path}")

    results = model.train(
        data=yaml_path,
        epochs=10,        # Number of epochs to train.
        imgsz=640,        # 640x640 images
        batch=8,          # Number of images per batch
        name="hardhat_model"  # The folder name where results/weights will be saved
    )
    
    print("Training finished yayyy!")

if __name__ == "__main__":
    main()