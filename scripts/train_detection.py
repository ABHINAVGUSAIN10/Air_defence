from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8s.pt")  # You can change to yolov8s.pt if VRAM allows

    model.train(
        data=r"C:\Users\Abhinav Gusain\Documents\Air_defence\data\detection\data.yaml",  # path to your data YAML
        epochs=40,
        imgsz=640,
        batch=16,
        device=0,  # Set to 0 for GPU
        project=r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs",
        name="weights",  # Weights will be saved to results/weights
        save=True,
        save_period=5,  # Save every 5 epochs
        plots=True,
        workers=0  # Needed for Windows sometimes, or set >0 if needed
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Safe for Windows
    main()
