from ultralytics import YOLO
import os

def main():
    model = YOLO("yolov8s.pt")  

    model.train(
        data=r"C:\Users\Abhinav Gusain\Documents\Air_defence\data\detection\data.yaml",  
        epochs=40,
        imgsz=640,
        batch=16,
        device=0,  
        project=r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs",
        name="weights", 
        save=True,
        save_period=5,  
        plots=True,
        workers=0  
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  
    main()
