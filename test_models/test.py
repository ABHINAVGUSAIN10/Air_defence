from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt


model = YOLO(r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs\weights\best.pt")


image_path = r"C:\Users\Abhinav Gusain\Documents\Air_defence\test_images\ShCH39Q.jpg"
results = model(image_path)


results[0].show()  
plt.imshow(results[0].plot())
plt.axis("off")
plt.show()


for r in results:
    print("Classes detected:", r.names)
    print("Predicted classes:", r.boxes.cls)
