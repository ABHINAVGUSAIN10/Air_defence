from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO(r"C:\Users\Abhinav Gusain\Documents\Air_defence\results\detection_outputs\weights\best.pt")

# Load the image
image_path = r"C:\Users\Abhinav Gusain\Documents\Air_defence\test_images\ShCH39Q.jpg"
results = model(image_path)

# Plot results
results[0].show()  # To visualize bounding boxes
plt.imshow(results[0].plot())
plt.axis("off")
plt.show()

# Print predicted classes
for r in results:
    print("Classes detected:", r.names)
    print("Predicted classes:", r.boxes.cls)
