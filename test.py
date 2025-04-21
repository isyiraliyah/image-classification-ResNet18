import sys, random
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Paths for image directory and model
IMDIR = "test"  # Set the path to your test folder
MODEL = 'model.pth'  # Path to your trained model

# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
class_names = ['safe', 'violent']

# Retrieve 9 random images from the test folder
# Retrieve valid image files from the directory
files = Path(IMDIR).resolve().glob('*.*')
images = [img for img in files if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]

# Ensure at least 9 images are available
if len(images) < 9:
    print("Not enough images in the directory!")
    sys.exit(1)

# Randomly select 9 images
images = random.sample(images, 9)

# Configure plots
fig = plt.figure(figsize=(9, 9))
rows, cols = 3, 3

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Enable GPU mode, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform prediction and plot results
with torch.no_grad():
    for num, img in enumerate(images):
        img = Image.open(img).convert('RGB')
        inputs = preprocess(img).unsqueeze(0).to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        label = class_names[preds]
        plt.subplot(rows, cols, num + 1)
        plt.title("Pred: " + label)
        plt.axis('off')
        plt.imshow(img)

# Show plots
plt.show()
