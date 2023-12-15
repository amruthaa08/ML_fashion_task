
import torch
import os
from torchvision import transforms, models

from torchvision.transforms.functional import resize, to_pil_image
from torchcam.methods import LayerCAM
import pickle

import numpy as np
from sklearn.cluster import KMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# category prediction model
cat_model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),

    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),

    torch.nn.Flatten(),

    torch.nn.Linear(32*18*13, 64),
    torch.nn.ReLU(),

    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),

    torch.nn.Linear(32, 7) 
)

# image transforms for category prediction
category_tsfm = transforms.Compose([
    transforms.Resize(size=(80, 60)),
    transforms.ToTensor() 
])

# categories
cat_classes = [
    'Accessories',
    'Apparel',
    'Footwear',
    'Free Items',
    'Home',
    'Personal Care',
    'Sporting Goods'
            ]


# color classification model
color_classifier = models.resnet50(pretrained=True).to(device)
    
color_classifier.fc = torch.nn.Sequential(
               torch.nn.Linear(2048, 128),
               torch.nn.ReLU(),
               torch.nn.Linear(128, 7)).to(device)

# function to obtain category predictions
def cat_predict(file):
    img = category_tsfm(file)
    logits = cat_model(img.unsqueeze(0))
    _, preds = torch.max(logits, 1)
    predicted_class = cat_classes[preds[0]]
    return predicted_class

# image transforms for color prediction
color_tsfm = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.CenterCrop(112),
    transforms.ToTensor() 
])

# class activation map extractor
cam_extractor = LayerCAM(color_classifier)

# knn classifier to find closest color
# knn_path =  "../models/knn_classifier.pkl"
root_dir = os.path.dirname(os.path.dirname(__file__))
knn_path = os.path.join(root_dir, "models/knn_classifier.pkl")
with open(knn_path, 'rb') as f:
    knn_classifier = pickle.load(f)

# function to obtain color predictions
def color_predict(img):
    input_tensor = resize(img, (112, 112))
    input_tensor = color_tsfm(input_tensor).to(device)

    out = color_classifier(input_tensor.unsqueeze(0))
    cams = cam_extractor(out.squeeze(0).argmax().item(), out)

    scores = cams[0].cpu()
    resized_scores = resize(to_pil_image(scores.squeeze(0)), size=(112, 112))
    scores_array = np.array(resized_scores)
    norm_scores = scores_array/255
    seg_mask = np.where(norm_scores > 0.8, 1, 0)
    
    img_array = np.array(to_pil_image(input_tensor.cpu()))
    masked_image = img_array[seg_mask == 1]

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(masked_image)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    idx = np.where(counts == counts.max())[0]
    s = tuple(map(int,kmeans.cluster_centers_[idx][0]))

    predicted_class = knn_classifier.predict(np.array(s).reshape(1, -1))

    return (s, predicted_class[0])