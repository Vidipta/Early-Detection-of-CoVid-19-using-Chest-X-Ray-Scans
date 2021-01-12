#Web App for Image Classification

'''Use command prompt to run the file; go to the folder containing this file
and write streamlit run <filename>.py'''


import streamlit as st
import torch
import torchvision
from PIL import Image

st.title('Early Detection of Covid-19 using Chest X-Ray images')
file_up = st.file_uploader("Upload an image", type="png")
image_path = 'C:/Users/sweet/OneDrive/Desktop/ResearchPaperz/PROJECT/COVID-19 Radiography Database/test/covid/COVID-19 (243).png'
image = Image.open(image_path).convert('RGB')
col1, col2 = st.beta_columns(2)
if file_up is not None:
    file_details = {"FileName":file_up.name,"FileType":file_up.type,"FileSize":file_up.size}
    #st.write(file_details)
    image = Image.open(file_up).convert('RGB')
    col1.image(image, caption='Uploaded Image.', use_column_width=True)

from torchvision import transforms, models

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )])

image_path = 'C:/Users/sweet/OneDrive/Desktop/ResearchPaperz/PROJECT/COVID-19 Radiography Database/test/covid/COVID-19 (243).png'
img = image.convert('RGB')

#img = Image.open(image_path).convert('RGB')

batch_t = torch.unsqueeze(transform(img), 0)
resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
resnet18.load_state_dict(torch.load('covid_classifier.pt'))
resnet18.eval()
out = resnet18(batch_t)

if(col2.button("Classify the uploaded image")):
    col2.write("")
    col2.write("Classifying...")
    print(out.detach().numpy())
    arr = out.detach().numpy()[0]
    x = (-arr).argsort()[:1]
    if x[0]==0:
        col2.success("Normal")
    elif x[0] ==1:
        col2.warning("Viral")
    else:
        col2.error("Covid-19")
#print(max)
#print(itemindex)
#st.write(out.detach().numpy())
