import torch
from torchvision import transforms
import cv2
from PIL import Image
from torchvision import transforms
import time
import os

from model_vgg13 import VGGnet, VGGnet_custom


def loadModel(weights):

    model = VGGnet(in_channels=3, num_classes=2)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()
    return model

def inferenceFrontalFaceModel(img , model):

    trans = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    # cv2 to PIL conversion
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = trans(img)
    img = torch.unsqueeze(img, 0)

    t1=time.time()
    out = model(img.cuda())
    print("inference time is: ", time.time()-t1)

    # out = model(img)
    #print("out:  ", out)
    _, index = torch.max(out, 1)
    return index.item()

def images_dir_inference(model):
    
    # images_path = os.getcwd() + '/Validated-dataset/Face_Position_Dataset/val/Front/'
    images_path = os.getcwd() + '/Face_Position_Dataset/val/Front/'
    write_path = os.getcwd() + '/inference_results/orig-dataset/Side/'

    inference = 'Front'

    mapper=['Front', 'Side']
    frontal_counter = 0
    side_counter = 0
    accuracy = 0

    list = os.listdir(images_path) # dir is your directory path
    number_files = len(list)

    for img in os.listdir(images_path):
        img_path = images_path + img
        img_name = img.split('.')[0]

        print(img_path)
        image = cv2.imread(img_path)

        # t1=time.time()
        output = inferenceFrontalFaceModel(image, model)
        # t2=time.time()
        # print(t2-t1)
        
        output= mapper[output]
        print(output)

        if output == 'Front':
            frontal_counter = frontal_counter + 1
        elif output == 'Side':
            side_counter = side_counter + 1

        cv2.putText(image, output, (50, 50), 2, 1, (0, 0, 255))

        output_img = write_path + img_name + '.jpg'
        # cv2.imwrite(output_img, image)

    print("Total images: ", number_files)
    print("Frontal predictions: ", frontal_counter)
    print("Side predictions: ", side_counter)

    if inference == 'Front':
        accuracy = (frontal_counter / number_files) * 100
    else:
        accuracy = (side_counter / number_files) * 100 

    print("Accuracy for {} is: {}".format(inference, round(accuracy, 2)))

if __name__ == "__main__":
    # model = loadModel('model_best.pth.tar')
    
    model = loadModel('./vgg13-training-orig-face-dataset/adam/model_best.pth.tar')
    images_dir_inference(model)   