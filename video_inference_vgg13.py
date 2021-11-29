import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import time
import os
import numpy as np
import cv2
import bleedfacedetector as fd
import os
from PIL import Image
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import time

import torchvision.models as models

from model_vgg13 import VGGnet, VGGnet_custom
import os
import torch.nn as nn

def loadModel(weights):

    model = VGGnet(in_channels=3, num_classes=2)

    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])

    model.cuda()
    model.eval()
    return model

def inference(model):
    #vid_path = os.getcwd() + '/videos/female1_lower_camera.avi'
    #vid = cv2.VideoCapture(vid_path)
    vid = cv2.VideoCapture(0) # for webcam
    
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    mapper = ['Front', 'Side']

    while (vid.isOpened()):
        ret, frame = vid.read()

        if ret == False:
            break

        # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = fd.ssd_detect(frame)

        for face in faces:

            x=face[0]
            y=face[1]
            w=face[2]
            h=face[3]

            x2=x+w
            y2=y+h

            cropped_face = frame[y:y2, x:x2]
            img_crop = Image.fromarray(cropped_face)
            transform_face = transform(img_crop)
            img = torch.unsqueeze(transform_face, 0)

            t1=time.time()
            out = model(img.cuda())
            print("inference time is: ", time.time()-t1)

            _, index = torch.max(out, 1)
            label = index.item()
            output = mapper[label]

            c1, c2 = (int(x),int(y)),(int(x2), int(y2))

            cv2.rectangle(frame, c1, c2, (0, 0, 255), 2)
            cv2.putText(frame, output, (c1[0], c1[1] - 2), 0, 2 / 3, (0,255,0), 2, lineType=cv2.LINE_AA)
            
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # model = loadModel(os.getcwd() + '/vgg13-training-validated-dataset/vgg13-adam-0.0001lr/model_best.pth.tar')
    model = loadModel('model_best.pth.tar')
    inference(model)
