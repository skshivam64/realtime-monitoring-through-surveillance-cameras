from keras.models import load_model
from numpy import load
import pickle
from res.utils import *
import cv2
import numpy as np
import time

model = load_model('models/facenet/facenet_keras.h5')

svc_model = pickle.load(open('models/classifier/model02.sav', 'rb'))

labels = load('labels/class_labels01.npz')['arr_0']

input  = Input()

while(True):
    cap = cv2.VideoCapture(input + '/video') # http://192.168.31.192:8080
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face = extract_face_live(frame)
    if len(face) is not 0:
        print("Face Detected: ", end = " ")
        sample_embeddings = get_embeddings(model, np.array((face,)))
        sample_embeddings = normalize(sample_embeddings)
        yhat_class = svc_model.predict(sample_embeddings)
        yhat_prob = svc_model.predict_proba(sample_embeddings)
        if yhat_prob[0][yhat_class[0]] > 0.75:
            print(labels[yhat_class[0]])
        else:
            print("No Person Matched")
        for j in range(labels.shape[0]):
            print(labels[j], "({:.2f}%)".format(yhat_prob[0][j]*100), end = " ")
        print("\n")
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break