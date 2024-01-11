import pandas as pd
import insightface
from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from insightface.app import FaceAnalysis
import requests
import json
import os
from tqdm import tqdm
import subprocess
import numpy as np 
from glob import glob
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from test import test

det_model_path = 'buffalo_l/det_500m.onnx'
rec_model_path = 'buffalo_l/w600k_mbf.onnx'
gender_age_model_path = 'buffalo_l/genderage.onnx'

det_model = model_zoo.get_model(f'./models/{det_model_path}')
rec_model = model_zoo.get_model(f'./models/{rec_model_path}')
gender_age = model_zoo.get_model(f'./models/{gender_age_model_path}')

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)


def search_flatten(known_embeddings, known_names, unknown_embeddings, threshold=0.5):
    pred_names = []
    for emb  in unknown_embeddings:
        scores = np.dot(emb, known_embeddings.T)
        scores = np.clip(scores, 0., 1.)

        idx = np.argmax(scores)
        score = scores[idx]
        if score > threshold:
            pred_names.append(known_names[idx])
        else:
            pred_names.append(None)
    
    return pred_names, score


def matching(img, video):
    img_to_save = cv2.imread(img)
    
    

    known_names, unknown_names = [], []
    known_embeddings, unknown_embeddings = [], []

    players = os.listdir(f'./data/db')

    for i, player in tqdm(enumerate(players)):
        player_embeddings, player_names = [], []

        photo_dir = f"./data/db/{player}/{i}.jpg"
        cv2.imwrite(photo_dir, img_to_save)

        img_paths = glob(f'./data/db/{player}/*')
        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue

            bboxes, kpss = det_model.detect(img, max_num=0, metric='defualt')
            if len(bboxes) != 1: continue

            bbox = bboxes[0, :4]
            det_score = bboxes[0, 4]
            kps = kpss[0]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)

            rec_model.get(img, face)
            player_embeddings.append(face.normed_embedding)
            player_names.append(player)
            if len(player_embeddings) == 10: break
            if os.path.exists(photo_dir):
                # Delete the file
                os.remove(photo_dir)
                print(f"Image {f'{photo_dir}'} deleted successfully.")
        player_embeddings = np.stack(player_embeddings, axis=0)
        known_embeddings.append(player_embeddings[0:5])
        unknown_embeddings.append(player_embeddings[5:10])
        known_names += player_names[0:5]
        unknown_names += player_names[5:10]

    known_embeddings = np.concatenate(known_embeddings, axis=0)
    unknown_embeddings = np.concatenate(unknown_embeddings, axis=0)

    try:
        video_path = video
        cap = cv2.VideoCapture(video_path)
        verifycap = 0
        liveness = ['T','T','F']
        score_mean = []
        liveness_notdetected = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face detection
            bboxes, kpss = det_model.detect(frame, max_num=0, metric='default')
            img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label = test(image=img_, model_dir='resources/anti_spoof_models', device_id=0)
            # Perform face recognition for each detected face

            for i, bbox in enumerate(bboxes):
                
                bbox = bbox[:4]
                x, y, width, height = bbox.astype(int)
                
                # Create a Face object
                face = Face(bbox=bbox, kps=kpss[i])

                # Use the recognition model to obtain the face embedding
                rec_model.get(frame[:,:,::-1], face)
                new_embedding = face.normed_embedding
                

                # Use the search function to predict the name
                # You can choose either search_flatten or search_average based on your preference
                pred_name = search_flatten(known_embeddings, known_names, [new_embedding])
                pred, score = pred_name
                score_mean.append(score)
                cv2.putText(frame, f'{pred[0]}', (x,y), cv2.FONT_HERSHEY_SIMPLEX ,  
                            1, (0, 255, 0), 2, cv2.LINE_AA) 
                
                print(verifycap, label, liveness_notdetected)
                if label == 2:
                    liveness_notdetected += 1
                if pred[0] != None:
                    if label != 2:
                        verifycap += 1
            if verifycap >= 20:
                print('verified',liveness[label], np.mean(score))
                cap.release()
                cv2.destroyAllWindows()
            if liveness_notdetected >= 3:
                print('scam detected')
                cap.release()
                cv2.destroyAllWindows()
            # Display the resulting frame
            cv2.imshow('Video', frame)

            # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('unverified')
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()
    except:
        print('unverified')
        cap.release()
        cv2.destroyAllWindows()