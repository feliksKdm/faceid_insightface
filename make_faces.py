from insightface.model_zoo import model_zoo
from insightface.app.common import Face
from glob import glob
from tqdm import tqdm
import os
import cv2
import numpy as np


det_model_path = 'buffalo_l/det_500m.onnx'
rec_model_path = 'buffalo_l/w600k_mbf.onnx'
gender_age_model_path = 'buffalo_l/genderage.onnx'

det_model = model_zoo.get_model(f'./models/{det_model_path}')
rec_model = model_zoo.get_model(f'./models/{rec_model_path}')
gender_age = model_zoo.get_model(f'./models/{gender_age_model_path}')

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.5)


known_names, unknown_names = [], []
known_embeddings, unknown_embeddings = [], []

db_faces_path = './data/db'
players = os.listdir(db_faces_path)
for player in tqdm(players):
    player_embeddings, player_names = [], []

    img_paths = glob(f'{db_faces_path}/{player}/*')
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
    
    player_embeddings = np.stack(player_embeddings, axis=0)
    known_embeddings.append(player_embeddings[0:5])
    unknown_embeddings.append(player_embeddings[5:10])
    known_names += player_names[0:5]
    unknown_names += player_names[5:10]

known_embeddings = np.concatenate(known_embeddings, axis=0)
unknown_embeddings = np.concatenate(unknown_embeddings, axis=0)

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
    
    return pred_names