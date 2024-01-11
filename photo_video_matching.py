import cv2
import numpy as np
from insightface.model_zoo import model_zoo
from insightface.app.common import Face
from test import test
from glob import glob
from tqdm import tqdm
import os
from insightface.app.common import Face

# Load models only once
det_model_path = 'buffalo_l/det_500m.onnx'
rec_model_path = 'buffalo_l/w600k_mbf.onnx'
gender_age_model_path = 'buffalo_l/genderage.onnx'

det_model = model_zoo.get_model(f'./models/{det_model_path}')
rec_model = model_zoo.get_model(f'./models/{rec_model_path}')
gender_age = model_zoo.get_model(f'./models/{gender_age_model_path}')

det_model.prepare(ctx_id=0, input_size=(640, 640), det_thres=0.4)

# Function to search and flatten embeddings
def search_flatten(known_embeddings, known_names, unknown_embeddings, threshold=0.65):
    pred_names = []
    scores_list = []
    for emb in unknown_embeddings:
        scores = np.dot(emb, known_embeddings.T)
        scores = np.clip(scores, 0., 1.)
        idx = np.argmax(scores)
        score = scores[idx]
        scores_list.append(score)
        pred_names.append(known_names[idx] if score > threshold else None)
    
    return pred_names, scores_list

# Function to process each player and create embeddings
def process_players(players, img):
    known_names, known_embeddings = [], []

    for player in tqdm(players):

        photo_dir = f"./data/db/{player}/{img.split('.')[0].split('/')[-1]}.jpg"
        img_ = cv2.imread(img)
        cv2.imwrite(photo_dir, img_)
        player_embeddings, player_names = [], []
        img_paths = glob(f'./data/db/{player}/*')

        for img_path in img_paths:
            img = cv2.imread(img_path)
            if img is None: continue

            bboxes, kpss = det_model.detect(img, max_num=0, metric='default')
            if len(bboxes) != 1: continue

            face = Face(bbox=bboxes[0, :4], kps=kpss[0], det_score=bboxes[0, 4])
            rec_model.get(img, face)
            player_embeddings.append(face.normed_embedding)
            player_names.append(player)

            if len(player_embeddings) == 10: break

        if player_embeddings:
            player_embeddings = np.stack(player_embeddings, axis=0)
            known_embeddings.append(player_embeddings[0:5])
            known_names += player_names[0:5]

    return np.concatenate(known_embeddings, axis=0), known_names

# Main function for matching
def matching(img, video):

    players = os.listdir(f'./data/db')

    known_embeddings, known_names = process_players(players, img)

    cap = cv2.VideoCapture(video)
    liveness_notdetected = 0
    approved = 0
    recognized_scores = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        bboxes, kpss = det_model.detect(frame, max_num=0, metric='default')

        if len(bboxes) > 0:  # Run anti-spoofing check only if faces are detected
            for i, bbox in enumerate(bboxes):
                face = Face(bbox=bbox[:4], kps=kpss[i])

                # # Crop the image
                rec_model.get(frame, face)
                pred_name, scores = search_flatten(known_embeddings, known_names, [face.normed_embedding])

                if pred_name[0] is not None:
                    recognized_scores.append(scores[0])
                    label = test(image=frame, model_dir='resources/anti_spoof_models', device_id=0)
                    print(label)
                    if label == 2:
                        liveness_notdetected += 1
                        if liveness_notdetected >= 500:
                            print('fake face detected')
                            cap.release()
                            cv2.destroyAllWindows()
                    else:
                        approved += 1

        # Display the resulting frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate mean score for recognized faces
    mean_score = np.mean(recognized_scores) if recognized_scores else 0
    print(f"Mean Score: {mean_score}, {approved}")
    print(f"fake face detection: {liveness_notdetected}")

# Run the matching function
matching('/home/feliks/Downloads/photo_5256188845681137684_y.jpg', 'video')