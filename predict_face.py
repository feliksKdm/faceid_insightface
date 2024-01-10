from make_faces import known_embeddings, known_names, det_model , rec_model, gender_age, search_flatten
from insightface.app.common import Face
from insightface.app import FaceAnalysis
import cv2

gender = ['F' ,'M']
def pred_face(new_image_path):
    new_image = cv2.imread(new_image_path)

    # Use the face detection model to detect faces
    bboxes, kpss = det_model.detect(new_image, max_num=0, metric='default')

    # Initialize lists to store results
    detected_people = []
    for i, bbox in enumerate(bboxes):
        face = Face(bbox=bbox, kps=kpss[i])
        rec_model.get(new_image, face)
        gender_age.get(new_image, face)
        new_embedding = face.normed_embedding
        pred_name = search_flatten(known_embeddings, known_names, [new_embedding])[0]
        detected_people.append((pred_name, face["age"], gender[face["gender"]]))
    return detected_people

def draw_on(new_image_path):
    app = FaceAnalysis(allowed_modules=['detection', 'genderage'], root = './')
    app.prepare(ctx_id=0 ,det_size=(640,640))
    new_image = cv2.imread(new_image_path)
    gender = ['F' ,'M']
    # Use the face detection model to detect faces
    bboxes, kpss = det_model.detect(new_image, max_num=0, metric='default')

    # Initialize lists to store results
    detected_names = []
    for i, bbox in enumerate(bboxes):
        # Extract bounding box coordinates
        bbox = bbox[:4]
        x, y, width, height = bbox.astype(int)
        cv2.rectangle(new_image, (x,y), (width, height), (0, 255, 0) , 2)
        
        # Create a Face object
        face = Face(bbox=bbox, kps=kpss[i])

        # Use the recognition model to obtain the face embedding
        rec_model.get(new_image, face)
        gender_age.get(new_image, face)
        new_embedding = face.normed_embedding
        drow = app.draw_on(new_image, [face])

        if face.kps is not None:
                    kps = face.kps.astype(int)
                    #print(landmark.shape)
                    for l in range(kps.shape[0]):
                        color = (0, 0, 255)
                        if l == 0 or l == 3:
                            color = (0, 255, 0)
                        cv2.circle(new_image, (kps[l][0], kps[l][1]), 1, color,
                                2)
                        

        # Use the search function to predict the name
        # You can choose either search_flatten or search_average based on your preference
        pred_name = search_flatten(known_embeddings, known_names, [new_embedding])[0]
        cv2.putText(new_image, f'{pred_name},{face["age"]},{gender[face["gender"]]}', (x,y), cv2.FONT_HERSHEY_SIMPLEX ,  
                    1, (0, 255, 0), 2, cv2.LINE_AA) 
        
        # Store the result
        detected_names.append(pred_name)

        return new_image, detected_names

