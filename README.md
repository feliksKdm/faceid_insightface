### make_face.py - файл для тренировки распознования лиц в папке 'data/db' а имена людей даются от названия папок внутри 
### predict_face.py - файл для предсказывания лиц через фотографию с помошью функций pred_face которая сразу же запускает make_face.py для тренироваки распознования
### подробности можете взглянуть в файле "make_faces.py" 


```
import cv2
import matplotlib.pyplot as plt
from predict_face import pred_face, draw_on

img_path = 'path2photo'
img = cv2.imread(img_path)

print(pred_face(img_path))

img, names = draw_on(img_path)
plt.imshow(img[:,:,::-1])
```
