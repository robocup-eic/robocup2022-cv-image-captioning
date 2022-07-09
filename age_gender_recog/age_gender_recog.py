import os
import cv2
import numpy as np

# import matplotlib.pyplot as plt

# https://github.com/serengil/tensorflow-101/blob/master/python/Age-Gender-Caffe.ipynb


haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
# pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")

# model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
# pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
gender_model = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")

output_indexes = np.array([i for i in range(0, 101)])


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_detector.detectMultiScale(gray, 1.3, 5)
    return faces


def analysis(img_path):
    img = cv2.imread(img_path)

    # plt.imshow(img[:, :, ::-1]); plt.axis('off'); plt.show()

    # detect face

    faces = detect_faces(img)

    for face in faces:
        x, y, w, h = face

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        # age model is a regular vgg and it expects (224, 224, 3) shape input

        detected_face = cv2.resize(detected_face, (224, 224))
        img_blob = cv2.dnn.blobFromImage(detected_face)  # caffe model expects (1, 3, 224, 224) shape input

        # ---------------------------

        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        apparent_predictions = round(np.sum(age_dist * output_indexes), 2)
        print("Apparent age: ", apparent_predictions)

        # ---------------------------

        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
        print("Gender: ", gender)

        # ---------------------------

        # plt.imshow(detected_face[:, :, ::-1]);
        # plt.axis('off')
        # plt.show()


def analysis_and_show(img_path):
    img = cv2.imread(img_path)

    # detect face

    faces = detect_faces(img)

    for face in faces:
        x, y, w, h = face

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        # age model is a regular vgg and it expects (224, 224, 3) shape input

        detected_face = cv2.resize(detected_face, (224, 224))
        img_blob = cv2.dnn.blobFromImage(detected_face)  # caffe model expects (1, 3, 224, 224) shape input

        # ---------------------------

        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        apparent_predictions = round(np.sum(age_dist * output_indexes))
        print("Apparent age: ", apparent_predictions)

        # ---------------------------

        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        gender = 'Woman' if np.argmax(gender_class) == 0 else 'Man'
        print("Gender: ", gender)

        # ---------------------------
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, f"{gender}, {apparent_predictions}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("img", img)

    if cv2.waitKey() == ord("q"):
        cv2.destroyAllWindows()


def frame_analysis(img):
    # img = cv2.imread(img_path)

    print("Analysing age and gender...")
    haar_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
    # pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
    age_model = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")

    # model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
    # pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
    gender_model = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel")

    output_indexes = np.array([i for i in range(0, 101)])

    # detect face

    faces = detect_faces(img)

    for face in faces:
        x, y, w, h = face

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        # age model is a regular vgg and it expects (224, 224, 3) shape input

        detected_face = cv2.resize(detected_face, (224, 224))
        img_blob = cv2.dnn.blobFromImage(detected_face)  # caffe model expects (1, 3, 224, 224) shape input

        # ---------------------------

        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        apparent_predictions = round(np.sum(age_dist * output_indexes))
        print("Apparent age: ", apparent_predictions)

        # ---------------------------

        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        gender = 'Woman' if np.argmax(gender_class) == 0 else 'Man'
        print("Gender: ", gender)


while 1:
    img_name = input("Type image name: ")
    img_path = os.path.join("dataset", img_name)
    try:
        analysis_and_show(img_path)
    except:
        print("Error")
