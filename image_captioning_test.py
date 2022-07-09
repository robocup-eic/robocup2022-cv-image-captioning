import requests
# import matplotlib.pyplot as plt
import json
import numpy as np
from human_crop import crop
import cv2


# https://github.com/Azure-Samples/cognitive-services-quickstart-code/blob/master/python/ComputerVision/REST/python-analyze.md


def describe(part_name, part_frame, min_h, resource_name="meen-test", Ocp_key='d579e048b37d46d683c1482b00e2696d',
             version=3.1,
             maxCandidates=1):
    if part_frame.shape[0] < min_h:
        print(part_name, " too small.")
        return None

    describe_url = f'https://{resource_name}.cognitiveservices.azure.com/vision/v{version}/describe'
    headers = {'Content-Type': 'application/octet-stream', 'Ocp-Apim-Subscription-Key': Ocp_key}
    params = {'language': 'en', 'maxCandidates': maxCandidates}
    im_buf_arr = cv2.imencode(".jpg", part_frame)[1]
    frame_bytes = im_buf_arr.tobytes()

    response = requests.post(describe_url, headers=headers, params=params, data=frame_bytes)
    response.raise_for_status()

    # The 'analysis' object contains various fields that describe the image. The most
    # relevant caption for the image is obtained from the 'description' property.
    analysis = response.json()
    print(json.dumps(analysis))
    # print(analysis["description"])
    # image_caption = analysis["description"]["captions"][0]["text"].capitalize()

    cv2.imshow(part_name, part_frame)
    # cv2.waitKey()

    return analysis["description"]


def describe_all(crop, min_h=100, resource_name="meen-test", Ocp_key='d579e048b37d46d683c1482b00e2696d', version=3.1,
                 maxCandidates=1):
    tags = set()
    captions = []
    if crop:
        print("\nAnalysing images....\n")
        for part_name in crop:
            part_frame = crop[part_name]
            print(part_name, f"w:{part_frame.shape[1]}, h:{part_frame.shape[0]}")
            description = describe(part_name, part_frame, min_h=100, resource_name="meen-test",
                                   Ocp_key='d579e048b37d46d683c1482b00e2696d', version=3.1, maxCandidates=3)
            if description:
                tags.update(description["tags"])

                for caption in description["captions"]:
                    captions.append(caption["text"])

                print()
    else:
        print("No human detected.")

    return tags, captions


def get_age_gender(crop, part="head"):
    print(f"\nAnalysing age and gender from {part}....\n")

    output_indexes = np.array([i for i in range(0, 101)])

    if part in crop:
        frame = crop[part]
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = np.array(haar_detector.detectMultiScale(gray, 1.3, 5))

        if faces.size != 0:
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
                gender = 'Female' if np.argmax(gender_class) == 0 else 'Male'
                print("Gender: ", gender)

                return apparent_predictions, gender

        elif part == "whole":
            print("Can't detect face.")
            return 0, ""
        else:
            print("Error")
            return get_age_gender(crop, "whole")

    else:
        return get_age_gender(crop, "whole")


def paraphrase(gender=None, age=None, captions=None, tags=None):
    text = ""

    if gender:
        gen = "man" if gender == "Male" else "woman"
        text += f"You are a {gen}. "
    # else:
    #     text += f"I can't identify your gender. "

    if age:
        text += f"Your apparent age is {age} years old. "
    # else:
    #     text += f"I can't approximate your age. "

    if captions:
        capts = captions

        for capt in capts:
            # capt = 'a person posing for a picture'
            capt = capt.lower()
            if "wearing" in capt:
                text += f"You are wearing {capt.split('wearing')[-1].lstrip()}. "

            elif gender:
                if gen in capt:
                    text += f"I see a {gen}{capt.split(gen)[-1]}. "

                elif "person" in capt:
                    text += f"I see {capt.replace('person', gen)}. "
                else:
                    text += f"I see {capt}. "

            elif "person" in capt:
                text += f"I see a person{capt.split('person')[-1]}. "

            else:
                text += f"I see {capt}. "

    return text


path = "dataset/no.jpg"

# age_gender_recog model
print("Initialing age and gender model...")
haar_detector = cv2.CascadeClassifier("age_gender_recog/haarcascade_frontalface_default.xml")
age_model = cv2.dnn.readNetFromCaffe("age_gender_recog/age.prototxt",
                                     "age_gender_recog/age_caffe.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe("age_gender_recog/gender.prototxt", "age_gender_recog/gender_caffe.caffemodel")
print("Done.")
###

frame = cv2.imread(path)
crop = crop(frame)
if crop:
    age, gender = get_age_gender(crop)
    tags, captions = describe_all(crop, min_h=100, resource_name="meen-test",
                                  Ocp_key='d579e048b37d46d683c1482b00e2696d',
                                  version=3.1,
                                  maxCandidates=1)

    print(captions)
    print(paraphrase(gender, age, captions, tags))

    cv2.waitKey()
else:
    print("No human detected.")
