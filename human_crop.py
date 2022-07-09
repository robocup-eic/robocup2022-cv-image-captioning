import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


def get_pose_coords(results, frame_width, frame_height, landmark_index):
    return tuple(np.multiply(
        np.array((results.pose_landmarks.landmark[landmark_index].x,
                  results.pose_landmarks.landmark[landmark_index].y)),
        [frame_width, frame_height]).astype(int))


def get_max_min_x(results, frame_width, frame_height):
    lx = [get_pose_coords(results, frame_width, frame_height, ix)[0] for ix in range(33)]
    max_lx, min_lx = max(lx), min(lx)
    c = max_lx - min_lx
    return min(frame_width, max(lx) + c // 4), max(0, min(lx) - c // 4)


def show(img_name, img):
    if img.shape[0] > 0:
        cv2.imshow(img_name, img)
    else:
        print("No visible", img_name)


def save(file_name, img):
    if img.shape[0] > 0:
        cv2.imwrite(file_name, img)

    else:
        print("Can't save", file_name)


def crop(cap):
    frame_height, frame_width, _ = cap.shape
    print(f"frame width: {frame_width}, frame height: {frame_height}")

    # Detection = initial detection, Tracking = Tracking after initial detection

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cap.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)  # Detection
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.pose_landmarks:
            # print(results.pose_landmarks)
            print("Cropping the picture....\n")
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            max_x, min_x = get_max_min_x(results, frame_width, frame_height)

            cropx = cap[:, min_x:max_x, :]

            const = abs(get_pose_coords(results, frame_width, frame_height, 9)[1] -
                        get_pose_coords(results, frame_width, frame_height, 11)[1])

            shoulder = (get_pose_coords(results, frame_width, frame_height, 11)[1] +
                        get_pose_coords(results, frame_width, frame_height, 12)[1]) // 2
            hip = (get_pose_coords(results, frame_width, frame_height, 23)[1] +
                   get_pose_coords(results, frame_width, frame_height, 24)[1]) // 2

            # print(shoulder, hip, const)

            whole = cropx[
                    max(get_pose_coords(results, frame_width, frame_height, 1)[1] - const, 0):
                    min(frame_height, get_pose_coords(results, frame_width, frame_height, 32)[1] + 10), :, :]

            head = cropx[
                   max(get_pose_coords(results, frame_width, frame_height, 1)[1] - const, 0):shoulder - const // 2, :,
                   :]
            body = cropx[shoulder - const // 2:hip, :, :]
            leg = cropx[
                  hip - const // 2:min(frame_height, get_pose_coords(results, frame_width, frame_height, 32)[1] + 10),
                  :, :]

            crop = {"whole": whole, "head": head, "body": body, "leg": leg}
            # print(crop)
            return crop

        else:
            print("No human detected.")
            return set()
