from flask import Flask, Response
import cv2
import time
import numpy as np
import mediapipe as mp

CAM_NO = 0
CONTROL_KEYBOARD = True
UP_ROLL_FRAME = 4
UP_THRESH_MIN_Y_DELTA = 0.35
UP_THRESH_MIN_SD = 0.90
DOWN_THRESH_MAX_SD = 0.70
DOWN_AFTER_DELAY = 10
KEY_DELAY = 0.05

# Initialize the objects
app = Flask(__name__)
video = cv2.VideoCapture(0)
# face_cascade = cv2.CascadeClassifier()

# Load the pretrained model
# face_cascade.load(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))

if CONTROL_KEYBOARD:
    from pynput.keyboard import Key, Controller
    def keyboard_press(key):
        keyboard.press(key)
        time.sleep(0.05)
        keyboard.release(key)
    keyboard = Controller()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

LANDMARKS_LEFT_HAND = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.LEFT_PINKY,
    mp_pose.PoseLandmark.LEFT_INDEX,
    mp_pose.PoseLandmark.LEFT_THUMB
]
LANDMARKS_RIGHT_HAND = [
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.RIGHT_PINKY,
    mp_pose.PoseLandmark.RIGHT_INDEX,
    mp_pose.PoseLandmark.RIGHT_THUMB
]
LANDMARKS_SHOULDER = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER
]
LANDMARKS_ELBOW = [
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW
]
LANDMARKS_HIP = [
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP
]
LANDMARKS_KNEE = [
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE
]
LANDMARKS_UPPERBODY = LANDMARKS_ELBOW + LANDMARKS_HIP
LANDMARKS_LOWERBODY = [i for i in range(23, 33)]

# Turn landmark dictionary to numpy array
# Also reverse y coordinate
def landmarks2numpy(image_landmarks, discard_visibility=False):
    if discard_visibility:
        arr = [[l.x, -l.y, l.z] for l in image_landmarks]
    else:
        arr = [[l.x, -l.y, l.z, l.visibility] for l in image_landmarks]
    return np.array(arr)

# Unnormalize x,y components of each landmark
def landmark_array_unnormalize(arr, width, height, inplace=True):
    if not inplace:
        arr = arr.copy()
    arr[:, 0] *= width
    arr[:, 1] *= height
    return arr

# Pose classifier
# Input should be unnormalized
def pose_classify(landmark_coor, rollings, image_width, image_height):
    # Coupling limbs
    y_left_hand = landmark_coor[np.ix_(LANDMARKS_LEFT_HAND, [1])].mean()
    y_right_hand = landmark_coor[np.ix_(LANDMARKS_RIGHT_HAND, [1])].mean()
    y_shoulder = landmark_coor[np.ix_(LANDMARKS_SHOULDER, [1])].mean()
    y_elbow = landmark_coor[np.ix_(LANDMARKS_ELBOW, [1])].mean()
    y_knee = landmark_coor[np.ix_(LANDMARKS_KNEE, [1])].mean()
    y_hip = landmark_coor[np.ix_(LANDMARKS_HIP, [1])].mean()
    
    # Local unit distance
    x_dist_shoulder = landmark_coor[mp_pose.PoseLandmark.LEFT_SHOULDER, 0] - landmark_coor[mp_pose.PoseLandmark.RIGHT_SHOULDER, 0]
    y_dist_elbow_knee = y_elbow - y_knee
    
    # Stats
    y_upper_body = np.average([y_shoulder, y_hip])
    y_sd_lower_body = np.std(landmark_coor[np.ix_(LANDMARKS_LOWERBODY, [1])])

    # Delta-Y for upper body (For jumping classification)
    sum_y_delta = 0
    for lm in LANDMARKS_UPPERBODY:
        rollings[lm].append(landmark_coor[lm, 1])
        if len(rollings[lm]) > UP_ROLL_FRAME:
            rollings[lm].pop(0)
        sum_y_delta += landmark_coor[lm, 1] - (sum(rollings[lm]) / min(len(rollings[lm]), UP_ROLL_FRAME))
    
    # Rule based classification
    if y_right_hand > y_shoulder and y_left_hand < y_shoulder:
        return "right"
    elif y_left_hand > y_shoulder and y_right_hand < y_shoulder:
        return "left"
    elif sum_y_delta / x_dist_shoulder > UP_THRESH_MIN_Y_DELTA and y_sd_lower_body / x_dist_shoulder > UP_THRESH_MIN_SD:
        return "up"
    elif y_sd_lower_body / x_dist_shoulder < DOWN_THRESH_MAX_SD:
        return "under"
    else:
        return "idle"

# Write text onto image
def write_text(image, txt, coor=(100, 100)):
    temp = image.flags.writeable
    image.flags.writeable = True
    cv2.putText(
        img=image,
        text=txt,
        org=coor,
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=2,
        color=(255, 255, 0),
        thickness=2,
    )
    image.flags.writeable = temp

# @app.route('/main')
def main(cap):
    # cap = cv2.VideoCapture(CAM_NO)
    delay_pose = 0
    before_position = 5
    rollings = {lm: [] for lm in LANDMARKS_UPPERBODY}
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            # Press ESC (0xFF) to stop
            if cv2.waitKey(10) & 0xFF == 27:
                break
            
            # Read camera frame-by-frame
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            image_height, image_width, _ = image.shape
            
            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Skip frames with no detection
            if not results.pose_landmarks:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow("MediaPipe Pose", image)
                continue
            
            # Classification
            landmark_coor = landmarks2numpy(results.pose_landmarks.landmark)
            landmark_array_unnormalize(landmark_coor, image_width, image_height)
            pose_result = pose_classify(landmark_coor, rollings, image_width, image_height)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            
            # Control and output to video
            CONTROL_KEYBOARD = False
            if pose_result == "right":
                if before_position != 1:
                    before_position = 1
                    if CONTROL_KEYBOARD:
                        keyboard_press(Key.right)
                write_text(image, "Right")
            elif pose_result == "left":
                if before_position != 2:
                    before_position = 2
                    if CONTROL_KEYBOARD:
                        keyboard_press(Key.left)
                write_text(image, "Left")
            elif pose_result == "up" and delay_pose <= 0:
                if before_position != 3 and before_position != 4:
                    before_position = 3
                    if CONTROL_KEYBOARD:
                        keyboard_press(Key.up)
                write_text(image, "Up")
            elif pose_result == "under":
                if before_position != 4:
                    before_position = 4
                    delay_pose = DOWN_AFTER_DELAY
                    if CONTROL_KEYBOARD:
                        keyboard_press(Key.down)
                write_text(image, "Under")
            else:
                before_position = 5
                write_text(image, "idle")

            delay_pose = max(delay_pose - 1, 0)

            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            # cv2.imshow("MediaPipe Pose", image)
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()        
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()
    cv2.destroyAllWindows()


# def gen(video):
#     while True:
#         success, image = video.read()
#         frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         frame_gray = cv2.equalizeHist(frame_gray)

#         faces = face_cascade.detectMultiScale(frame_gray)

#         for (x, y, w, h) in faces:
#             center = (x + w//2, y + h//2)
#             cv2.putText(image, "X: " + str(center[0]) + " Y: " + str(center[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
#             image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#             faceROI = frame_gray[y:y+h, x:x+w]
#         ret, jpeg = cv2.imencode('.jpg', image)

#         frame = jpeg.tobytes()
        
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/cam')
def video_feed():
		# Set to global because we refer the video variable on global scope, 
		# Or in other words outside the function
    global video

		# Return the result on the web
    return Response(main(video),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)