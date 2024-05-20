import cv2
import time
import face_recognition
import numpy as np
import mediapipe as mp

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter


#library for fast API
from typing import List
from fastapi import FastAPI,status,File, UploadFile
from pydantic import BaseModel


app = FastAPI()
# limiter = FastAPILimiter(
#     app,
#     key_func=lambda request: request.client.host,
#     default_limits=["100/minute"]
# )
@app.get('/')
def hello():
    return {"message":"Welcom to hand pose estimation  API"}

# 
# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils
 

# Function to check if hand pose resembles "OK" or "Yes" pose
def is_okay_pose(hand_landmarks):
    # Extract landmarks for specific fingers
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP]
    index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP]

    # Calculate the distances between specific finger landmarks
    dist_index_thumb = abs(index_finger_tip.x - thumb_tip.x)
    dist_middle_thumb = abs(middle_finger_tip.x - thumb_tip.x)
    dist_ring_thumb = abs(ring_finger_tip.x - thumb_tip.x)
    dist_pinky_thumb = abs(pinky_tip.x - thumb_tip.x)

    # Check if the distances meet the criteria for "OK" or "Yes" pose
    if (dist_index_thumb < 0.1 and dist_middle_thumb < 0.1 and
            dist_ring_thumb < 0.1 and dist_pinky_thumb < 0.1) and (thumb_tip.y < thumb_ip.y):
        return True
    else:
        return False


def ThumbPose(image1,imageprofile):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image of the hands.
        draw:    A boolean value that is if set to true the function writes the total count of fingers of the hands on the
                output image.
        display: A boolean value that is if set to true the function displays the resultant image and returns nothing.
    Returns:
        output_image:     A copy of the input image with the fingers count written, if it was specified.
        fingers_statuses: A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:            A dictionary containing the count of the fingers that are up, of both hands.
    '''
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    imgRGBprofile = cv2.cvtColor(imageprofile, cv2.COLOR_BGR2RGB)
  

    pose_encoding = face_recognition.face_encodings(imgRGB)[0]
    profile_encoding = face_recognition.face_encodings(imgRGBprofile)[0]

    profileresponse= face_recognition.compare_faces([pose_encoding], profile_encoding)
    print(profileresponse)
    if(profileresponse== [True]):
        # Perform the Hands Landmarks Detection.
        results = hands.process(imgRGB)
       
        # Check if the hands landmarks in the frame are detected.
        if results.multi_hand_landmarks:
            for hand_index, hand_info in enumerate(results.multi_handedness):
            
                # Retrieve the label of the found hand.
                hand_label = hand_info.classification[0].label
                print(hand_label)
                if len(results.multi_hand_landmarks) > 1:
                    print('false')
                    return { "Status":"Only show one hand Pose"}
                else:
                    
                    # Retrieve the landmarks of the found hand.
                    hand_landmarks =  results.multi_hand_landmarks[hand_index]
                    
                    # hand_landmarks =  results.multi_hand_landmarks
                

                   
                    print(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y)
                    
                    if is_okay_pose(hand_landmarks):
                         return { "Status":"verified"} 
                    else:
                         return { "Status":"Hand Pose not matching"}
                    
                   
        else:
               
            return { "Status":"Hand Pose Not detected"}
    else:
        return{ "Status":"Face Not Match"}

def FingerPose(image,imageprofile):
    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    imgRGBprofile = cv2.cvtColor(imageprofile, cv2.COLOR_BGR2RGB)
  

    pose_encoding = face_recognition.face_encodings(imgRGB)[0]
    profile_encoding = face_recognition.face_encodings(imgRGBprofile)[0]

    profileresponse= face_recognition.compare_faces([pose_encoding], profile_encoding)
    print(profileresponse)
    if(profileresponse== [True]):
    
        # Perform the Hands Landmarks Detection.
        results = hands.process(imgRGB)
        # Check if the hands landmarks in the frame are detected.

        # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
        fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                            'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                            'LEFT_RING': False, 'LEFT_PINKY': False}
        
        # Store the indexes of the tips landmarks of each finger of a hand in a list.
        fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
        
        # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
        fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                            'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                            'LEFT_RING': False, 'LEFT_PINKY': False}
        
        if results.multi_hand_landmarks:
            # Iterate over the found hands in the image.
            for hand_index, hand_info in enumerate(results.multi_handedness):
                
                # Retrieve the label of the found hand.
                hand_label = hand_info.classification[0].label


                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                    print('false')
                    return { "Status":"Only show one hand Pose"}
                else:
                
                    # Retrieve the landmarks of the found hand.
                    hand_landmarks =  results.multi_hand_landmarks[hand_index]
                    
                    # Iterate over the indexes of the tips landmarks of each finger of the hand.
                    for tip_index in fingers_tips_ids:
                        
                        # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
                        finger_name = tip_index.name.split("_")[0]
                        
                        # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
                        if (hand_landmarks.landmark[tip_index].y < hand_landmarks.landmark[tip_index - 2].y):
                            
                            # Update the status of the finger in the dictionary to true.
                            fingers_statuses[hand_label.upper()+"_"+finger_name] = True
                            
                        
                    
                    # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
                    thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x
                    
                    # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
                    if (hand_label=='Right' and (thumb_tip_x < thumb_mcp_x)) or (hand_label=='Left' and (thumb_tip_x > thumb_mcp_x)):
                        
                        # Update the status of the thumb in the dictionary to true.
                        fingers_statuses[hand_label.upper()+"_THUMB"] = True
                        
                        
                
                
               
                if (fingers_statuses['RIGHT_INDEX']==True and fingers_statuses['RIGHT_MIDDLE'] == True and fingers_statuses['RIGHT_RING'] ==True and fingers_statuses['RIGHT_PINKY']== False and fingers_statuses['RIGHT_THUMB'] == False )or (fingers_statuses['LEFT_INDEX']==True and fingers_statuses['LEFT_MIDDLE'] == True and fingers_statuses['LEFT_RING'] ==True  and fingers_statuses['LEFT_PINKY']== False and fingers_statuses['LEFT_THUMB'] == False):
                     return{ "Status":"Verified"}
                
                    
                    
                else:
                    return{ "Status":"Verified"}

        else:
            return{ "Status":"Hand Pose Not detected"}     
    
    else:
        return{ "Status":"Face Not Match"}
    
    return { "Status":'Try Again'}

        
        # Count the number of fingers up of each hand in the frame.
        # frame, fingers_statuses, count = countFingers(frame, results, display=False)


# api for find sentiment in the text
@app.post("/ThumbPoseImage", status_code=status.HTTP_201_CREATED )
# @limiter.limit("10/minute")  # Specific rate limit for this route
async def PoseEstimate(hands_image: UploadFile = File(...),profieImage: UploadFile = File(...)):
    contentsHand =  await hands_image.read()
    np_imageHand = np.fromstring(contentsHand, np.uint8)
    imageHand = cv2.imdecode(np_imageHand, cv2.IMREAD_COLOR)
    contentsHand =  await profieImage.read()
    np_imageprofile = np.fromstring(contentsHand, np.uint8)
    imageprofile = cv2.imdecode(np_imageprofile, cv2.IMREAD_COLOR)
    return ThumbPose(imageHand,imageprofile)  
 


 
@app.post("/FingerPoseImage", status_code=status.HTTP_201_CREATED )
# @limiter.limit("10/minute")  # Specific rate limit for this route
async def PoseEstimate(hands_image: UploadFile = File(...),profieImage: UploadFile = File(...)):
    contents =  await hands_image.read()
    np_image = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    contentsHand =  await profieImage.read()
    np_imageprofile = np.fromstring(contentsHand, np.uint8)
    imageprofile = cv2.imdecode(np_imageprofile, cv2.IMREAD_COLOR)

    return FingerPose(image,imageprofile)  