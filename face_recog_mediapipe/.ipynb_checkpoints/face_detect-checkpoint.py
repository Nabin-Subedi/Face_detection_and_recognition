import cv2
import mediapipe as mp
import time
# import face_recognition

def detect_and_draw(image):
    face_detector=mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    results=face_detector.process(image)
    img_height=image.shape[0]
    img_width=image.shape[1]
    if results.detections != None:

        for face in results.detections:
            bounding_box= face.location_data.relative_bounding_box
            x= int(bounding_box.xmin * img_width)
            y= int(bounding_box.ymin * img_height)

            w= int(bounding_box.width *img_width)
            h= int(bounding_box.height*img_height)

            for landmarks in face.location_data.relative_keypoints:
                xl= int(landmarks.x*img_width)
                yl= int(landmarks.y*img_height)
                cv2.circle(image,(xl,yl),2,(255,0,0),2,)

            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

    
def main():
    save_vid= False
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    frame=0    
    while True:
        start_time= time.time()
        ret_val, image = cap.read()
        detect_and_draw(image)
        fps= str(int(1/(time.time() - start_time)))
        cv2.putText(image,fps, (10 ,40 ), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.imshow('Face Detection', image) 
        if save_vid==True:
            if frame == 0:
                out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                    20,(image.shape[1],image.shape[0]))
                frame+=1
            out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        
    cap.release()
    if save_vid==True:
        out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()