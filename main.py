import cv2
import face_recognition

def get_location_encoding(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(image)
    face_encoding = face_recognition.face_encodings(image, face_location)
    return face_location, face_encoding

def main():
    # video_path = './test_file/Messi_vid.mp4'
    video_path=0
    save_flag=False
    test_img = cv2.imread("./test_file/Messi_1.png")
    _ , test_img_encodings= get_location_encoding(test_img)
    
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    frame=0    
    while True:
        ret_val, image = cap.read()
        face_locations, face_encodings =get_location_encoding(image)
        if (len(face_encodings)!=0):
            for count, face_encoding in enumerate (face_encodings):
                result = face_recognition.compare_faces(test_img_encodings, face_encoding)
                y1,x2,y2,x1 = face_locations[count][0],face_locations[count][1],face_locations[count][2],face_locations[count][3]
                cv2.putText(image, str(result[0]), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 4)
    
        cv2.imshow('Face Detection', image)

        # For saving Video
        if save_flag==True:
            if frame == 0:   # It's use to intialize video writer ones
                out = cv2.VideoWriter('outputt.avi',cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                        20,(image.shape[1],image.shape[0]))
                frame+=1
            out.write(image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if save_flag==True:
        out.release()
    cv2.destroyAllWindows()
         
if __name__ == '__main__':
    main()