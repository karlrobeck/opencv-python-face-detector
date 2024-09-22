import cv2
import typing
DEVICE_WIDTH = 640
DEVICE_HEIGHT = 480

def setup_resolution(device:cv2.VideoCapture):
    device.set(3,DEVICE_WIDTH)
    device.set(4,DEVICE_HEIGHT)
    return device 

def get_frame(device:cv2.VideoCapture) -> cv2.typing.MatLike:
    
    success, img = device.read()
    if success:
        return img
    else:
        raise ValueError(f"Unable to read frame")

def get_model(path:str) -> cv2.CascadeClassifier:
    
    return cv2.CascadeClassifier(f"models/{path}")

def set_rectangle(source:cv2.typing.MatLike,frame:typing.Sequence[cv2.typing.Rect],color:cv2.typing.Scalar,thickness:int):
    for (x,y,w,h) in frame:
        cv2.rectangle(source,(x,y),(x+w,y+h),color,thickness)

def collect_objects(sequence:list[typing.Sequence[cv2.typing.Rect]]) -> list[tuple[int,int,int,int]]:
    
    data = []

    for (x,y,w,h) in sequence:
        data.append((x,y,w,h))

    return data
def main():    
    capture_device = cv2.VideoCapture(0)
    
    capture_device = setup_resolution(capture_device)
    
    face_model = get_model("haarcascade_frontalface_default.xml")

    eye_model = get_model("haarcascade_eye.xml")

    while True:
        try:
            frame = get_frame(capture_device)

            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            
            faces = face_model.detectMultiScale(gray_frame, 1.3, 5)

            face_list = collect_objects(faces)
            
            if face_list:
                (x,y,w,h) = face_list[0]
                # crop
                cropped = gray_frame[y:y+h,x:x+w]
                eyes, numDetects = eye_model.detectMultiScale2(cropped,minNeighbors=10)
                if len(numDetects) == 2:
                    print("detected two eyes")
                    eyes[0][0] += x
                    eyes[1][0] += x
                    eyes[0][1] += y
                    eyes[1][1] += y
                    
                    for (e_x,e_y,e_w,e_h) in eyes:        
                        cv2.rectangle(frame,(e_x,e_y),(e_x+e_w,e_y+e_h),(255,0,0),4)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            
            cv2.imshow("Face Detector system",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()