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

class Coordinates:
    x:int
    y:int
    width:int
    height:int

class EyeObject:

    left:Coordinates
    right:Coordinates

    def __init__(self,eyes:tuple[typing.Sequence[cv2.typing.Rect],typing.Sequence[cv2.typing.Rect]]):
        self.left = Coordinates()
        self.right = Coordinates()
        
        self.left.x = eyes[0][0]
        self.left.y = eyes[0][1]
        self.left.width = eyes[0][2]
        self.left.height = eyes[0][3]
        self.right.x = eyes[1][0]
        self.right.y = eyes[1][1]
        self.right.width = eyes[1][2]
        self.right.height = eyes[1][3]

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
                if len(numDetects) != 2:
                    continue
                # fix eye coordinates
                eyes[0][0] += x
                eyes[1][0] += x
                eyes[0][1] += y
                eyes[1][1] += y
                eye_obj = EyeObject(eyes)
                left_eye = eye_obj.left
                right_eye = eye_obj.right
                cv2.rectangle(frame,(left_eye.x,left_eye.y),(left_eye.x+left_eye.width,left_eye.y+left_eye.height),(255,0,0),4)
                cv2.rectangle(frame,(right_eye.x,right_eye.y),(right_eye.x+right_eye.width,right_eye.y+right_eye.height),(255,0,0),4)  

                left_eye_center_x = left_eye.x + left_eye.width // 2
                left_eye_center_y = left_eye.y + left_eye.height // 2

                right_eye_center_x = right_eye.x + right_eye.width // 2
                right_eye_center_y = right_eye.y + right_eye.height // 2

                center_x = (left_eye_center_x + right_eye_center_x) // 2
                center_y = (left_eye_center_y + right_eye_center_y) // 2

                middle_rect_x = center_x - 100 // 2
                middle_rect_y = center_y - left_eye_center_y // 2

                roi_frame = frame[middle_rect_y:middle_rect_y+50,middle_rect_x:middle_rect_x+100]
                cv2.imshow("Region of interest",roi_frame)
                roi_filtered = cv2.cvtColor(roi_frame,cv2.COLOR_BGR2HSV)
                cv2.imshow("HSV Colored",roi_filtered)
                print(roi_filtered.mean())
                cv2.rectangle(frame,(middle_rect_x,middle_rect_y),(middle_rect_x + 100,middle_rect_y + 50),(0,255,0),4)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)
            
            cv2.imshow("Face Detector system",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()