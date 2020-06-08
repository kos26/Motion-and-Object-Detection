import cv2
import numpy as np
import datetime
import sys
import objectDetection

def motion_detect(video_file):

    cap = cv2.VideoCapture(video_file)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    video_name = video_file.split("/")
    fourcc = cv2.VideoWriter_fourcc(*"X264")
    out_colored = cv2.VideoWriter("output_videos/colored/"+ video_name[1], fourcc, 15.0, (frame_width, frame_height))
    out_blurred = cv2.VideoWriter("output_videos/blurred/"+ video_name[1], fourcc, 15.0, (frame_width, frame_height))
    out_dilated = cv2.VideoWriter("output_videos/dilated/"+ video_name[1], fourcc, 15.0, (frame_width, frame_height))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    count = 0 

    while cap.isOpened():
        
        total_frames = total_frames + 1
        
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Movement: {}".format('True'), (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)

            fps_text = "Frames/Second: {:.2f}".format(fps)

            cv2.putText(frame1, fps_text, (5, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        colored_image = cv2.resize(frame1, (frame_width, frame_height))
        blurred_image = cv2.resize(blur, (frame_width, frame_height))
        dilated_image = cv2.resize(dilated, (frame_width, frame_height))
        out_colored.write(frame1.astype(np.uint8))
        out_blurred.write(blurred_image)
        out_dilated.write(dilated_image)
        image_name = video_file.split("/")
        if count == 120:
            cv2.imwrite("output_images/"+image_name[1][:-4]+"/colored.jpg", frame1)
            cv2.imwrite("output_images/"+image_name[1][:-4]+"/blurred.jpg", blur)
            cv2.imwrite("output_images/"+image_name[1][:-4]+"/dilated.jpg", dilated)
        cv2.imshow("Blur Frame", blur)
        cv2.imshow("Dilated Frame", dilated)
        cv2.imshow("Color Frame", frame1)

        count = count+1

        frame1 = frame2
        ret, frame2 = cap.read()
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    if str(sys.argv[1]) == "motion":
        motion_detect(str(sys.argv[2]))
    elif str(sys.argv[1]) == "objectDetection":
        objectDetection.objectDetection(str(sys.argv[2]))
    
    
