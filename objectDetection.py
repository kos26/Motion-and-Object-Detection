import cv2
import numpy as np

class objectDetection():
    def __init__(self, video_file, confidence_threshold = 0.50, non_maximums_suppression_threshold = 0.40):
        self.video_file = video_file
        self.confidence_threshold = confidence_threshold  # to classify only if the confidence value is above 50%
        self.non_maximums_suppression_threshold = non_maximums_suppression_threshold  # to remove redundant overlapping bounding boxes

        #Load names of classes and turn that into a list
        self.classes_name_file = "coco.names.txt"

        #Model configuration
        self.model_configuration = 'yolov3.cfg.txt'
        self.model_weights = 'yolov3.weights'

        self.classes = None
        with open(self.classes_name_file,'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        #Set up the net
        self.net = cv2.dnn.readNetFromDarknet(self.model_configuration, self.model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        cap = cv2.VideoCapture(self.video_file)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height =int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #Process inputs
        self.window_name = 'Object Detection Frame'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(self.window_name, 1000, 1000)

        self.input_width = 416
        self.input_height = 416
        self.fourcc = cv2.VideoWriter_fourcc(*"X264")
        out = cv2.VideoWriter("output_videos/object_detected/"+ video_file +".mp4", self.fourcc, 15.0, (frame_width, frame_height))
        count = 0
        image_name = self.video_file.split("/")
        while cv2.waitKey(1) < 0:

            #get frame from video
            self.ret, self.frame = cap.read()

            #Create a 4D blob from a frame 
            blob = cv2.dnn.blobFromImage(self.frame, 1/255, (self.input_width, self.input_height), [0,0,0], 1, crop = False)

            #Set the input the the net
            self.net.setInput(blob)
            outs = self.net.forward(self.get_output_names(self.net))

            self.post_process(self.frame, outs)

            out.write(self.frame.astype(np.uint8))
            if count == 120:
                cv2.imwrite("output_images/"+image_name[1][:-4]+"/obeject_detected.jpg", self.frame.astype(np.uint8))
            count = count+1

            #show the image
            cv2.imshow(self.window_name, self.frame)

    def post_process(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence_threshold:
                    centerX = int(detection[0] * frame_width)
                    centerY = int(detection[1] * frame_height)

                    width = int(detection[2]* frame_width)
                    height = int(detection[3]* frame_height)

                    left = int(centerX - width/2)
                    top = int(centerY - height/2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.non_maximums_suppression_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            self.draw_prediction(class_ids[i], confidences[i], left, top, left + width, top + height)


    def draw_prediction(self, class_id, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 1)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (class_id < len(self.classes))
            label = '%s:%s' % (self.classes[class_id], label)
        cv2.putText(self.frame, label, (left,top), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    def get_output_names(self, net):
        # Get the names of all the layers in the network
        layers_names = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
