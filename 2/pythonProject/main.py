from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, "objects.jpeg"),
                                             output_image_path=os.path.join(execution_path, "imagenew.jpg"))

for eachObject in detections:
    print(f'{eachObject["name"]} : {eachObject["percentage_probability"]}')