# Object Tracking

Object detection using YOLOv4 and object tracking using DeepSort and TensorFlow.

It'll work fine with images and videos files as well as videos streaming platforms like YouTube.

# Want to test in local?

1. Clone the repository.
2. Download [YOLO weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) into **weights** folder.
3. Copy  `.env.example` to `.env`.
4. Run `docker-compose build`.
5. Run `docker-compose up`.
6. Open Postman or similar and make a POST to `http://127.0.0.1:5000/track-it`
7. Payload:
   ```json
    {
      "output_filename": "some_filename_here",
      "source": "your image or video URL here"
    }
      ```
8. Wait until terminal "processed" message.
9. Open **outputs** folder and see the result.

NOTE: By default, this will run only with CPU support.

# References

1. [DeepSort YOLOv4 based object tracking](https://github.com/MatPiech/DeepSORT-YOLOv4-TensorRT-OpenVINO)
2. [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
3. [YOLOv4-Tutorials](https://github.com/augmentedstartups/YOLOv4-Tutorials/blob/master/3.%20YOLOv4%20Video%20and%20Webcam/darknet_video_mod.py#L21)
4. [Deep SORT](https://github.com/mk-michal/deep_sort)




