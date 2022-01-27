# ❓ About this project

This project is made in order to learn and put into practice my knowledge of Python and some AI models implementation, it was born as a personal project after asking myself how many cars pass by my window every day.

I implemented Object detection using YOLOv4 and object tracking using DeepSort and TensorFlow.

It'll work fine with images and videos files and URLs as well as videos streaming platforms like YouTube.

https://user-images.githubusercontent.com/13983577/151461274-4e540ad6-cdc3-4078-b1b0-f966667ec3d8.mp4

# 💡 Getting Started

To see how this project works you have two options

### 🟡 Option 1: Online

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emiliogrv/object-tracking-core/blob/main/opencv_yolo_deep_sort.ipynb)

### 🟠 Option 2: In local?

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

# 🧰 References

1. [DeepSort YOLOv4 based object tracking](https://github.com/MatPiech/DeepSORT-YOLOv4-TensorRT-OpenVINO)
2. [yolov4-deepsort](https://github.com/theAIGuysCode/yolov4-deepsort)
3. [YOLOv4-Tutorials](https://github.com/augmentedstartups/YOLOv4-Tutorials/blob/master/3.%20YOLOv4%20Video%20and%20Webcam/darknet_video_mod.py#L21)
4. [Deep SORT](https://github.com/mk-michal/deep_sort)

# ❗ Troubleshooting

If you find any problem in my code or anything else, feel free to contact me, open an issue or do a pull request, that
way I can keep learning, and I can improve the code so that way anyone else can learn from it in the best way.




