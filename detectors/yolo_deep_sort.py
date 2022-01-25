from time import sleep

import cv2
import magic
import numpy as np
import requests
import tensorflow as tf
from vidgear.gears import VideoGear, WriteGear

from config.color_dict import color_dict
from helpers import is_url
from libs.deep_sort import nn_matching, preprocessing
from libs.deep_sort.detection import Detection
from libs.deep_sort.tracker import Tracker
from libs.deep_sort_tools import generate_detections as gdet


class YOLODeepSort(object):
    options = {
        "DEBUG": False,
        "OUTPUT_GLOBAL_MAX_HEIGHT": 1080,
        "OUTPUT_GLOBAL_MAX_WIDTH": 1920,
        "OUTPUT_GLOBAL_PATH": "outputs/",
        "OUTPUT_IMAGE_QUALITY": 100,
        "OUTPUT_VIDEO_FRAMERATE": "1500k",
        "OUTPUT_VIDEO_MAX_SECONDS_LENGTH": 0,
        "OUTPUT_VIDEO_SOURCE_QUALITY": "best",
        "REQUEST_HEADERS": {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36",
        },
    }
    texts_sizes = {}

    def __init__(self, options: dict = None):
        if options is None:
            options = {}

        self._setup_options(options)
        self._init_names()
        self._init_tf()
        self._init_ds()
        self._init_yolo()

        # self.image_error = self.image_encode(self.image_read("error.jpg"), tobytes=True)
        # self.image_last = self.image_encode(
        #     self.image_read("last-frame.jpg"), tobytes=True
        # )

    def __del__(self):
        self.allowed_classes = None
        self.ds_encoder = None
        self.ds_tracker = None
        self.yolo_model = None

    def _setup_options(self, options: dict):
        self.options.update(options)

    def _init_names(self):
        with open("classes/coco.names", "r", encoding="utf-8") as f:
            self.class_names = f.read().splitlines()

        # by default allow all classes in .names file
        self.allowed_classes = self.class_names

        # custom allowed classes (uncomment line below to customize tracker for only people)
        # self.allowed_classes = ["person"]

    def _init_tf(self):
        # initialize tensorflow
        # https://github.com/hunglc007/tensorflow-yolov4-tflite/issues/171
        physical_devices = tf.config.experimental.list_physical_devices("GPU")

        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    def _init_ds(self):
        # Definition of the parameters
        max_cosine_distance = 0.9
        self.ds_nms_max_overlap = 1

        # initialize deep sort
        model_filename = "models/ds-mars-small128.pb"
        self.ds_encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance
        )
        self.ds_tracker = Tracker(metric, max_age=40)

    def _init_yolo(self):
        # initialize YOLOv4
        net = cv2.dnn.readNetFromDarknet("config/yolov4.cfg", "weights/yolov4.weights")
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.yolo_model = cv2.dnn_DetectionModel(net)
        self.yolo_model.setInputParams(scale=1 / 255, size=(608, 608), swapRB=True)

    def _setup_detection(self, source: str, output_filename: str):
        url, mime = self._get_source_mimetype(source)

        # stream = video; frame = image
        stream, frame = self._open_stream(source, url, mime)

        writer, output = self._init_writer(stream, output_filename, mime)

        if stream:
            # Getting first frame
            frame = stream.read()

            # Stopping if anything went wrong
            if frame is None:
                raise Exception("Reading source went wrong")

        dimensions = self._check_frame_resize_needs(*frame.shape[:2])

        return (stream, writer, frame, dimensions), output

    def _get_source_mimetype(self, source: str):
        url = is_url(source)

        if url:
            response = requests.head(
                source, headers=self.options.get("REQUEST_HEADERS")
            )

            mime = response.headers.get("Content-Type")
        else:
            mime = magic.from_file(source, mime=True)

        return url, mime

    def _open_stream(self, source: str, url: bool, mime: str):
        stream = frame = None

        if mime in ["image/jpeg", "image/png", "image/gif"]:
            frame = self._open_stream_image(source, url)
        else:
            stream = self._open_stream_video(source)

        return stream, frame

    def _open_stream_image(self, source: str, url: bool):
        if url:
            response = requests.get(source, headers=self.options.get("REQUEST_HEADERS"))
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)

            return cv2.imdecode(arr, -1)

        return cv2.imread(source)

    def _open_stream_video(self, source: str):
        options = {
            "STREAM_RESOLUTION": self.options.get("OUTPUT_VIDEO_SOURCE_QUALITY"),
            "STREAM_PARAMS": {"headers": self.options.get("REQUEST_HEADERS")},
        }
        stream_mode = True
        stream = None

        while True:
            try:
                stream = VideoGear(
                    source=source,
                    stream_mode=stream_mode,
                    logging=self.options.get("DEBUG"),
                    **options
                ).start()
            except ValueError:
                if stream_mode is False:
                    raise Exception("Opening source went wrong")

                stream_mode = False
                options = {}

            if stream is not None:
                break

        return stream

    def _init_writer(self, stream: VideoGear, output_filename: str, mime: str):
        output = self.options.get("OUTPUT_GLOBAL_PATH") + output_filename

        if stream:
            # define required FFmpeg parameters for your writer
            output_params = {
                "-movflags": "+faststart",
                "-input_framerate": stream.framerate,
            }

            output_framerate = self.options.get("OUTPUT_VIDEO_FRAMERATE")

            if output_framerate:
                output_params.update(
                    {
                        "-b:v": output_framerate,
                        "-maxrate": output_framerate,
                        "-bufsize": output_framerate,
                    }
                )

            output = output + ".mp4"
            return WriteGear(output, logging=True, **output_params), output

        output = output + mime.replace("image/", ".")

        class WI(object):
            def write(self, frame):
                cv2.imwrite(output, frame)

            def close(self):
                pass

        return WI(), output

    def _check_frame_resize_needs(self, height: int, width: int):
        dimensions = None
        overflow_width = 0
        overflow_height = 0
        max_width = self.options.get("OUTPUT_GLOBAL_MAX_WIDTH", 1920)
        max_height = self.options.get("OUTPUT_GLOBAL_MAX_HEIGHT", 1080)

        if max_width > 0 or max_height > 0:
            overflow_width = width / max_width
            overflow_height = height / max_height

        if overflow_width > 1 or overflow_height > 1:
            if overflow_width > overflow_height:
                dimensions = (int(width / overflow_width), int(height / overflow_width))
            else:
                dimensions = (
                    int(width / overflow_height),
                    int(height / overflow_height),
                )

        return dimensions

    def _process_source(
        self,
        stream: VideoGear,
        writer: any,
        frame: np.ndarray,
        dimensions: tuple[int, int],
    ):
        if stream:
            self._process_source_video(stream, writer, frame, dimensions)
        else:
            self._process_source_image(writer, frame, dimensions)

        writer.close()

    def _process_source_image(
        self, writer, frame: np.ndarray, dimensions: tuple[int, int]
    ):
        if dimensions:
            frame = self._resize_frame(frame, dimensions)

        detections = self._perform_yolo_detection(frame)

        self._draw_yolo_detections(frame, detections)

        writer.write(frame)

    def _process_source_video(
        self,
        stream: VideoGear,
        writer: any,
        frame: np.ndarray,
        dimensions: tuple[int, int],
    ):
        skip = self.options.get("OUTPUT_VIDEO_MAX_SECONDS_LENGTH")
        skip_count = 0

        if skip > 0:
            skip = int(stream.framerate) * skip

            if skip > 100:
                skip = 100

        while True:
            sleep(0.025)

            if frame is None:
                # Getting current frame
                frame = stream.read()

            # Stopping if anything went wrong
            if frame is None:
                break

            if dimensions:
                frame = self._resize_frame(frame, dimensions)

            detections = self._perform_yolo_detection(frame)
            detections = self._perform_deep_sort_detection(frame, detections)

            self._draw_deep_sort_detections(frame, detections)

            writer.write(frame)
            frame = None

            skip_count += 1
            if 0 < skip == skip_count:
                break

        stream.stop()

    def _resize_frame(
        self, frame: np.ndarray, dimensions: tuple[int, int]
    ) -> np.ndarray:
        return cv2.resize(
            frame,
            dimensions,
            interpolation=cv2.INTER_AREA,
        )

    def _perform_yolo_detection(self, frame: np.ndarray):
        class_ids, scores, boxes = self.yolo_model.detect(
            frame, confThreshold=0.8, nmsThreshold=0.4
        )
        class_ids = np.array(class_ids).flatten()

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_index = []
        for index, class_id in enumerate(class_ids):
            class_name = self.class_names[class_id]

            if class_name not in self.allowed_classes:
                deleted_index.append(index)
            else:
                names.append(class_name)

        # delete detections that are not in allowed_classes
        boxes = np.delete(boxes, deleted_index, axis=0)
        class_ids = np.delete(class_ids, deleted_index, axis=0)
        scores = np.delete(scores, deleted_index, axis=0)
        names = np.array(names)

        return boxes, names, class_ids, scores

    def _perform_deep_sort_detection(self, frame: np.ndarray, yolo_detections: tuple):
        boxes, names, _, scores = yolo_detections

        # encode yolo detections and feed to tracker
        features = self.ds_encoder(frame, boxes)
        detections = [
            Detection(box, score, class_name, feature)
            for box, score, class_name, feature in zip(boxes, scores, names, features)
        ]

        # run non-maxima suppression
        ds_boxes = np.array([d.tlwh for d in detections])
        ds_scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(
            ds_boxes, classes, self.ds_nms_max_overlap, ds_scores
        )
        detections = [detections[i] for i in indices]

        # call the tracker
        self.ds_tracker.predict()
        self.ds_tracker.update(detections)

        return self.ds_tracker.tracks

    def _draw_yolo_detections(self, frame: np.ndarray, detections: tuple):
        boxes, names, _, _ = detections

        for (box, name) in zip(boxes, names):
            x_min, y_min, w, h = box

            x_max = x_min + w
            y_max = y_min + h

            self._draw_detections(frame, x_min, y_min, x_max, y_max, name)

    def _draw_deep_sort_detections(self, frame: np.ndarray, detections: tuple):
        for track in detections:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlbr()
            class_name = track.class_name

            x_min = int(box[0])
            y_min = int(box[1])
            x_max = int(box[2])
            y_max = int(box[3])

            text = class_name + "-" + str(track.track_id)

            self._draw_detections(frame, x_min, y_min, x_max, y_max, class_name, text)

    def _draw_detections(
        self,
        frame: np.ndarray,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
        class_name: str,
        text: str = None,
    ):
        # draw box on screen
        color = color_dict[class_name]

        if text is None:
            text = class_name

        # drawing object box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # drawing text box
        size = self._get_text_size(frame, text)

        cv2.rectangle(
            frame,
            (x_min, y_min - 30),
            (
                x_min + size,
                y_min,
            ),
            color,
            -1,
        )

        # drawing text
        cv2.putText(
            frame,
            text,
            (x_min + 10, y_min - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            2,
        )

    def _get_text_size(self, frame: np.ndarray, text: str):
        size = self.texts_sizes.get(text)

        if not size:
            # create same size image of background color
            bg = np.full(frame.shape, (0, 0, 0), dtype=np.uint8)

            # draw text on bg
            cv2.putText(
                bg, text, (0, 30), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2
            )

            # get bounding box
            # use channel corresponding to color so that text is white on black background
            x, _, w, _ = cv2.boundingRect(bg[:, :, 2])

            size = w - x + 20

            self.texts_sizes[text] = size

        return size

    # def image_encode(
    #     self,
    #     frame,
    #     quality=100,
    #     tobytes=False,
    # ):
    #     ok, image = cv2.imencode(
    #         ".jpg",
    #         frame,
    #         [
    #             int(cv2.IMWRITE_JPEG_QUALITY),
    #             quality,
    #         ],
    #     )
    #
    #     if tobytes:
    #         return image.tobytes()
    #
    #     return ok, image

    # def image_read(self, path):
    #     return cv2.imread("Resources/" + path)

    def detect(self, source: str, output_filename: str):
        # https://www.youtube.com/watch?v=1EiC9bvVGnk  # https://worldcams.tv/united-states/jackson-hole/town-square
        # https://www.youtube.com/watch?v=XV1q_2Cl6mI  # https://worldcams.tv/china/taiwan/traffic
        # https://vimeo.com/588496072
        # https://vimeo.com/549069676
        # https://www.videezy.com/industry/5955-traffic-square-barcelona-spain-stock-video
        # https://www.videezy.com/urban/4298-random-cars-driving-by-4k-stock-video
        # https://motchallenge.net/sequenceVideos/MOT16-13-raw.mp4
        # https://s3-us-west-2.amazonaws.com/uw-s3-cdn/wp-content/uploads/sites/6/2017/11/04133712/waterfall-750x500.jpg

        setup, output = self._setup_detection(source, output_filename)
        self._process_source(*setup)

        is_video = bool(setup[0])

        if self.options.get("DEBUG"):
            # TODO: make some useful message here
            print("DEBUG: {} processed".format(output))

        return is_video, output
