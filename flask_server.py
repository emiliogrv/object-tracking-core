import os
import threading
from base64 import b64decode

import firebase_admin
import requests
from dotenv import load_dotenv
from firebase_admin import credentials, storage
from flask import Flask, send_from_directory
from flask_cors import CORS

from detectors.yolo_deep_sort import YOLODeepSort
from helpers import is_url, hmac_generate, hmac_validate
from libs.request_validator import validate_request

load_dotenv()
app = Flask(__name__)
CORS(app)

app_env = os.getenv("APP_ENV", "local")
webhook_url = os.getenv("WEBHOOK_URL")

if app_env and app_env != "local":
    project_id = os.getenv("FIREBASE_PROJECT_ID")
    firebase_json = {
        "type": os.getenv("FIREBASE_TYPE"),
        "project_id": project_id,
        "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": b64decode(os.getenv("FIREBASE_PRIVATE_KEY")),
        "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.getenv("FIREBASE_CLIENT_ID"),
        "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
        "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.getenv(
            "FIREBASE_AUTH_PROVIDER_X509_CERT_URL"
        ),
        "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL"),
    }

    cred = credentials.Certificate(firebase_json)

    firebase_admin.initialize_app(cred, {"storageBucket": project_id + ".appspot.com"})

    # Put your local file path
    bucket = storage.bucket()

options = {
    "DEBUG": bool(os.getenv("APP_DEBUG")),
    "OUTPUT_GLOBAL_MAX_WIDTH": int(os.getenv("OUTPUT_GLOBAL_MAX_WIDTH") or 1920),
    "OUTPUT_GLOBAL_MAX_HEIGHT": int(os.getenv("OUTPUT_GLOBAL_MAX_HEIGHT") or 1080),
    "OUTPUT_GLOBAL_PATH": os.getenv("OUTPUT_GLOBAL_PATH") or "outputs/",
    "OUTPUT_IMAGE_QUALITY": int(os.getenv("OUTPUT_IMAGE_QUALITY") or 100),
    "OUTPUT_VIDEO_FRAMERATE": os.getenv("OUTPUT_VIDEO_FRAMERATE") or "1500k",
    "OUTPUT_VIDEO_MAX_SECONDS_LENGTH": int(
        os.getenv("OUTPUT_VIDEO_MAX_SECONDS_LENGTH") or 0
    ),
    "OUTPUT_VIDEO_SOURCE_QUALITY": os.getenv("OUTPUT_VIDEO_SOURCE_QUALITY") or "best",
}


def detect(detector: YOLODeepSort, source: str, output_filename: str):
    try:
        is_video, output_path = detector.detect(source, output_filename)

        if app_env and app_env != "local":
            blob = bucket.blob(output_path)
            blob.upload_from_filename(output_path)

            # Opt : if you want to make public access from the URL
            blob.make_public()

            data, headers = hmac_generate(
                {
                    "output_filename": output_filename,
                    "is_video": is_video,
                    "public_url": blob.public_url,
                }
            )

            if webhook_url:
                requests.post(webhook_url, data=data, headers=headers)
    except Exception as error:
        # TODO: make some useful message here
        # TODO: get traceback and use sentry
        print("ERROR: {}".format(error))


@app.route("/track-it", methods=["POST"])
@validate_request(**hmac_validate())
@validate_request(
    {
        "source": ["required", "string", is_url],
        "output_filename": "required|string",
    },
)
def track_it(payload):
    detector = YOLODeepSort(options)

    thread = threading.Thread(
        target=detect,
        args=(
            detector,
            payload["source"],
            payload["output_filename"],
        ),
    )
    thread.start()

    return "", 204


@app.route("/healthz")
def healthz():
    return "", 204


if app_env == "local":

    @app.route("/<output_file>")
    def output(output_file):
        return send_from_directory(options.get("OUTPUT_GLOBAL_PATH"), output_file)


if __name__ == "__main__":
    app.run(
        debug=bool(os.getenv("APP_DEBUG")),
        host=os.getenv("FLASK_HOST"),
        port=int(os.getenv("FLASK_PORT") or 5000),
        threaded=True,
    )
