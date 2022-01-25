import hashlib
import hmac
import json
import os

from validators import ValidationFailure, url as v_url


def is_url(source):
    try:
        return v_url(source or "")
    except ValidationFailure:
        return False


def _hmac_generate_signature(message):
    return hmac.new(
        os.getenv("AUTH_PRIVATE_KEY").encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def hmac_generate(data: dict):
    message = json.dumps(data)

    signature = _hmac_generate_signature(message)

    return message, {
        "X-Api-Key": os.getenv("AUTH_PUBLIC_KEY"),
        "X-Api-Signature": signature,
    }


def hmac_validate():
    def _validate(value, request):
        if os.getenv("APP_ENV") == "local":
            return True

        message = json.dumps(request.json)

        signature = _hmac_generate_signature(message)

        return hmac.compare_digest(value or "", signature)

    return {
        "headers": {
            "x-api-signature": _validate,
        },
        "messages": {"_validate": "Invalid authentication credentials"},
        "status_code": 401,
        "return_valid": False,
    }
