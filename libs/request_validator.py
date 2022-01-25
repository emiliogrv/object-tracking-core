import inspect
import sys
from functools import wraps

from flask import request
from validator import validate


def validate_request(
    payload: dict = None,
    params: dict = None,
    headers: dict = None,
    messages: dict = None,
    status_code: int = 422,
    return_valid: bool = True,
    return_errors: bool = False,
    return_destructured: bool = True,
    formatter_errors: callable = None,
):
    """Flask requests object validator.

    Parameters
    ----------
    payload: dict
        Dictionary of rules for request.values and request.form.
        Works for: GET|POST query, form-data, x-www-form-urlencoded values.
        If request.is_json is True then it'll use request.json

    params: dict
        Dictionary of rules for URL params.
        Params values will be passed to function as flask default.

        @app.route("/profile/<username>")
        def profile(username):
            return f"Hello {username}"

    headers: dict
        Dictionary of rules for request headers.
        Headers will be compared with lowercase keys.

    messages: dict
        Dictionary text messages on error.

    status_code: int
        On error response's code.

    return_valid: bool
        Flag to return validated values or not.

    return_errors: bool
        Flag to return errors values or not.

    return_destructured: bool
        Flag to return **validated and **errors values or not.

    formatter_errors: callable
        Function to return personalized formatted errors.
        It'll receive unformatted errors dict and messages dict if provided.

    Returns
    -------
    func:
        Wrapped function with validated payload dict as first argument (as set up).
        If validation fails, then request will be stopped and return errors messages with status_code directly client
        or sent as first argument to Wrapped function (as set up).

    """

    def decorator(func: callable):
        def _format_errors_messages(errors: dict, messages_dict: dict = None):
            if messages_dict:
                _errors = []

                for k, v in errors.items():
                    key = k.lower()
                    _errors.append(messages_dict.get(key) or key + ": " + v)

                return _errors

            return [(k.lower() + ": " + v) for k, v in errors.items()]

        def _format_errors_levels(where: str, errors: dict, messages_dict: dict = None):
            return [
                (
                    {
                        "where": where,
                        "field": k,
                        "messages": _format_errors_messages(v, messages_dict),
                    }
                )
                for k, v in errors.items()
            ]

        def _format_errors(errors: dict, messages_dict: dict = None):
            _errors = []

            for k, v in errors.items():
                for kk in _format_errors_levels(k, v, messages_dict):
                    _errors.append(kk)

            return {"errors": _errors}

        def _wrap_rule(rule: callable, req: request):
            def wrap(value):
                return rule(value, request=req)

            if hasattr(rule, "__name__"):
                wrap.__name__ = rule.__name__

            return wrap

        def _check_rule_wrapper(rule):
            if isinstance(rule, str):
                return rule

            if inspect.isfunction(rule):
                if sys.version_info[0] < 3:
                    args = inspect.getargspec(rule)[0]
                else:
                    args = inspect.getfullargspec(rule).args

                if "request" in args:
                    return _wrap_rule(rule, request)

            if hasattr(rule, "request"):
                rule.request = request

            return rule

        def _check_rules(rules):
            if isinstance(rules, str):
                return rules

            if isinstance(rules, list):
                _rules = rules.copy()
            else:
                _rules = [rules]

            return [_check_rule_wrapper(rule) for rule in _rules]

        def _validator(errors: dict, validated: dict):
            def _validate(
                data: dict, rules: dict, key: str, ignore_validated: bool = False
            ):
                _, _validated, _errors = validate(
                    data, dict((k, _check_rules(v)) for k, v in rules.items()), True
                )

                if _errors:
                    errors.update({key: _errors})
                elif not ignore_validated:
                    validated.update({key: _validated})

            return _validate

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not payload and not params and not headers:
                raise Exception("You must provide at least one rules dict.")

            errors = {}
            validated = {}
            _validate = _validator(errors, validated)

            if payload:
                _validate(
                    request.json if request.is_json else request.values,
                    payload,
                    "payload",
                )

            if params:
                _validate(kwargs, params, "params", True)

            if headers:
                _validate(
                    dict((k.lower(), v) for k, v in request.headers),
                    headers,
                    "headers",
                )

            # On validation errors
            if len(errors):
                f_e = formatter_errors or _format_errors
                errors = f_e(errors, messages)

                if return_errors:
                    if return_destructured and type(errors) == "dict":
                        errors.update(kwargs)

                        return func(*args, **errors)
                    else:
                        return func(errors, *args, **kwargs)

                return errors, status_code

            # Free validation errors
            if return_valid:
                if return_destructured:
                    validated.update(kwargs)

                    return func(*args, **validated)
                else:
                    return func(validated, *args, **kwargs)

            return func(*args, **kwargs)

        return wrapper

    return decorator
