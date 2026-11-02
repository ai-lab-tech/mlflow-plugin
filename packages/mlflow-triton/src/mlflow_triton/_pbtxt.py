"""Protobuf text formatting without an application schema."""

import re

from google.protobuf import text_encoding


class EnumValue(str):
    """An unquoted Protobuf enum identifier, distinct from a string value."""

    def __new__(cls, value):
        if not isinstance(value, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", value
        ):
            raise ValueError("Invalid Protobuf enum identifier: " + repr(value))
        return super().__new__(cls, value)


def message_text(fields):
    """Format dict messages, repeated lists, strings/bytes, integers and booleans.

    Enum identifiers must be wrapped in EnumValue. Maps are represented explicitly
    as repeated messages with key/value fields. Application fields are not checked
    against a version-specific schema.
    """

    def field_lines(name, value, indent):
        if not isinstance(name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", name
        ):
            raise ValueError("Invalid Protobuf field identifier: " + repr(name))
        prefix = " " * indent + name
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from field_lines(name, item, indent)
        elif isinstance(value, dict):
            yield prefix + " {"
            for key, item in value.items():
                yield from field_lines(key, item, indent + 2)
            yield " " * indent + "}"
        else:
            if isinstance(value, EnumValue):
                literal = value
            elif isinstance(value, (str, bytes)):
                literal = '"' + text_encoding.CEscape(value, as_utf8=False) + '"'
            elif isinstance(value, bool):
                literal = "true" if value else "false"
            elif isinstance(value, int):
                literal = str(value)
            else:
                raise TypeError(
                    "Unsupported Protobuf text value: " + type(value).__name__
                )
            yield prefix + ": " + literal

    lines = [
        line for name, value in fields.items() for line in field_lines(name, value, 0)
    ]
    return "\n".join(lines) + ("\n" if lines else "")
