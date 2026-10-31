"""Protobuf text syntax and escaping, independent of Triton's schema."""

import unittest

from google.protobuf import descriptor_pb2, text_format, wrappers_pb2
from mlflow_triton._pbtxt import EnumValue, message_text


class ProtobufTextTests(unittest.TestCase):
    def test_nested_repeated_fields_enums_and_strings_parse_as_protobuf(self):
        name = 'café"\\\n\t\x001结果'
        text = message_text(
            {
                "name": name,
                "dependency": ["one.proto", "two.proto"],
                "options": {"cc_enable_arenas": True, "deprecated": False},
                "message_type": [
                    {
                        "name": "Example",
                        "field": [
                            {
                                "name": "value",
                                "number": 1,
                                "label": EnumValue("LABEL_OPTIONAL"),
                                "type": EnumValue("TYPE_STRING"),
                            }
                        ],
                    }
                ],
            }
        )
        parsed = text_format.Parse(text, descriptor_pb2.FileDescriptorProto())
        self.assertEqual(parsed.name, name)
        self.assertEqual(list(parsed.dependency), ["one.proto", "two.proto"])
        self.assertTrue(parsed.options.cc_enable_arenas)
        self.assertFalse(parsed.options.deprecated)
        field = parsed.message_type[0].field[0]
        self.assertEqual(field.number, 1)
        self.assertEqual(
            field.label, descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        )
        self.assertEqual(field.type, descriptor_pb2.FieldDescriptorProto.TYPE_STRING)

    def test_arbitrary_bytes_round_trip_without_hex_escape_ambiguity(self):
        value = bytes(range(256)) + b"\x001\xffabc"
        parsed = text_format.Parse(
            message_text({"value": value}), wrappers_pb2.BytesValue()
        )
        self.assertEqual(parsed.value, value)

    def test_new_fields_and_enum_names_need_no_schema_update(self):
        text = message_text(
            {"future_option": {"mode": EnumValue("FUTURE_MODE"), "count": 0}}
        )
        self.assertIn("future_option {", text)
        self.assertIn("mode: FUTURE_MODE", text)
        self.assertIn("count: 0", text)

    def test_identifiers_cannot_inject_text_format_fields(self):
        for name in ('bad"name', "name\nother: 1", ""):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    message_text({name: "value"})
                with self.assertRaises(ValueError):
                    EnumValue(name)


if __name__ == "__main__":
    unittest.main()
