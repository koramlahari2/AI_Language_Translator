import unittest

from app import LANGUAGES, validate_request


class ValidateRequestTests(unittest.TestCase):
    def test_rejects_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "Enter some text"):
            validate_request("  \n", LANGUAGES["English"], LANGUAGES["Hindi"])

    def test_rejects_unsupported_source_language(self) -> None:
        with self.assertRaisesRegex(ValueError, "source language"):
            validate_request("Hello", "unsupported", LANGUAGES["Hindi"])

    def test_rejects_unsupported_target_language(self) -> None:
        with self.assertRaisesRegex(ValueError, "target language"):
            validate_request("Hello", LANGUAGES["English"], "unsupported")

    def test_rejects_identical_languages(self) -> None:
        with self.assertRaisesRegex(ValueError, "different languages"):
            validate_request(
                "Hello",
                LANGUAGES["English"],
                LANGUAGES["English"],
            )

    def test_accepts_and_trims_supported_translation_request(self) -> None:
        self.assertEqual(
            validate_request(
                "  Hello, world!  ",
                LANGUAGES["English"],
                LANGUAGES["Telugu"],
            ),
            "Hello, world!",
        )


if __name__ == "__main__":
    unittest.main()
