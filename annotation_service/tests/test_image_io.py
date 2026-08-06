import unittest
from io import BytesIO

from PIL import Image

from annotation_service.errors import (
    RequestTooLargeError,
    UnsupportedMediaTypeError,
    ValidationServiceError,
)
from annotation_service.image_io import (
    JPEG_SIGNATURE,
    PNG_SIGNATURE,
    detect_image_format,
    encoded_image_dimensions,
    validate_image_bytes,
)


def image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (3, 2),
    color: tuple[int, int, int] = (10, 20, 30),
) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color).save(output, format=image_format)
    return output.getvalue()


class ImageIOTest(unittest.TestCase):
    def test_valid_png_and_jpeg(self):
        png = validate_image_bytes(
            image_bytes("PNG"),
            max_image_bytes=1024,
            max_image_pixels=100,
        )
        jpeg = validate_image_bytes(
            image_bytes("JPEG"),
            max_image_bytes=1024,
            max_image_pixels=100,
        )

        self.assertEqual(png.media_type, "image/png")
        self.assertEqual(png.image_format, "png")
        self.assertEqual((png.width, png.height), (3, 2))
        self.assertEqual(jpeg.media_type, "image/jpeg")
        self.assertEqual(jpeg.image_format, "jpeg")

    def test_header_dimensions_are_read_before_decode(self):
        png_raw = image_bytes("PNG", (11, 7))
        jpeg_raw = image_bytes("JPEG", (13, 9))
        self.assertEqual(detect_image_format(png_raw), "png")
        self.assertEqual(detect_image_format(jpeg_raw), "jpeg")
        self.assertEqual(
            encoded_image_dimensions(png_raw, "png"),
            (11, 7),
        )
        self.assertEqual(
            encoded_image_dimensions(jpeg_raw, "jpeg"),
            (13, 9),
        )

    def test_rejects_unsupported_and_corrupt_images(self):
        with self.assertRaises(UnsupportedMediaTypeError):
            validate_image_bytes(
                b"GIF89a",
                max_image_bytes=1024,
                max_image_pixels=100,
            )
        with self.assertRaises(ValidationServiceError):
            validate_image_bytes(
                PNG_SIGNATURE + b"\x00" * 32,
                max_image_bytes=1024,
                max_image_pixels=100,
            )
        with self.assertRaises(ValidationServiceError):
            validate_image_bytes(
                JPEG_SIGNATURE + b"\x00" * 32,
                max_image_bytes=1024,
                max_image_pixels=100,
            )

    def test_rejects_byte_and_pixel_limits(self):
        raw = image_bytes("PNG", (20, 20))
        with self.assertRaises(RequestTooLargeError):
            validate_image_bytes(
                raw,
                max_image_bytes=len(raw) - 1,
                max_image_pixels=1000,
            )
        with self.assertRaises(ValidationServiceError):
            validate_image_bytes(
                raw,
                max_image_bytes=1024,
                max_image_pixels=399,
            )


if __name__ == "__main__":
    unittest.main()
