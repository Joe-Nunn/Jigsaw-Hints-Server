from abc import ABC, abstractmethod
import base64
import cv2
import os
from PIL import Image


class Match(ABC):

    def __init__(self):
        pass

    @staticmethod
    def encode_base64(file):
        """
        Encode to base64
        :param file: file to encode
        :return: encoded file
        """
        fh = open(file, "rb")
        out_bytes = fh.read()
        fh.close()

        out_bytes_base64 = base64.b64encode(out_bytes)
        out_bytes_base64 = out_bytes_base64.decode("ascii")

        return out_bytes_base64

    @abstractmethod
    def find_match(self, base, piece, save_image, request_time, hint_accuracy, no_pieces):
        pass

    def save_solved_image(self, request_time, image):
        # Save image
        solved_png_path = "output/" + str(request_time) + "_solved.jpg"
        cv2.imwrite(solved_png_path, image)

        # Convert to webp
        solved_webp = Image.open(solved_png_path)
        solved_webp = solved_webp.convert("RGB")
        solved_webp_path = "output/" + str(request_time) + "_solved.webp"
        solved_webp.save(solved_webp_path, "webp")

        # Delete the PNG file
        try:
            os.remove(solved_png_path)
        except:
            pass

        # Encode to base64 and return
        return self.encode_base64(solved_webp_path)
