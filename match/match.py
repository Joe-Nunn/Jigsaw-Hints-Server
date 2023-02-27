from abc import ABC, abstractmethod
import base64


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
