import base64
import json


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


piece = encode_base64("piece.png")
base = encode_base64("base.jpg")

req_dict = {
    "algorithm_type": "SIFT",
    "hint_accuracy": 100,
    "number_of_pieces": 0,
    "piece_data": piece,
    "base_data": base
}

request = open("request.json", "wt")
request.write(json.dumps(req_dict))
request.close()
