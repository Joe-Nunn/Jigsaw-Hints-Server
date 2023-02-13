import flask
import time
import base64
import cv2
import os

import image_processor as ip
import sift_match as sift

from json_encoder import NumpyArrayEncoder

OUTPUT_DIR = "output/"

# Initialize the Flask server
app = flask.Flask(__name__)

# Set the JSON encoder to the custom one
app.json_encoder = NumpyArrayEncoder


@app.route("/process", methods=["POST"])
def process():
    """
    Processes a piece image using the image processor library.
    Then it feeds the AI model with base and piece images.
    Returns the location of the piece within the base to the caller.
    """

    print("Request received")

    # Get the request data
    json = flask.request.get_json()

    # Get the request time
    request_time = time.time_ns()

    # Decode the base64 data
    piece_data = base64.decodebytes(str.encode(json["piece_data"]))
    base_data = base64.decodebytes(str.encode(json["base_data"]))

    # Save the received base and piece images
    raw_piece_path = OUTPUT_DIR + str(request_time) + ".png"
    base_image_path = OUTPUT_DIR + str(request_time) + "_base.png"
    fh = open(raw_piece_path, "wb")
    fh.write(piece_data)
    fh.close()
    fh = open(base_image_path, "wb")
    fh.write(base_data)
    fh.close()

    # Process the piece image
    processed_piece_path = ip.process_from_code("\"" + OUTPUT_DIR + str(request_time) + ".png\"")[0]

    # Feed the AI model/SIFT
    base_cv2 = cv2.imread(base_image_path, cv2.IMREAD_UNCHANGED)
    piece_cv2 = cv2.imread(processed_piece_path, cv2.IMREAD_UNCHANGED)
    solved_piece_base64 = sift.find_match(base_cv2, piece_cv2, True, request_time)
    print("Solved piece coordinates: \n" + str(solved_piece_base64))

    # Return the location of the piece within the base
    out = {
        "request_id": request_time,
        "solved_data": solved_piece_base64,
    }

    # Return the response to the caller
    return flask.jsonify(out)


# Run the app
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    app.run()
