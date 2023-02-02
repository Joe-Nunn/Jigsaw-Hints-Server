import flask
import image_processor as ip
import time
import base64

app = flask.Flask(__name__)


@app.route("/process", methods=["POST"])
def process():
    """
    Processes an image using the image processor library and sends it back to the client.
    """
    json = flask.request.get_json()
    request_time = time.time_ns()
    imagedata = str.encode(json["imagedata"])
    imagedata2 = base64.decodebytes(imagedata)

    fh = open(str(request_time) + ".png", "wb")
    fh.write(imagedata2)
    fh.close()

    output_path = ip.process_from_code("\"" + str(request_time) + ".png\"")[0]

    fh = open(output_path, "rb")
    out_bytes = fh.read()
    fh.close()

    out_bytes_base64 = base64.b64encode(out_bytes)

    out = {
        "request_id": request_time,
        "data": out_bytes_base64.decode("ascii")
    }

    return flask.jsonify(out)


if __name__ == "__main__":
    app.run()
