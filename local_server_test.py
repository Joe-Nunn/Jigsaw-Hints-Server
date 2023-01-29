from flask import Flask, jsonify

#########################################################
#   Simple test program for sending a local request to a
#   Flask server using the 'requests' library in Python.
#   @author James Venables (jv264)
#   @version 1.0 - 28/01/2023
#########################################################

app = Flask(__name__)

@app.route('/endpoint')
def endpoint():
    return jsonify(message='This is your endpoint')

if __name__ == '__main__':
    app.run()

# This endpoint can be accessed locally by visiting http://localhost:5000/endpoint