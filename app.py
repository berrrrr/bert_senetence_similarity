import json
import logging

from flask import Flask, jsonify, request

import service

app = Flask(__name__)
app.config.from_object(__name__)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


@app.route('/health_check', methods=['GET'])
def health_check():
    return jsonify('good')


@app.route("/predict", methods=["POST"])
def predict():
    received_data = request.get_json()
    logger.info('# received_data' + str(received_data))

    result = service.predict(received_data)
    logging.info('# result' + str(result))

    resp = jsonify(result)
    resp.status_code = 200

    return resp


if __name__ == '__main__':
    PORT = 5000

    app.config['JSON_AS_ASCII'] = False
    app.run(host="0.0.0.0", debug=True, port=PORT)
