#!/usr/bin/python3 
from flask import Flask, request, jsonify
from base64 import b64decode

app = Flask(__name__)




@app.route('/api/img', methods=['POST'])
def upload():
    image = request.json['img']
    if (not image or type(image) is not str or not image.startswith('data:image')):
        return jsonify({'result':'no "img" uploaded'}), 400
    image_id = request.json['id']
    if (not image_id):
        return jsonify({'result':'no "image_id" passed'}), 400
    image = image.split(',')[1]
    with open(image_id+".png","wb") as f:
        f.write(b64decode(image))
    return jsonify({'result':'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
