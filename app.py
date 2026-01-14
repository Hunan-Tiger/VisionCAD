from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'screenshot/'

@app.route('/upload', methods=['POST'])
def upload_file():
    # 从请求头中提取文件名
    filename = request.headers.get('X-Filename')
    if not filename:
        return jsonify({"error": "No filename provided in headers"}), 400

    # 确保文件名是安全的
    filename = secure_filename(filename)
    filename = f'screenshot.png'

    # 获取原始二进制数据
    image_data = request.data
    if not image_data:
        return jsonify({"error": "No image data"}), 400

    # 保存文件
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    with open(save_path, 'wb') as f:
        f.write(image_data)

    return jsonify({"message": "File uploaded successfully", "filename": filename}), 200


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)