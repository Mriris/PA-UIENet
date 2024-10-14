from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import torch
from torchvision.utils import save_image
from werkzeug.utils import secure_filename
from net.Ushape_Trans import Generator
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # 上传的图像保存路径
app.config['ENHANCED_FOLDER'] = 'enhanced/'  # 增强后的图像保存路径

# 确保文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ENHANCED_FOLDER'], exist_ok=True)

# 加载预训练的图像增强模型
generator = Generator().cuda()
generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth"))
generator.eval()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255.0
    return img.cuda()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    # 保存上传的图像
    file = request.files['image']
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # 预处理并通过模型生成增强图像
    img_tensor = preprocess_image(input_path)
    with torch.no_grad():
        enhanced_img_tensor = generator(img_tensor)

    # 保存增强后的图像
    enhanced_path = os.path.join(app.config['ENHANCED_FOLDER'], filename)
    save_image(enhanced_img_tensor[3], enhanced_path)

    # 返回增强后的图像
    return send_file(enhanced_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
