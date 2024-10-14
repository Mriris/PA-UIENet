import os
import numpy as np
import torch
from torch.autograd import Variable
from flask import Flask, request, send_file, render_template
from werkzeug.utils import secure_filename
import cv2
from torchvision.utils import save_image
from net.Ushape_Trans import Generator  # 导入你的生成器模型
from loss.LAB import *  # 根据需要导入的其他模块

# 配置Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['OUTPUT_FOLDER'] = 'output/'

# 创建文件夹用于保存上传和处理后的图像
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# 初始化模型并加载预训练权重
generator = Generator().cuda()
generator.load_state_dict(torch.load("./saved_models/G/generator_795.pth"))
generator.eval()


# 图片缩放函数
def preprocess_image(img_path):
    imgx = cv2.imread(img_path)
    imgx = cv2.resize(imgx, (256, 256))  # 缩放到256x256
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = np.array(imgx).astype(np.float32)
    imgx = torch.from_numpy(imgx).permute(2, 0, 1).unsqueeze(0)  # 转换为张量
    imgx = imgx / 255.0  # 归一化
    imgx = Variable(imgx).cuda()
    return imgx


# 路由：主页
@app.route('/')
def index():
    return render_template('index.html')  # 显示HTML页面


# 路由：处理上传并增强图像
@app.route('/enhance', methods=['POST'])
def enhance_image():
    if 'image' not in request.files:
        return '没有上传图像'

    file = request.files['image']
    if file.filename == '':
        return '没有选择图像'

    # 保存上传的图像
    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    # 预处理并通过模型生成增强图像
    imgx = preprocess_image(input_path)
    with torch.no_grad():
        output = generator(imgx)

    # 保存增强后的图像
    output_img_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    save_image(output[3], output_img_path, nrow=1, normalize=True)

    # 返回增强后的图像
    return send_file(output_img_path, mimetype='image/jpeg')


# 路由：显示PSNR计算结果（可选）
@app.route('/psnr', methods=['POST'])
def compute_psnr_route():
    if 'image1' not in request.files or 'image2' not in request.files:
        return '需要两张图像进行PSNR计算'

    # 获取图像
    file1 = request.files['image1']
    file2 = request.files['image2']

    filename1 = secure_filename(file1.filename)
    filename2 = secure_filename(file2.filename)

    img1_path = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
    img2_path = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

    file1.save(img1_path)
    file2.save(img2_path)

    # 读取图像并计算PSNR
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))

    psnr_value = compute_psnr(img1, img2)
    return f"PSNR: {psnr_value:.2f}"


# PSNR计算函数
def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


if __name__ == '__main__':
    app.run(debug=True)
