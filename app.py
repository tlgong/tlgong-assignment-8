from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from logistic_regression import do_experiments

app = Flask(__name__)


# 定义主页路由
@app.route('/')
def index():
    return render_template('index.html')


# 处理实验参数并触发实验的路由
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    start = float(request.json['start'])
    end = float(request.json['end'])
    step_num = int(request.json['step_num'])

    # 使用提供的参数运行实验
    do_experiments(start, end, step_num)

    # 检查结果图像是否生成并返回它们的路径
    dataset_img = "results/dataset.png"
    parameters_img = "results/parameters_vs_shift_distance.png"

    return jsonify({
        "dataset_img": dataset_img if os.path.exists(dataset_img) else None,
        "parameters_img": parameters_img if os.path.exists(parameters_img) else None
    })


# 用于提供结果图像的路由
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
