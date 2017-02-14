#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import cStringIO
from cStringIO import StringIO
import bson
import cv2
import flask
import numpy
from PIL import Image
from bson.json_util import dumps
from flask import Flask, request, Response, jsonify, redirect, json
from pymongo import MongoClient
from werkzeug.utils import secure_filename
import keras_predict
from imageFilter import ImageFilter

app = Flask(__name__, static_url_path="")

# 读取配置文件
app.config.from_object('config')

# 连接数据库，并获取数据库对象
db = MongoClient(app.config['DB_HOST'], app.config['DB_PORT']).test

# 将矫正后图片与图片识别结果（JSON）存入数据库
def save_file(file_str, f, report_data):
    content = StringIO(file_str)
    try:
#         mime = Image.open("temp_pics/region.jpg").format.lower()
        mime = Image.open(f).format.lower()
        print 'content of mime is：', mime
        if mime not in app.config['ALLOWED_EXTENSIONS']:
            raise IOError()
    except IOError:
        print "abort(400)"
#         abort(400)
    c = dict(report_data=report_data, content=bson.binary.Binary(content.getvalue()), 
             filename=secure_filename(f.name),mime=mime)
    db.files.save(c)
    return c['_id'], c['filename']

@app.route('/', methods=['GET', 'POST'])
def index():
    return redirect('/index.html')

#上传加载图片
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        if 'imagefile' not in request.files:
            return jsonify({"error": "No file part"})
        imgfile = request.files['imagefile']
        if imgfile.filename == '':
            return jsonify({"error": "No selected file"})
        if imgfile:
            img = cv2.imdecode(numpy.fromstring(imgfile.read(), numpy.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
            report_data = ImageFilter(image=img).ocr(22)
            if report_data == None:
                data = {
                    "error": 1,
                }
                return jsonify(data)

            with open('temp_pics/region.jpg','rb') as f:
                if f is None:
                    print 'Error! f is None!'
                else:
                    #定义file_str存储矫正后的图片文件f的内容（str格式）,方便之后对图片做二次透视以及将图片内容存储至数据库中
                    file_str = f.read()
                    #使用矫正后的图片，将矫正后图片与识别结果（JSON数据）一并存入mongoDB，
                    #这样前台点击生成报告时将直接从数据库中取出JSON数据，而不需要再进行图像透视，缩短生成报告的响应时间
                    fid, filename = save_file(file_str, f, report_data)
            print 'fid:', fid
            if fid is not None:
                templates = "<div><img id=\'filtered-report\' src=\'/file/%s\' class=\'file-preview-image\' width=\'100%%\' height=\'512\'></div>" % (fid)
                data = {
                    "templates": templates,
                }
            return jsonify(data)
    return jsonify({"error": "No POST methods"})

#  根据图像oid，在mongodb中查询，并返回Binary对象
@app.route('/file/<fid>')
def find_file(fid):
    try:
        file = db.files.find_one(bson.objectid.ObjectId(fid))
        if file is None:
            raise bson.errors.InvalidId()
        return Response(file['content'], mimetype='image/' + file['mime'])
    except bson.errors.InvalidId:
        flask.abort(404)

# 直接从数据库中取出之前识别好的JSON数据，并且用bson.json_util.dumps将其从BSON转换为JSON格式的str类型
@app.route('/report/<fid>')
def get_report(fid):
    try:
        file = db.files.find_one(bson.objectid.ObjectId(fid))
        if file is None:
            raise bson.errors.InvalidId()
        report_data = bson.json_util.dumps(file['report_data'])
        if report_data is None:
            print 'report_data is NONE! Error!!!!'
            return jsonify({"error": "can't ocr'"})
        return jsonify(report_data)
    except bson.errors.InvalidId:
        flask.abort(404)

#使用keras库对年龄，性别进行预测
@app.route("/predict", methods=['POST'])
def predict():
    print ("predict now!") 
    data = json.loads(request.form.get('data'))
    ss = data['value']
    arr = numpy.array(ss)
    arr = numpy.reshape(arr, [1, 22])
    #预测性别
    sex_array = keras_predict.predict_sex(arr)
    #预测年龄
    age = keras_predict.predict_age(arr)
    #判断是男是女，将原先的ndarray形式的性别转成int形式的性别，以供前端判断
    if sex_array==0:
        sex = 1
    else:
        sex = 0
        
    result = {
        "sex":sex,
        "age":int(age)
    }
 
    return json.dumps(result)

if __name__ == '__main__':
    app.run(host=app.config['SERVER_HOST'], port=app.config['SERVER_PORT'])