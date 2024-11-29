
import os
import random
import pathlib
import base64
from flask import Flask, request, redirect, url_for

asset_dir = os.environ.get('ASSET', '/assets')
out_path = os.environ.get('OUT_PATH', '/data.csv')

app = Flask(__name__)

@app.get('/')
def index():
    paths = list(pathlib.Path(asset_dir).glob('*.png'))
    imgpath = random.choice(paths)
    with open(imgpath, 'rb') as fp:
        imgdata = base64.b64encode(fp.read()).decode('ascii')
    return f"""<p>label image</p>
<img src="data:image/png;base64,{imgdata}" />
<form action="/collect" method="post">
    <input type="hidden" name="imageid" value="{imgpath.stem}" />
    <input type="text" name="label" />
    <input type="submit" />
</form>"""

@app.post('/collect')
def collect():
    with open(out_path, 'a') as fp:
        fp.write(f'{request.form["imageid"]},{request.form["label"]}\n')
    return redirect(url_for('index'))
