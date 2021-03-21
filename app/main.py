import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse

from check_inference import check_inference, check_live
from inference import parse_image

app = FastAPI()


@app.get('/')
def read_root():
    return 'Hello'


@app.post('/parse/', response_class=PlainTextResponse)
async def create_upload_file(file: UploadFile = File(...)):
    result = await parse_image(file)
    return f'{result}'


@app.get('/is_live/')
def is_live():
    return check_live()


@app.get('/is_inference/')
def is_inference():
    return check_inference()
