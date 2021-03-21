import torch
from fastapi import FastAPI, File, Form,  Request, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates

from check_inference import check_inference, check_live
from inference import parse_image

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/")
async def root(request: Request, message='Sirius ML Face Recognition'):
    return templates.TemplateResponse("index.html", {"request": request, "message": message})


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
