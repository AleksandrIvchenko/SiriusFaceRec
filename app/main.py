import json
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

import crud
import models
from check_inference import check_inference, check_live
from database import SessionLocal, engine
from inference import get_embedding, get_user_by_photo, get_embedding2

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_emb_and_create_user(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        emb = json.dumps(get_embedding(file.file).tolist())
    except:
        return 'На фотографии не обнаружено лицо'
    crud.create_user(db=db, name=name, emb=emb, filename=file.filename)
    return f'Создан пользователь {name}'


def _find_user_by_photo(file: UploadFile = File(...), db: Session = Depends(get_db)):
    users = crud.get_users(db)
    try:
        user_name = get_user_by_photo(file.file, users)
        if user_name == '':
            return 'Пользователя нет в базе'
    except:
        return 'На фотографии не обнаружено лицо'
    return f'Это пользователь {user_name}'


@app.get('/')
async def root(request: Request, message='Добро пожаловать!'):
    return templates.TemplateResponse('index.html', {'request': request, 'message': message})


@app.get('/users/')
def read_users(request: Request, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return templates.TemplateResponse('users.html', {'request': request, 'users': users})


@app.post("/users/")
def create_user(request: Request, name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    message = _get_emb_and_create_user(name, file, db)
    return templates.TemplateResponse('index.html', {'request': request, 'message': message})


@app.post('/add/', response_class=PlainTextResponse)
async def create_user_api(name: str = Form(...), file: UploadFile = File(...), db: Session = Depends(get_db)):
    return _get_emb_and_create_user(name, file, db)


@app.post('/inference_photo/')
def inference_photo_web(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    message = _find_user_by_photo(file, db)
    return templates.TemplateResponse('index.html', {'request': request, 'message': message})


@app.post('/parse/', response_class=PlainTextResponse)
async def inference_photo_api(file: UploadFile = File(...), db: Session = Depends(get_db)):
    return _find_user_by_photo(file, db)


@app.get('/is_live/')
def is_live():
    return check_live()


@app.get('/is_inference/')
def is_inference():
    return check_inference()


@app.post('/test_normalize/')
async def test_normalize(file: UploadFile = File(...)):
    return StreamingResponse(get_embedding2(file.file), media_type="image/png")
