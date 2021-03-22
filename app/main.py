from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

import crud
import models
from check_inference import check_inference, check_live
from database import SessionLocal, engine
from inference import get_embedding, get_user_by_photo

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

templates = Jinja2Templates(directory="templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get('/')
async def root(request: Request, message='Добро пожаловать!'):
    return templates.TemplateResponse('index.html', {'request': request, 'message': message})


@app.get('/users/')
def read_users(request: Request, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = crud.get_users(db, skip=skip, limit=limit)
    return templates.TemplateResponse('users.html', {'request': request, 'users': users})


@app.post("/users/")
def create_user(
        request: Request,
        db: Session = Depends(get_db),
        name: str = Form(...),
        file: UploadFile = File(...),
):
    try:
        emb = get_embedding(file)
    except Exception:
        emb = '123'
    crud.create_user(db=db, name=name, emb=emb, filename='test')
    message = f'Создан пользователь {name}'
    return templates.TemplateResponse('index.html', {'request': request, 'message': message})


@app.post('/inference_photo/')
def inference_photo(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    users = crud.get_users(db)
    user = get_user_by_photo(file, users)
    message = f' Это пользователь {user.name}'
    return templates.TemplateResponse('index.html', {'request': request, 'message': message})


@app.post('/parse/', response_class=PlainTextResponse)
async def create_upload_file(file: UploadFile = File(...)):
    result = get_embedding(file)
    return f'{result}'


@app.get('/is_live/')
def is_live():
    return check_live()


@app.get('/is_inference/')
def is_inference():
    return check_inference()
