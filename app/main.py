from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse

app = FastAPI()


async def parse_image(file):
    try:
        file_bytes = await file.read()
        embedding = len(file_bytes)
        user_id = file.filename
        file.close()
    except Exception as e:
        print(e)
        return 'Error parsing'

    return user_id, embedding


@app.get('/')
def read_root():
    return 'Hello'


@app.post('/parse/', response_class=PlainTextResponse)
async def create_upload_file(file: UploadFile = File(...)):
    result = await parse_image(file)
    return f'{result}'
