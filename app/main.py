from fastapi import FastAPI, File, UploadFile
from fastapi.responses import PlainTextResponse

import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc

URL = 'triton:8001'

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


@app.get('/live/')
def read_root():
    model_name = "inception_graphdef"
    model_version = ""
    message = ''

    # Create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(URL)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Health
    try:
        request = service_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request)
        message = "server {}".format(response)
        print(message)
    except Exception as ex:
        print(ex)

    request = service_pb2.ModelInferRequest()
    request.model_name = model_name
    request.model_version = model_version
    request.id = "my request id"

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = "input"
    input.datatype = "FP32"
    input.shape.extend([1, 299, 299, 3])
    request.inputs.extend([input])

    output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
    output.name = "InceptionV3/Predictions/Softmax"
    request.outputs.extend([output])

    request.raw_input_contents.extend([bytes(1072812 * 'a', 'utf-8')])

    response = grpc_stub.ModelInfer(request)
    print("model infer:\n{}".format(response))
    return message
