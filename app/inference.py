import sys

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

from const import URL, CLIENT_TIMEOUT


def load_image(filename):
    img = Image.open(filename)
    img.load()
    img = img.convert('RGB')
    resized_img = img.resize((640, 640), Image.BILINEAR)
    resized_img = np.array(resized_img)
    resized_img = resized_img.astype(np.float32)
    resized_img = np.transpose(resized_img, (2, 0, 1))
    return resized_img


def inference_client(file):
    try:
        triton_client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "facedetector"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 640, 640], "FP32"))

    # Initialize the data
    image_file = load_image(file)
    image_file = image_file[np.newaxis, ...]
    inputs[0].set_data_from_numpy(image_file)

    outputs.append(grpcclient.InferRequestedOutput('output__0'))
    outputs.append(grpcclient.InferRequestedOutput('output__1'))
    outputs.append(grpcclient.InferRequestedOutput('output__2'))

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        client_timeout=CLIENT_TIMEOUT,
        headers={'test': '1'},
    )

    # Get the output arrays from the results
    output0_data = results.as_numpy('output__0')
    output1_data = results.as_numpy('output__1')
    output2_data = results.as_numpy('output__2')

    print(output0_data.shape)
    print(output1_data.shape)
    print(output2_data.shape)
    return f'{output0_data.shape}, {output1_data.shape}, {output2_data.shape}'


def get_embedding(file):
    embedding = inference_client(file.file)
    return embedding


def get_user_by_photo(file, users):
    pass
    return users[0]
