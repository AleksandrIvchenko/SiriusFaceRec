import sys

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

from const import URL, CLIENT_TIMEOUT


def get_triton_client() -> grpcclient.InferenceServerClient:
    try:
        triton_client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def load_image(file):
    img = Image.open(file)
    img.load()
    img = img.convert('RGB')
    resized_img = img.resize((640, 640), Image.BILINEAR)
    resized_img = np.array(resized_img)
    resized_img = resized_img.astype(np.float32)
    resized_img = np.transpose(resized_img, (2, 0, 1))
    resized_img = resized_img[np.newaxis, ...]
    return resized_img


def detector(image_file):
    triton_client = get_triton_client()
    model_name = "facedetector"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 640, 640], "FP32"))

    # Initialize the data

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
    return output0_data, output1_data, output2_data


def extractor(o1, o2, o3):
    triton_client = get_triton_client()
    model_name = "extractor"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 128, 128], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(o1)

    outputs.append(grpcclient.InferRequestedOutput('output__0'))

    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        client_timeout=CLIENT_TIMEOUT,
        headers={'test': '1'},
    )

    # Get the output arrays from the results
    embedding = results.as_numpy('output__0')
    return embedding


def get_embedding(file):
    image_array = load_image(file.file)
    o1, o2, o3 = detector(image_array)
    embedding = extractor(o1, o2, o3)
    return embedding


def get_user_by_photo(file, users):
    pass
    return users[0]
