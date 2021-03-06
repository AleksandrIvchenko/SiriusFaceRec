import json
import sys
from io import BytesIO
from typing import List, Optional, IO

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

from const import URL, CLIENT_TIMEOUT
from models import User
from utils import load_image, detector_postprocessing


def get_triton_client() -> grpcclient.InferenceServerClient:
    try:
        triton_client = grpcclient.InferenceServerClient(url=URL, verbose=False)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def detector(image_array):
    triton_client = get_triton_client()
    model_name = "facedetector"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 640, 640], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(image_array)

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
    return output0_data, output1_data, output2_data


def extractor(image):
    triton_client = get_triton_client()
    model_name = "extractor"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input__0', [1, 3, 128, 128], "FP32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(image)

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


def get_embedding(file: Optional[IO]) -> np.ndarray:
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])

    image_array = load_image(file)
    o1, o2, o3 = detector(image_array)

    image, landmarks = detector_postprocessing(o1, o2, o3, Image.open(file))
    image_array = np.float32(image)
    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = image_array[np.newaxis, ...]
    embedding = extractor(image_array.astype(np.float32))
    return np.squeeze(embedding)


def get_user_by_photo(file: Optional[IO], users: List[User]) -> str:
    new_embedding = get_embedding(file)
    embeddings = []
    names = []
    for user in users:
        embeddings.append(np.array(json.loads(user.emb), dtype=np.float32))
        names.append(user.name)
    embeddings = np.array(embeddings)
    dists = np.linalg.norm(new_embedding - embeddings, axis=1)
    min_index = np.argmin(dists)
    print(names)
    print(dists)
    if dists[min_index] > 1.1:
        return ''
    return users[min_index].name


def get_embedding2(file: Optional[IO]):
    mean = np.array([0.485*255, 0.456*255, 0.406*255])
    std = np.array([0.229*255, 0.224*255, 0.225*255])

    image_array = load_image(file)
    o1, o2, o3 = detector(image_array)

    image, landmarks = detector_postprocessing(o1, o2, o3, Image.open(file))
    image_array = np.float32(image)
    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = image_array[np.newaxis, ...]
    embedding = extractor(image_array.astype(np.float32))
    # print(embedding)
    image = Image.fromarray(image, 'RGB')
    temp = BytesIO()
    image.save(temp, format="png")
    temp.seek(0)
    return temp
