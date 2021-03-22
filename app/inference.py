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


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(file):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    size = 640, 640
    img = Image.open(file)
    img.load()
    img = img.convert('RGB')
    img = expand2square(img, (0, 0, 0))
    img.thumbnail(size, Image.ANTIALIAS)
    image_array = np.float32(img)
    image_array = (image_array - mean) / (std + sys.float_info.epsilon)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = image_array[np.newaxis, ...]
    image_array = image_array.astype(np.float32)
    # resized_img = img.resize((640, 640), Image.BILINEAR)
    # resized_img = np.array(resized_img)
    # resized_img = resized_img.astype(np.float32)
    # resized_img = np.transpose(resized_img, (2, 0, 1))
    # resized_img = resized_img[np.newaxis, ...]
    return image_array


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

    print(output0_data.shape)
    print(output1_data.shape)
    print(output2_data.shape)
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


def get_embedding(file):
    image_array = load_image(file.file)
    o1, o2, o3 = detector(image_array)
    embedding = extractor(o1)
    return embedding


def get_user_by_photo(file, users):
    pass
    return users[0]
