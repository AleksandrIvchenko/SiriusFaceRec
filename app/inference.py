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
        triton_client = grpcclient.InferenceServerClient(
            url=URL,
            verbose=True,
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "facedetector"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('input__0', [3, 640, 640], "FP32"))

    # Initialize the data
    image_file = load_image(file)
    # image_file = image_file[np.newaxis, ...]
    inputs[0].set_data_from_numpy(image_file)

    outputs.append(grpcclient.InferRequestedOutput('output__0'))

    # Test with outputs
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs,
                                  client_timeout=CLIENT_TIMEOUT,
                                  headers={'test': '1'})

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)

    # Get the output arrays from the results
    output0_data = results.as_numpy('output__0')

    # print(output0_data)
    return output0_data


async def parse_image(file):
    try:
        # file_bytes = await file.read()
        # embedding = len(file_bytes)
        user_id = file.filename
        embedding = inference_client(file.file)
        file.close()
    except Exception as e:
        print(e)
        return 'Error parsing file'

    return user_id, embedding
