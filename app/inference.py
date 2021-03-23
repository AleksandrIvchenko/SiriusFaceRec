import sys

import tritonclient.grpc as grpcclient

from const import URL, CLIENT_TIMEOUT
from utils import extractor_preprocessing, load_image, detector_postprocessing


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
    image, landmarks = detector_postprocessing(o1, o2, o3, image_array)
    image = extractor_preprocessing(
        img=image,
        ldm=landmarks,
        resize=128,
    )
    embedding = extractor(image)

    return embedding


def get_user_by_photo(file, users):
    pass
    return users[0]
