import sys

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

from const import URL, CLIENT_TIMEOUT
from utils import extractor_preprocessing
from crop_and_landmarks import pipeline


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

def detector_postprocessing(output0_data, output1_data, output2_data, raw_image):
    device = 'cuda'
    net = torch.jit.load("./weights/FaceDetector.pt", map_location=torch.device(device))
    net.eval()
    """
    image_path = "./curve/scar.jpeg"
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    """
    loc, conf, landms = output0_data, output1_data, output2_data #net(img)  # forward pass

    #image, landmarks = pipeline(img_raw, loc, conf, landms)
    image, landmarks = pipeline(raw_image, loc, conf, landms)

    return image, landmarks



def extractor(image):
    triton_client = get_triton_client()
    model_name = "featureextractor"

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
