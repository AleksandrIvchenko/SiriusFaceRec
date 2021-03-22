from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time
import io


parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/FaceDetector.pt',
                    type=str, help='Trained state_dict file path to open')

parser.add_argument('--image_path', default='./curve/scar.jpeg',
                    type=str, help='Input image to process')

#parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
#parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

#Cutting face function
#@torch.jit.script
def cut_face(img, ldm, resize=256):
    dst = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    dst = dst * resize / 112
    M, inliers = cv2.estimateAffinePartial2D(ldm, dst, method=cv2.RANSAC, ransacReprojThreshold=5)
    face = cv2.warpAffine(img, M, (resize, resize))
    return face


#@torch.jit.script

@torch.jit.script
def pipeline ():
    confidence_threshold = 0.02
    #top_k = 5000
    nms_threshold = 0.4
    keep_top_k = 750
    save_image = True
    vis_thres = 0.6

    torch.set_grad_enabled(False)
    cfg = cfg_re50


    #device = 'cuda' if torch.cuda.is_available() else 'gpu'
    device = 'cuda'
    #path = args.trained_model
    #path = "./weights/FaceDetector.pt"
    #f = open(path, 'rb')

    # Load all tensors onto GPU, using a device
    net = torch.jit.load("FaceDetector.pt", map_location=torch.device(device))
    net.eval()

    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    resize = 1


    # Input image processing
    #image_path = args.image_path
    image_path = "./curve/scar.jpeg"
    #for i in range(100):

    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:args.top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    #end of for i in range (100)

    # show image
    if save_image:
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            #(image, start_point, end_point, color, thickness)
            #cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            #cx = b[0]
            #cy = b[1] + 12
            #cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # landms
            #cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            #cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            #cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            #cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            #cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

    # save image

    #name = "test.jpg"
    name_rect = "rect.jpg"
    #cv2.imwrite(name, img_raw)

    croped_image = img_raw[b[1]: b[1] + b[3] - b[1] , b[0]: b[0] + b[2] - b[0]]
    cv2.imwrite(name_rect, croped_image)

    landmarks = np.array([
                        [b[5], b[6]],
                        [b[7], b[8]],
                        [b[9], b[10]],
                        [b[11], b[12]],
                        [b[13], b[14]]
                    ], dtype=np.float32)

    face = cut_face(img = img_raw, ldm = landmarks, resize=256)
    name_rect_affin = 'rect_affin.jpg'
    cv2.imwrite(name_rect_affin, face)

    #cv2.imwrite(name_rect, img_raw[b[0]:b[1], b[2]:b[3]]) #Bounding_Box

    # Save landmarks to txt
    file_name = image_path.split('/')[-1]
    landmarks_str = '{file_name}    {}  {}  {}  {}  {}  {}  {}  {}  {}  {}'.format(
        b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], file_name=file_name)

    text_file = open("landmarks.txt", "w")
    text_file.write(landmarks_str)
    text_file.close()

    print(landmarks_str)

if __name__ == '__main__':
    pipeline()



    #traced = torch.jit.trace(pipeline(), image_path_test)

    #traced = torch.jit.script (pipeline())

    #output_pt = 'Image_and_landmarks_generator.pt'

    # Save to file
    #torch.jit.save(m, 'scriptmodule.pt')

    #inputs = torch.randn(1, 3, args.long_side, args.long_side).to(device)

    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    #traced_script_module = torch.jit.trace(net, inputs)

    # Save the TorchScript model
    #traced_script_module.save(output_pt)

    #print("Exported to PT")

