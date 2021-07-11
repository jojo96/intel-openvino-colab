from openvino.inference_engine import IENetwork 
from openvino.inference_engine import IECore

def load_IR_to_IE(model_xml):
    plugin = IECore()
    model_bin = model_xml[:-3]+"bin" 
    network = IENetwork(model=model_xml, weights=model_bin)
    executable_net = plugin.load_network(network,"CPU")
    print("Network succesfully loaded into the Inference Engine")
    return executable_net
	
def synchronous_inference(executable_net, image):
    input_blob = next(iter(executable_net.inputs))
    result = executable_net.infer(inputs = {input_blob: image})
    return result
	
en = load_IR_to_IE('age.xml')
import cv2
image = cv2.imread('age2.jpg')
face_img = cv2.dnn.blobFromImage(image, 1./127.5, (128, 128), (1, 1, 1), True)
resized = cv2.resize(image, (62,62), interpolation = cv2.INTER_AREA)

from torchvision import transforms
tran = transforms.ToTensor()  # Convert the numpy array or PIL.Image read image to (C, H, W) Tensor format and /255 normalize to [0, 1.0]
img_tensor = tran(resized)
img_tensor = img_tensor.unsqueeze_(0)

res = synchronous_inference(en, img_tensor)
print(round(res['age_conv3'][0][0][0][0]*100))