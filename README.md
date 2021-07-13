## Intel-OpenVINO-Colab

Using Google Colab to do inference using Intel OpenVINO.

## Tutorial

[Intel OpenVINO on Google Colab](https://medium.com/analytics-vidhya/intel-openvino-on-google-colab-20ac8d2eede6)

## How to use

### Step 1: Importing Libraries. <a href="https://github.com/jojo96/intel-openvino-colab/blob/main/notebooks/AllModels.ipynb" target=_blank>Ref: AllModels.ipynb</a>

```python3
!pip install openvino
```

### Step 2: Setting up environment. <a href="https://github.com/jojo96/intel-openvino-colab/blob/main/notebooks/AllModels.ipynb" target=_blank>Ref: AllModels.ipynb</a>

```python3
from openvino.inference_engine import IENetwork 
from openvino.inference_engine import IECore
import warnings
from google.colab.patches import cv2_imshow
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_IR_to_IE(model_xml):
    ### Load the Inference Engine API
    plugin = IECore()
    ### Loading the IR files to IENetwork class
    model_bin = model_xml[:-3]+"bin" 
    network = IENetwork(model=model_xml, weights=model_bin)
    ### Loading the network
    executable_net = plugin.load_network(network,"CPU")
    print("Network succesfully loaded into the Inference Engine")
    return executable_net
    
def synchronous_inference(executable_net, image):
    ### Get the input blob for the inference request
    input_blob = next(iter(executable_net.inputs))
    ### Perform Synchronous Inference
    result = executable_net.infer(inputs = {input_blob: image})
    return result

```
For use cases refer [notebook](https://github.com/jojo96/intel-openvino-colab/blob/main/notebooks/AllModels.ipynb).

## Notebooks

Demo1: [Inference Demo](https://github.com/jojo96/intel-openvino-colab/blob/main/notebooks/AllModels.ipynb)

Demo2: [IE File Generation and inference](https://github.com/jojo96/intel-openvino-colab/blob/main/notebooks/CompleteDemoOpenVINOColab.ipynb)

## References

The model descriptions have been borrowed from [Intel](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel) and code has been adapted from this [repository](https://github.com/alihussainia/openvino-colab).
