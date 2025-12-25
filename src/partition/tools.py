import psutil
import torch , yaml , json , gc
def extract_input_layer(file_name ):
    file = "cfg/" + file_name
    cfg = yaml.safe_load(open(file, 'r', encoding='utf-8'))

    cut_point_path = "res/cut_point.log"
    with open(cut_point_path, 'r') as file:
        content = file.read()
        parsed = json.loads(content)
        cut_layer = parsed['data'][0]

    res_dict = {
        "output" : [cut_layer - 1] ,
        "res_head" : [] ,
        "res_tail" : []
    }

    lst = []

    for index , layer in enumerate(cfg["head"]) :
        if layer[0] != -1 :
            for j in range(len(layer[0])):
                if layer[0][j] != -1 :
                    new_val = [layer[0][j] , index + 11]
                    lst.append(new_val)

    for pair in lst :
        if pair[0] < cut_layer and  pair[1] < cut_layer :
            res_dict["res_head"].append(pair[0])
        elif pair[0] < cut_layer and pair[1] >= cut_layer :
            if pair[0] in res_dict["output"]:
                pass
            else :
                res_dict["output"].append(pair[0])
        else:
            res_dict["res_tail"].append(pair[0])

    return res_dict

#
# print(extract_input_layer("yolo11n.yaml"))

def load_weights_optimized(model, path):
    """Load weights and remove architecture with random weights.

    Args:
        model : model after load by DetectionModel class
        path : path of weights ( only weights )

    Return :
        model : model with weights loaded .

    """
    print(f"[Weights] Loading {path}...")
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except:
        print("[Warning] mmap failed, using standard load")
        ckpt = torch.load(path, map_location='cpu', weights_only=True)

    state_dict = ckpt['model'].state_dict() if 'model' in ckpt else ckpt

    model.load_state_dict(state_dict, strict=False)

    del ckpt, state_dict
    gc.collect()
    print("[Weights] Loaded & RAM cleaned.")



