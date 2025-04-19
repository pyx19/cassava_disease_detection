import torch
import numpy as np
import cv2

class YOLOGradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        # For YOLOv8/9, the last conv in the backbone is a good default
        self.target_layer = target_layer or self._find_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_target_layer(self):
        # Try to find the last Conv layer in the backbone
        for m in reversed(list(self.model.model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
        raise ValueError("No Conv2d layer found in YOLO model")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, box_xyxy):
        # input_tensor: (1, 3, H, W), box_xyxy: [x1, y1, x2, y2] in image coords
        self.model.zero_grad()
        output = self.model(input_tensor)[0]
        # For each detection, find the one with max confidence in the box
        boxes = output.boxes.xyxy.cpu().numpy()
        confs = output.boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            return None
        # Find the detection that matches the box
        ious = [self._iou(box_xyxy, b) for b in boxes]
        idx = np.argmax(ious)
        # Backprop on the confidence of this detection
        score = output.boxes.conf[idx]
        score.backward(retain_graph=True)
        # Compute Grad-CAM
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        cam = np.sum(weights[:, None, None] * activations, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

    def _iou(self, boxA, boxB):
        # box: [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou
