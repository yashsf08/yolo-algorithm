import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        # Use mean squared error with sum reduction for all loss components
        self.mse_loss = nn.MSELoss(reduction="sum")

    def xywh2xyxy(self, boxes):
        """
        Converts bounding boxes from [x_center, y_center, width, height] format to [x1, y1, x2, y2] format.

        Parameters:
        boxes: (tensor) Bounding boxes in [x_center, y_center, w, h] format, sized (N, 4).

        Returns:
        (tensor) Bounding boxes in [x1, y1, x2, y2] format, sized (N, 4).
        """
        x, y, w, h = boxes.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return torch.stack((x1, y1, x2, y2), dim=1)

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Finds the predictor with the highest IoU for each ground truth box.

        Parameters:
        pred_box_list: (list) A list of B tensors, each of size (-1, 4), representing predicted boxes in [x,y,w,h] format.
        box_target: (tensor) Ground truth boxes, size (-1, 4) in [x,y,w,h] format.

        Returns:
        best_iou: (tensor) The highest IoU for each object, size (-1, 1).
        best_idx: (tensor) The index of the best predictor for each object, size (-1).
        """
        target_xyxy = self.xywh2xyxy(box_target)
        ious = []
        for pred_box in pred_box_list:
            pred_xyxy = self.xywh2xyxy(pred_box)
            # Calculate IoU for each prediction against its corresponding target
            iou = torch.diag(compute_iou(pred_xyxy, target_xyxy))
            ious.append(iou)
        
        ious = torch.stack(ious, dim=1)  # Shape: (num_obj, B)
        best_iou, best_idx = ious.max(dim=1)
        
        return best_iou.unsqueeze(1), best_idx

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Calculates the classification loss for grid cells that contain an object.

        Parameters:
        classes_pred: (tensor) Predicted classes, size (N, S, S, 20).
        classes_target: (tensor) Ground truth classes, size (N, S, S, 20).
        has_object_map: (tensor) Mask indicating which cells contain an object, size (N, S, S).

        Returns:
        (scalar) The classification loss.
        """
        # Only compute loss for cells that contain an object
        return self.mse_loss(classes_pred[has_object_map], classes_target[has_object_map])

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Calculates confidence loss for all predictors in cells that do NOT contain an object.

        Parameters:
        pred_boxes_list: (list) A list of B tensors, each size (N, S, S, 5).
        has_object_map: (tensor) Mask indicating object presence, size (N, S, S).

        Returns:
        (scalar) The no-object confidence loss.
        """
        no_object_map = ~has_object_map
        loss = 0.0
        
        for pred_boxes in pred_boxes_list:
            pred_conf = pred_boxes[..., 4][no_object_map]
            target_conf = torch.zeros_like(pred_conf)
            loss += self.mse_loss(pred_conf, target_conf)
            
        return self.l_noobj * loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Calculates confidence loss for the responsible predictor in cells that contain an object.

        Parameters:
        box_pred_conf: (tensor) Confidence of the responsible predictor, size (-1, 1).
        box_target_conf: (tensor) Target confidence (IoU), size (-1, 1).

        Returns:
        (scalar) The object-present confidence loss.
        """
        return self.mse_loss(box_pred_conf, box_target_conf.detach())

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Calculates bounding box regression loss (x, y, w, h).

        Parameters:
        box_pred_response: (tensor) Predicted box parameters [x,y,w,h], size (-1, 4).
        box_target_response: (tensor) Target box parameters [x,y,w,h], size (-1, 4).

        Returns:
        (scalar) The regression loss.
        """
        # Loss for x, y coordinates
        loss_xy = self.mse_loss(box_pred_response[:, :2], box_target_response[:, :2])
        
        # Loss for width, height (using sqrt to emphasize small boxes)
        # Clamp is used to avoid gradient issues with values <= 0
        loss_wh = self.mse_loss(
            torch.sqrt(box_pred_response[:, 2:4].clamp(min=1e-6)),
            torch.sqrt(box_target_response[:, 2:4].clamp(min=1e-6))
        )
        
        return self.l_coord * (loss_xy + loss_wh)

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        Computes the total YOLO loss from the model's prediction and ground truth targets.
        """
        N = pred_tensor.size(0)
        
        # Split prediction tensor into its components
        pred_boxes1 = pred_tensor[..., :5]
        pred_boxes2 = pred_tensor[..., 5:10]
        pred_boxes_list = [pred_boxes1, pred_boxes2]
        pred_cls = pred_tensor[..., 10:]

        # 1. Compute classification loss (for object-containing cells)
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        # 2. Compute no-object confidence loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # Handle object-related losses only if objects exist in the batch
        if not has_object_map.any():
            total_loss = cls_loss + no_obj_loss
            return {
                "total_loss": total_loss,
                "reg_loss": torch.tensor(0.0, device=pred_tensor.device),
                "containing_obj_loss": torch.tensor(0.0, device=pred_tensor.device),
                "no_obj_loss": no_obj_loss,
                "cls_loss": cls_loss,
            }

        # 3. Prepare tensors for object-related losses
        obj_mask = has_object_map.bool()
        box_pred_list_obj = [p[obj_mask] for p in pred_boxes_list]
        box_target_obj = target_boxes[obj_mask]

        # Convert box coordinates from cell-relative to image-relative for IoU
        obj_indices = obj_mask.nonzero(as_tuple=False)
        grid_ij = obj_indices[:, 1:].to(pred_tensor.device, dtype=torch.float32)

        def to_img_relative(boxes, grid_cells):
            center_xy = (grid_cells.flip(1) + boxes[:, :2]) / self.S
            wh = boxes[:, 2:4]
            return torch.cat([center_xy, wh], dim=1)

        pred_xywh_list = [to_img_relative(p, grid_ij) for p in box_pred_list_obj]
        target_xywh = to_img_relative(box_target_obj, grid_ij)
        
        # 4. Find the best predictor and its IoU
        best_iou, best_idx = self.find_best_iou_boxes([p[:,:4] for p in pred_xywh_list], target_xywh)
        
        # 5. Select the responsible predictor's outputs
        box_pred_response = torch.zeros_like(box_target_obj)
        box_pred_conf = torch.zeros_like(best_iou)
        mask_b1, mask_b2 = (best_idx == 0), (best_idx == 1)
        
        box_pred_response[mask_b1] = box_pred_list_obj[0][mask_b1, :4]
        box_pred_response[mask_b2] = box_pred_list_obj[1][mask_b2, :4]
        
        box_pred_conf[mask_b1] = box_pred_list_obj[0][mask_b1, 4:5]
        box_pred_conf[mask_b2] = box_pred_list_obj[1][mask_b2, 4:5]

        # 6. Compute regression and object-confidence loss
        reg_loss = self.get_regression_loss(box_pred_response, box_target_obj)
        containing_obj_loss = self.get_contain_conf_loss(box_pred_conf, best_iou)

        # 7. Compute total loss
        total_loss = reg_loss + containing_obj_loss + no_obj_loss + cls_loss

        loss_dict = {
            "total_loss": total_loss,
            "reg_loss": reg_loss,
            "containing_obj_loss": containing_obj_loss,
            "no_obj_loss": no_obj_loss,
            "cls_loss": cls_loss,
        }
        return loss_dict
