import numpy as np
import matplotlib.pyplot as plt
import torch

def list_to_colormap(x_list):
    """
    Convert class labels to RGB colors using a fixed color palette.
    """
    y = np.zeros((x_list.shape[0], 3))
    # Fixed color palette for consistent visualization across experiments
    palette = [
        (0, 0, 0),           # 0: Background
        (147, 67, 46),       # 1
        (0, 0, 255),         # 2
        (255, 100, 0),       # 3
        (0, 255, 123),       # 4
        (164, 75, 155),      # 5
        (101, 174, 255),     # 6
        (118, 254, 172),     # 7
        (60, 91, 112),       # 8
        (255, 255, 0),       # 9
        (255, 255, 125),     # 10
        (255, 0, 255),       # 11
        (100, 0, 255),       # 12
        (0, 172, 254),       # 13
        (0, 255, 0),         # 14
        (171, 175, 80),      # 15
        (101, 193, 60),      # 16
    ]
    
    for index, item in enumerate(x_list):
        item = int(item)
        if item < len(palette):
            color = palette[item]
        else:
            # Fallback for datasets with more classes
            color = (item * 20 % 255, item * 50 % 255, item * 80 % 255)
        y[index] = np.array(color) / 255.0
    
    return y


def get_cls_map(model, data_loader, gt_labels, save_path, dataset_name):
    """
    Generates and saves a classification map from model predictions.
    """
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.cuda()
            output, _ = model(batch_data)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
    
    # Create the classification map
    height, width = gt_labels.shape
    cls_map = np.zeros((height, width))
    
    labeled_pixels = np.where(gt_labels > 0)
    for i, pred in enumerate(all_preds):
        row, col = labeled_pixels[0][i], labeled_pixels[1][i]
        cls_map[row, col] = pred + 1
    
    # Convert class indices to RGB colors
    flat_pred_map = cls_map.flatten()
    flat_gt_map = gt_labels.flatten()
    
    pred_colors = list_to_colormap(flat_pred_map)
    gt_colors = list_to_colormap(flat_gt_map)
    
    pred_img = pred_colors.reshape(height, width, 3)
    gt_img = gt_colors.reshape(height, width, 3)
    
    # Save classification map
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width * 2.0 / 300, height * 2.0 / 300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(pred_img)
    
    map_filename = f"{dataset_name}_predictions.png"
    fig.savefig(save_path + map_filename, dpi=300)
    plt.close()
    
    # Save ground truth map
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width * 2.0 / 300, height * 2.0 / 300)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(gt_img)
    
    gt_filename = f"{dataset_name}_gt.png"
    fig.savefig(save_path + gt_filename, dpi=300)
    plt.close()
    
    print(f"Classification and ground truth maps saved to {save_path}")
