import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
import torch.optim as optim
import torch.nn as nn


from PIL import Image
import torch
from transformers import DetrImageProcessor
from datasets import load_dataset
from transformers import DetrImageProcessor, DetrForObjectDetection, DetrConfig
import torch
from torch import nn
from torch.utils.data import DataLoader



# Initialize the processor
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-101')

# Load the dataset
dataset = load_dataset("detection-datasets/coco", split='train[:1%]')

def transform_data(sample):
    image = sample['image']
    inputs = processor(images=image, return_tensors="pt")

    # Directly use image dimensions for verification
    actual_width, actual_height = image.size
    print("Actual dimensions:", actual_width, actual_height)

    # Convert bounding box format [x_min, y_min, width, height] to [x0, y0, x1, y1] without scaling
    bboxes = []
    for box in sample['objects']['bbox']:
        x0 = box[0]
        y0 = box[1]
        x1 = x0 + box[2]
        y1 = y0 + box[3]
        print(f"Box: ({x0}, {y0}, {x1}, {y1})")  # Debug print
        bboxes.append([x0, y0, x1, y1])
    tensor_boxes = torch.tensor(bboxes, dtype=torch.float32)

    labels = torch.tensor(sample['objects']['category'], dtype=torch.long)

    return inputs, tensor_boxes, labels

# Example usage for demonstration
sample_data = dataset[0]
print(sample_data)
inputs, boxes, labels = transform_data(sample_data)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_image_with_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for box in boxes:
        # Adjust rectangle drawing based on [x0, y0, x1, y1]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

inputs, boxes, labels = transform_data(sample_data)
show_image_with_boxes(sample_data['image'], boxes.numpy())  # Ensure boxes are converted to numpy if needed



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-101').to(device)

class CustomCocoDataset(Dataset):
    def __init__(self, dataset, target_size=(800, 800)):
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']

        # Ensure image is in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image to target size
        image = F.resize(image, self.target_size)

        inputs = processor(images=image, return_tensors="pt")
        
        # Scale bounding boxes according to the new image size
        scale_x = self.target_size[0] / sample['width']
        scale_y = self.target_size[1] / sample['height']
        bboxes = []
        for box in sample['objects']['bbox']:
            x0 = box[0] * scale_x
            y0 = box[1] * scale_y
            x1 = (box[0] + box[2]) * scale_x
            y1 = (box[1] + box[3]) * scale_y
            bboxes.append([x0, y0, x1, y1])
        tensor_boxes = torch.tensor(bboxes, dtype=torch.float32)

        labels = torch.tensor(sample['objects']['category'], dtype=torch.long)
        
        return {'inputs': inputs['pixel_values'].squeeze(0), 'boxes': tensor_boxes, 'labels': labels}

# Adjust collate function as needed:
def collate_fn(batch):
    inputs = torch.stack([item['inputs'] for item in batch])
    max_boxes = max(len(item['boxes']) for item in batch)
    padded_boxes = torch.zeros((len(batch), max_boxes, 4))
    box_masks = torch.zeros((len(batch), max_boxes), dtype=torch.bool)
    padded_labels = torch.full((len(batch), max_boxes), -1)  # Fill labels that are not present with -1

    for i, item in enumerate(batch):
        num_boxes = item['boxes'].shape[0]
        padded_boxes[i, :num_boxes] = item['boxes']
        padded_labels[i, :num_boxes] = item['labels']
        box_masks[i, :num_boxes] = 1

    return {'inputs': inputs, 'boxes': padded_boxes, 'labels': padded_labels, 'box_masks': box_masks}


import torch.nn as nn

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target_labels):
        # Flatten logits and target_labels to compute cross entropy loss
        logits_flat = logits.view(-1, logits.shape[-1])  # Shape: (batch_size * num_queries, num_classes)
        target_labels_flat = target_labels.view(-1)      # Shape: (batch_size * num_queries)
        
        # Compute cross entropy loss
        loss_ce = nn.CrossEntropyLoss()(logits_flat, target_labels_flat)
        return loss_ce

class BboxRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        # Flatten predicted boxes and target boxes
        pred_boxes_flat = pred_boxes.view(-1, 4)  # Shape: (batch_size * num_queries, 4)
        target_boxes_flat = target_boxes.view(-1, 4)  # Shape: (batch_size * num_queries, 4)
        
        # Compute L1 loss (smooth L1 loss) for bbox regression
        loss_bbox = nn.SmoothL1Loss(reduction='sum')(pred_boxes_flat, target_boxes_flat)  # Can also use nn.L1Loss()
        return loss_bbox


coco_dataset = CustomCocoDataset(dataset)

# Now use this custom collate function in your DataLoader
data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


# Initialize the loss functions
classification_loss_fn = ClassificationLoss()
bbox_regression_loss_fn = BboxRegressionLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define the training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(data_loader):
        inputs = batch['inputs'].to(device)
        boxes = batch['boxes'].to(device)
        labels = batch['labels'].to(device)
        box_masks = batch['box_masks'].to(device)
        
        print(f"inputs size: {inputs.shape}")
        print(f"boxes size: {boxes.shape}")
        print(f"labels size: {labels.shape}")
        print(f"box_masks size: {box_masks.shape}")

        # Forward pass
        outputs = model(pixel_values=inputs)
        print(f"Output logits size: {outputs['logits'].shape}")
        # Prepare targets for all non-padded areas
        targets = []
        for i in range(inputs.size(0)):

            active_indices = box_masks[i]

            active_boxes = boxes[i][active_indices]

            active_labels = labels[i][active_indices]

            targets.append({'labels': active_labels, 'boxes': active_boxes})
        
        # Compute the classification loss
        logits = outputs['logits']
        max_target_labels = max(len(target['labels']) for target in targets)
        num_queries = logits.size(1)
        padded_logits = torch.zeros((logits.size(0), num_queries, logits.size(2)), device=device)
        for i, target in enumerate(targets):
            padded_logits[i, :len(target['labels'])] = logits[i, :len(target['labels'])]
        logits = padded_logits
        print(f"Padded logits size: {logits.shape}")

        target_labels = torch.cat([target['labels'] for target in targets], dim=0)

        # Resize logits to match the number of target labels
        num_target_labels = target_labels.size(0)
        print(f"num target labels: {num_target_labels}")
        logits_flat = logits.view(-1, logits.shape[-1])  # Shape: (batch_size * num_queries, num_classes)
        print(f"logits_flat: {logits_flat.shape}")
        
        target_labels_flat = target_labels.view(-1)      # Shape: (batch_size * num_queries)
        print(f"target_labels_flat: {target_labels_flat.shape}")

        # Slice the logits tensor to match the size of the target_labels tensor
        logits_flat = logits_flat[:target_labels_flat.size(0)]
        print(f"logits_flat: {logits_flat.shape}")

        classification_loss = classification_loss_fn(logits_flat, target_labels_flat)

        
        # Compute the bbox regression loss
        pred_boxes = outputs['pred_boxes']
        target_boxes = torch.cat([target['boxes'] for target in targets])

        # Compute the bbox regression loss
        pred_boxes_flat = pred_boxes.view(-1, 4)  # Shape: (batch_size * num_queries, 4)
        target_boxes_flat = target_boxes.view(-1, 4)  # Shape: (batch_size * num_queries, 4)

        # Slice the pred_boxes_flat tensor to match the size of the target_boxes_flat tensor
        pred_boxes_flat = pred_boxes_flat[:target_boxes_flat.size(0)]

        # Compute L1 loss (smooth L1 loss) for bbox regression
        loss_bbox = nn.SmoothL1Loss(reduction='sum')(pred_boxes_flat, target_boxes_flat)

        #bbox_regression_loss = bbox_regression_loss_fn(pred_boxes, target_boxes)
        
        # Compute total loss
        loss = classification_loss + loss_bbox
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}")

    print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss / len(data_loader)}")

print("Training complete!")

