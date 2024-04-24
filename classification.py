import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection
from datasets import load_dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
from torch.nn import AdaptiveAvgPool2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the DETR model
detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
backbone = detr_model.model.backbone.conv_encoder.model
backbone.to(device)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize images to the input size expected by DETR
    transforms.ToTensor(),          # Convert images to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

class ImageNetDataset(Dataset):
    def __init__(self, split): 
        self.dataset = load_dataset('zh-plus/tiny-imagenet', split=split) 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        if isinstance(image, str) and os.path.isfile(image):
            image = Image.open(image).convert('RGB')  # Convert image to RGB
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')  # Ensure it's in RGB
        else:
            raise TypeError("Unsupported image format.")

        # Apply the transformations
        image = transform(image).to(device)
        label = torch.tensor(sample['label']).to(device)
        return image, label

train_dataset = ImageNetDataset('train')
train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)

optimizer = optim.Adam(backbone.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
'''
[torch.Size([10, 256, 200, 200]), 
torch.Size([10, 512, 100, 100]), 
torch.Size([10, 1024, 50, 50]), 
torch.Size([10, 2048, 25, 25])]
'''

import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_features, output_features, num_heads=8):
        super(Classifier, self).__init__()
        self.num_heads = num_heads

        # Define the convolutional paths
        self.conv_path1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(33, stride=33)  # Adjust to correct pool size if needed
        )
        self.conv_path2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(16, stride=16)
        )
        self.conv_path3 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(8, stride=8)
        )
        self.conv_path4 = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(4, stride=4)
        )

        # Ensure the flatten dimension is correct
        self.reshape_transform = nn.Linear(512 * 6 * 6, 512)

        # Attention and output layers
        self.multi_head_attention = MultiHeadAttention(512, num_heads=self.num_heads)
        self.fc5 = nn.Linear(2048, output_features)

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        if isinstance(x, list):
            attention_outputs = []
            conv_paths = [self.conv_path1, self.conv_path2, self.conv_path3, self.conv_path4]
            for tensor, conv_path in zip(x, conv_paths):
                tensor = conv_path(tensor)
                tensor = tensor.view(tensor.size(0), -1)
                tensor = self.reshape_transform(tensor)
                attention_output = self.multi_head_attention(tensor)
                attention_outputs.append(attention_output)
            x = torch.cat(attention_outputs, dim=-1)
            #print(f"classifier output: {x.shape}")
        return self.fc5(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "Feature dim must be divisible by num_heads"

        self.query_proj = nn.Linear(feature_dim, self.head_dim * num_heads)
        self.key_proj = nn.Linear(feature_dim, self.head_dim * num_heads)
        self.value_proj = nn.Linear(feature_dim, self.head_dim * num_heads)

        self.output_proj = nn.Linear(self.head_dim * num_heads, feature_dim)  # Ensure dimensions match
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, F = x.shape  # F should be the flat feature dimension
        Q = self.query_proj(x).view(B, self.num_heads, self.head_dim)
        K = self.key_proj(x).view(B, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(B, self.num_heads, self.head_dim)

        # Attention calculations
        attention = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5))
        attention = self.dropout(attention)
        output = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, -1)  # Flatten the output

        # Linear transformation to expected feature dimension
        output = self.output_proj(output)
        #print(f"output of Multi head Attention: {output.shape}")

        return output


# Define classifier and adaptive pooling
classifier = Classifier(256, 200)  # Adjust the input features to match your backbone's output
classifier.to(device)
adaptive_pool = AdaptiveAvgPool2d((1, 1))
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  


backbone.train()
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = backbone(images)  # Get list of feature maps

        # Move each feature map to the GPU
        outputs = [tensor.to(device) for tensor in outputs] 
        #print("Output from backbone")
        #print([tensor.shape for tensor in outputs] )

        # No more selection of a specific map; we process them all
        if isinstance(outputs, list):
            outputs = classifier(outputs)  # Pass the list directly to the classifier

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        scheduler.step(total_loss / len(train_loader))

        _, predicted = torch.max(outputs, 1)
        #print(f"Predicted: {predicted}")
        #print(f"Actual: {labels}")

    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}")

torch.save(backbone.state_dict(), 'fine_tuned_backbone.pth')
torch.save(classifier.state_dict(), 'fine_tuned_imgnet_classifier.pth')

