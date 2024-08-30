# TODO:
# 1. make sure that images and their corresponding videos are being installed in sync
# 
# 
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Load the MATLAB file
mat = scipy.io.loadmat('dataset/data/mpii_human_pose_v1_u12_1.mat')
release = mat['RELEASE']

annolist = release['annolist'][0][0][0]
img_train = release['img_train'][0, 0][0]
single_person = release['single_person'][0][0]
act = release['act'][0, 0]
video_list = release['video_list'][0, 0]

print(len(annolist))


for idx in range(len(annolist)):
    for actor_id in single_person[idx]:
        image_name = annolist[idx]['image'][0]['name'][0][0]
        # single_ids = single_person[idx][0].astype(int)  # Convert IDs to integers

        # Extract `annorect` data
        try:
            annorect_data = annolist[idx]['annorect'][actor_id]
        except (IndexError, KeyError):
            annorect_data = annolist[idx]['annorect'][0]


        # TODO: CHECK WITH EMPTY ANNORECT!!!!!!!!!
        if annorect_data is None:
            continue  # Skip if the annotation is None
        
        # List to hold annotations
        annotations = []

        # Check if 'annopoints' exists and is valid
        if annorect_data.dtype.names and 'annopoints' in annorect_data.dtype.names:
            points = annorect_data['annopoints'][0, 0]['point']
            joint_coords = []
            for p in points:
                x = p['x'][0][0][0]
                y = p['y'][0][0][0]
                joint_id = p['id'][0][0][0]
                is_visible = p['is_visible'][0][0][0] if p['is_visible'].size > 0 else 1
                joint_coords.append((x, y, joint_id, is_visible))
            person_id = annorect_data['id'][0][0][0]
            annotations.append({
                'image_name': image_name,
                'person_id': person_id,
                'joints': joint_coords
            })

# Example output
for anno in annotations:
    print(anno)

class PoseEstimationDataset(Dataset):
    def __init__(self, annotations, image_dir, transform=None):
        self.annotations = annotations
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation['image_name'])
        image = Image.open(image_path).convert('RGB')
        joints = annotation['joints']

        if self.transform:
            image = self.transform(image)

        joints_tensor = torch.tensor(joints, dtype=torch.float32)

        return image, joints_tensor

# Usage
dataset = PoseEstimationDataset(annotations, image_dir='/path/to/images', transform=your_transform)

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for images, targets in dataloader:
    # Your training code here
    pass
