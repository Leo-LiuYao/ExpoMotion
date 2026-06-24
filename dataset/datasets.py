import os
import glob
import torch
import random
from torch.utils.data import Dataset
import cv2
import numpy as np
from itertools import permutations

def read_8bit(img_path):
    """Read an 8-bit RGB image and return an HWC numpy array in [0, 1]."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

def get_uniform_crop_coords(img_shape, crop_size, crop_num):
    """
    Compute evenly spaced crop coordinates.

    Args:
        img_shape (tuple): Image shape (h, w, c)
        crop_size (tuple): Crop size (crop_h, crop_w)
        crop_num (int): Number of uniform crops

    Returns:
        list: List of (y, x) top-left coordinates
    """
    h, w = img_shape[:2]
    crop_h, crop_w = crop_size
    
    assert crop_h <= h and crop_w <= w, f"Crop size {crop_size} exceeds image size ({h},{w})"
    
    step_h = (h - crop_h) / max(1, crop_num - 1) if crop_num > 1 else 0
    step_w = (w - crop_w) / max(1, crop_num - 1) if crop_num > 1 else 0
    
    crop_coords = []
    for i in range(crop_num):
        y = int(min(i * step_h, h - crop_h))
        x = int(min(i * step_w, w - crop_w))
        crop_coords.append((y, x))
    
    if len(crop_coords) < crop_num:
        crop_coords += random.choices(crop_coords, k=crop_num - len(crop_coords))
    
    return crop_coords

def uniform_crop_imgs(imgs, crop_size, coord):
    """
    Crop multiple images at the same coordinate.

    Args:
        imgs (list): List of HWC numpy arrays
        crop_size (tuple): Crop size (crop_h, crop_w)
        coord (tuple): Top-left coordinate (y, x)

    Returns:
        list: Cropped images
    """
    crop_h, crop_w = crop_size
    y, x = coord
    cropped_imgs = [img[y:y+crop_h, x:x+crop_w, :] for img in imgs]
    return cropped_imgs

def get_random_crop_coord(img_shape, crop_size):
    """
    Sample a random crop coordinate.

    Args:
        img_shape (tuple): Image shape (h, w, c)
        crop_size (tuple): Crop size (crop_h, crop_w)

    Returns:
        tuple: Top-left coordinate (y, x)
    """
    h, w = img_shape[:2]
    crop_h, crop_w = crop_size
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return (y, x)

def random_rotate_imgs(imgs, rotate_range, crop_size):
    """
    Rotate multiple images by the same random angle.

    Args:
        imgs (list): List of HWC numpy arrays
        rotate_range (tuple): Angle range (min_angle, max_angle)
        crop_size (tuple): Target crop size (height, width) after rotation

    Returns:
        list: Rotated images
    """
    angle = random.uniform(rotate_range[0], rotate_range[1])
    h, w = imgs[0].shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated_imgs = []
    for img in imgs:
        rotated = cv2.warpAffine(img, M, (new_w, new_h), borderMode=cv2.BORDER_REFLECT)
        if rotated.shape[0] != crop_size[0] or rotated.shape[1] != crop_size[1]:
            rotated = cv2.resize(rotated, (crop_size[1], crop_size[0]))
        rotated_imgs.append(rotated)
    return rotated_imgs

class Train_Dataset(Dataset):
    def __init__(self, dataset_dir, crop_num_uniform=20, crop_num_random=10, crop_size=(256, 256), rotate_range=(-15, 15)):
        """
        Training dataset with uniform and random crops.

        Supports two directory layouts:
        1. Independent sequences: folder name without '-', or not parseable as LargeID-SmallID.
           Each sequence needs N inputs + 1 GT. Uses Inputs[0], Inputs[mid], Inputs[-1] as img0, img1, img2.
        2. Large-group / sub-group layout: folder name "LargeID-SmallID".
           For each sub-group i as the main group (provides img1 and gt):
             img0, img2 are taken from other groups' first and last inputs.
           All permutations of other groups are enumerated.

        Args:
            dataset_dir (str): Dataset root directory
            crop_num_uniform (int): Number of uniform crops per sample
            crop_num_random (int): Number of random crops per sample
            crop_size (tuple): Crop size (height, width)
            rotate_range (tuple): Rotation angle range (min, max), or None to disable
        """
        self.dataset_dir = dataset_dir
        self.crop_num_uniform = crop_num_uniform
        self.crop_num_random = crop_num_random
        self.crop_num_total = crop_num_uniform + crop_num_random
        self.crop_size = crop_size
        self.rotate_range = rotate_range

        # 1. Scan directories and group by large ID
        subdirs = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        
        large_groups = {}  # Key: LargeID, Value: list of subdir names
        independent_seqs = []
        
        for d in subdirs:
            if '-' in d:
                parts = d.split('-')
                # Naming convention: "LargeID-SmallID" (split on the first '-')
                large_id = parts[0]
                if large_id not in large_groups:
                    large_groups[large_id] = []
                large_groups[large_id].append(d)
            else:
                independent_seqs.append(d)
        
        self.samples = []  # {'img0', 'img1', 'img2', 'gt', 'info'}
        
        # 2. Independent sequences
        for seq in independent_seqs:
            seq_path = os.path.join(dataset_dir, seq)
            files = sorted(glob.glob(os.path.join(seq_path, '*.JPG')) + glob.glob(os.path.join(seq_path, '*.jpg')))
            if len(files) < 2:  # at least 1 input + 1 GT
                continue
            
            inputs = files[:-1]
            gt = files[-1]
            
            img0 = inputs[0]
            mid_idx = len(inputs) // 2
            img1 = inputs[mid_idx]
            img2 = inputs[-1]
            
            self.samples.append({
                'img0': img0,
                'img1': img1,
                'img2': img2,
                'gt': gt,
                'info': f"Independent: {seq}"
            })
            
        # 3. Large-group sequences
        for large_id, sub_group_names in large_groups.items():
            sub_group_data = []
            
            for sub_name in sub_group_names:
                seq_path = os.path.join(dataset_dir, sub_name)
                files = sorted(glob.glob(os.path.join(seq_path, '*.JPG')) + glob.glob(os.path.join(seq_path, '*.jpg')))
                if len(files) < 2:
                    print(f"Warning: sub-group {sub_name} in large group {large_id} has too few images; skipped")
                    continue
                
                inputs = files[:-1]
                gt = files[-1]
                
                mid_idx = len(inputs) // 2
                
                sub_group_data.append({
                    'mid': inputs[mid_idx],
                    'gt': gt,
                    'first': inputs[0],
                    'last': inputs[-1],
                    'name': sub_name
                })
            
            if len(sub_group_data) < 2:
                # Fall back to independent-style sampling when cross-group pairs are unavailable
                for g in sub_group_data:
                    self.samples.append({
                        'img0': g['first'],
                        'img1': g['mid'],
                        'img2': g['last'],
                        'gt': g['gt'],
                        'info': f"Degraded LargeGroup: {g['name']}"
                    })
                continue

            # For each main group i, pair img0/img2 from permutations of other groups
            for i, main_group in enumerate(sub_group_data):
                others = [g for j, g in enumerate(sub_group_data) if j != i]
                
                if len(others) >= 2:
                    # GroupA provides img0 (first), GroupB provides img2 (last)
                    for g_a, g_b in permutations(others, 2):
                        self.samples.append({
                            'img0': g_a['first'],
                            'img1': main_group['mid'],
                            'img2': g_b['last'],
                            'gt': main_group['gt'],
                            'info': f"LargeGroup {large_id}: Main={main_group['name']}, Left={g_a['name']}, Right={g_b['name']}"
                        })
                elif len(others) == 1:
                    # Fallback: both img0 and img2 from the single other group
                    other = others[0]
                    self.samples.append({
                        'img0': other['first'],
                        'img1': main_group['mid'],
                        'img2': other['last'],
                        'gt': main_group['gt'],
                        'info': f"LargeGroup {large_id} (Pair): Main={main_group['name']}, Other={other['name']}"
                    })

        # Precompute uniform crop coordinates (assume all images share the same size)
        self.crop_coords_dict = {}
        if len(self.samples) > 0:
            try:
                ref_img_path = self.samples[0]['img0']
                img = read_8bit(ref_img_path)
                ref_shape = img.shape
                
                for idx in range(len(self.samples)):
                    self.crop_coords_dict[idx] = get_uniform_crop_coords(
                        ref_shape, self.crop_size, self.crop_num_uniform
                    )
            except Exception as e:
                print(f"Failed to initialize crop coordinates: {e}")

    def __len__(self):
        return len(self.samples) * self.crop_num_total

    def __getitem__(self, idx):
        original_idx = idx // self.crop_num_total
        crop_idx = idx % self.crop_num_total

        if original_idx >= len(self.samples):
            raise IndexError("Index out of dataset range")
        
        sample = self.samples[original_idx]
        
        try:
            img0 = read_8bit(sample['img0'])
            img1 = read_8bit(sample['img1'])
            img2 = read_8bit(sample['img2'])
            img_hdr = read_8bit(sample['gt'])
        except Exception as e:
            raise RuntimeError(f"Failed to read images: {e}, Info: {sample.get('info', '')}")
        
        raw_imgs = [img0, img1, img2, img_hdr]
        img_shape = img0.shape
        crop_type = "uniform" if crop_idx < self.crop_num_uniform else "random"

        if crop_idx < self.crop_num_uniform:
            coords = self.crop_coords_dict.get(original_idx)
            if not coords or crop_idx >= len(coords):
                crop_coord = get_random_crop_coord(img_shape, self.crop_size)
            else:
                crop_coord = coords[crop_idx]
        else:
            crop_coord = get_random_crop_coord(img_shape, self.crop_size)
        
        cropped_imgs = uniform_crop_imgs(raw_imgs, self.crop_size, crop_coord)
        if self.rotate_range is not None:
            augmented_imgs = random_rotate_imgs(cropped_imgs, self.rotate_range, self.crop_size)
        else:
            augmented_imgs = cropped_imgs
        
        aug_img0, aug_img1, aug_img2, aug_hdr = augmented_imgs
        
        img0_tensor = torch.from_numpy(aug_img0).permute(2, 0, 1)
        img1_tensor = torch.from_numpy(aug_img1).permute(2, 0, 1)
        img2_tensor = torch.from_numpy(aug_img2).permute(2, 0, 1)
        label = torch.from_numpy(aug_hdr).permute(2, 0, 1)

        inputs = [img0_tensor, img1_tensor, img2_tensor]

        return {
            'inputs': inputs,
            'label': label,
            'crop_type': crop_type,
            'original_idx': original_idx,
            'crop_idx': crop_idx,
            'aug_imgs': [aug_img0, aug_img1, aug_img2, aug_hdr],
            'info': sample.get('info', 'Unknown')
        }


class Test_Dataset(Dataset):
    def __init__(self, dataset_dir):
        """
        Test dataset without cropping or rotation; GT is detected automatically.

        Args:
            dataset_dir (str): Dataset root directory
        """
        self.dataset_dir = dataset_dir
        self.sample_info = []  # (sample paths dict or file list, has_gt flag)

        sequences = sorted(os.listdir(dataset_dir))
        for seq in sequences:
            seq_path = os.path.join(dataset_dir, seq)
            if not os.path.isdir(seq_path):
                continue
            files = sorted(glob.glob(os.path.join(seq_path, '*.JPG')) + glob.glob(os.path.join(seq_path, '*.jpg')))
            
            # 3 images: inputs only; 4+ images: last file is GT
            if len(files) == 3:
                self.sample_info.append((files, False))
            elif len(files) >= 4:
                inputs = files[:-1]
                gt_file = files[-1]
                
                if len(inputs) == 3:
                    chosen_inputs = inputs
                else:
                    chosen_inputs = [inputs[0], inputs[len(inputs)//2], inputs[-1]]
                
                self.sample_info.append(({
                    'inputs': chosen_inputs,
                    'gt': gt_file
                }, True))
            else:
                print(f"Warning: sequence {seq} has {len(files)} images and was skipped")
                continue

        if len(self.sample_info) == 0:
            print("Warning: test dataset is empty")

    def __len__(self):
        return len(self.sample_info)

    def __getitem__(self, idx):
        """Return one test sample."""
        sample_data, has_gt = self.sample_info[idx]
        
        if isinstance(sample_data, tuple) or isinstance(sample_data, list):
            files = sample_data
            input_files = files[:3]
            gt_file = files[3] if has_gt and len(files) > 3 else None
        else:
            input_files = sample_data['inputs']
            gt_file = sample_data['gt']

        try:
            img0 = read_8bit(input_files[0])
            img1 = read_8bit(input_files[1])
            img2 = read_8bit(input_files[2])
        except Exception as e:
            raise RuntimeError(f"Failed to read input images (index {idx}): {e}")

        img0_tensor = torch.from_numpy(img0).permute(2, 0, 1)
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1)
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1)
        inputs = [img0_tensor, img1_tensor, img2_tensor]

        result = {
            'inputs': inputs,
            'has_gt': has_gt,
            'original_idx': idx,
            'img_shape': img0.shape
        }

        if has_gt and gt_file:
            try:
                img_hdr = read_8bit(gt_file)
                label = torch.from_numpy(img_hdr).permute(2, 0, 1)
                result['label'] = label
                result['gt_img'] = img_hdr
            except Exception as e:
                raise RuntimeError(f"Failed to read GT image (index {idx}): {e}")

        result['input_imgs'] = [img0, img1, img2]

        return result
