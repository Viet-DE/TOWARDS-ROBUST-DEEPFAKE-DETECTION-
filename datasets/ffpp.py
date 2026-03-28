import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import json
import os
from .transforms import TriXNetTransforms

class FFPPDataset(Dataset):
    def __init__(self, 
                 json_path: str,
                 root_dir: str,
                 freq_dir: str = 'data/frequency',
                 flow_dir: str = 'data/flow',
                 parts_dir: str = 'data/parts',
                 transform=None,
                 image_size: int = 224):
        
        self.root_dir = Path(root_dir)
        self.freq_dir = Path(freq_dir)
        self.flow_dir = Path(flow_dir)
        self.parts_dir = Path(parts_dir)
        
        # Load danh sách dữ liệu từ JSON (Flat List)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing split file: {json_path}")
            
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
            
        self.transform = transform
        self.image_size = image_size
        
        # Thống kê
        real_count = sum(1 for s in self.samples if s['label'] == 0)
        fake_count = len(self.samples) - real_count
        print(f"Loaded {json_path}: {len(self.samples)} samples ({real_count} Real, {fake_count} Fake)")

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        return Image.open(path).convert('RGB')

    def _get_feature_path(self, frame_rel_path, feature_dir, ext):
        # frame_rel_path có dạng: data/frames/class_name/filename.jpg
        # Cần lấy: class_name/filename.ext
        
        # Chuẩn hóa path object
        path_obj = Path(frame_rel_path)
        
        # Lấy tên file (bỏ đuôi .jpg) -> video_01_f0
        stem = path_obj.stem 
        
        # Lấy tên class (deepfakes, original...) -> thư mục cha
        parent_name = path_obj.parent.name 
        
        # Tạo đường dẫn feature
        return feature_dir / parent_name / f"{stem}{ext}"

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Load RGB Frame (Input cho visual backbone nếu cần)
        # Đường dẫn trong JSON có thể là tương đối "data/frames/..."
        # Cần nối với Project Root nếu cần, hoặc dùng trực tiếp nếu chạy từ root
        rgb_path = item['path'] # data/frames/...
        
        # Xử lý đường dẫn tuyệt đối/tương đối
        if os.path.exists(rgb_path):
            full_rgb_path = rgb_path
        else:
            # Fallback: Thử nối với root_dir nếu path trong json chỉ là tên file
            full_rgb_path = self.root_dir / Path(rgb_path).name
            
        try:
            image = self._load_rgb(full_rgb_path)
        except Exception as e:
            # Fallback nếu lỗi ảnh: Trả về tensor đen (tránh crash training)
            print(f"Error loading {full_rgb_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size))

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.zeros(3, self.image_size, self.image_size)

        # 2. Load Frequency (FRS Branch)
        freq_path = self._get_feature_path(rgb_path, self.freq_dir, ".npy")
        if freq_path.exists():
            freq_data = np.load(freq_path)
            # Freq thường là 1 channel [H, W] -> Thêm channel dim [1, H, W]
            if len(freq_data.shape) == 2:
                freq_tensor = torch.from_numpy(freq_data).unsqueeze(0).float()
            else:
                freq_tensor = torch.from_numpy(freq_data).permute(2, 0, 1).float()
            
            freq_tensor = torch.nn.functional.interpolate(freq_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)
        else:
            freq_tensor = torch.zeros(1, self.image_size, self.image_size)

        # 3. Load Optical Flow (DOF Branch)
        flow_path = self._get_feature_path(rgb_path, self.flow_dir, ".npy")
        if flow_path.exists():
            flow_data = np.load(flow_path) # [H, W, 2]
            flow_tensor = torch.from_numpy(flow_data).permute(2, 0, 1).float() # [2, H, W]
            flow_tensor = torch.nn.functional.interpolate(flow_tensor.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear').squeeze(0)
        else:
            flow_tensor = torch.zeros(2, self.image_size, self.image_size)

        # 4. Load Parts (LPC Branch)
        parts_path = self._get_feature_path(rgb_path, self.parts_dir, ".npz")
        if parts_path.exists():
            try:
                parts_data = np.load(parts_path)
                # Eyes: [64, 128, 3] -> [3, 64, 128]
                eyes = torch.from_numpy(parts_data['eyes']).permute(2, 0, 1).float() / 255.0
                # Mouth: [64, 64, 3] -> [3, 64, 64]
                mouth = torch.from_numpy(parts_data['mouth']).permute(2, 0, 1).float() / 255.0
            except:
                 eyes = torch.zeros(3, 64, 128)
                 mouth = torch.zeros(3, 64, 64)
        else:
            eyes = torch.zeros(3, 64, 128)
            mouth = torch.zeros(3, 64, 64)

        return {
            'image': image_tensor, 
            'frequency': freq_tensor,
            'flow': flow_tensor,
            'parts': {'eyes': eyes, 'mouth': mouth},
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def create_dataloaders(config):
    """
    Tạo 3 dataloaders đọc từ config
    """
    ds_cfg = config['dataset']
    split_dir = Path(ds_cfg['split_dir']) # Folder 'splits'
    
    # Transforms
    train_transform = TriXNetTransforms(is_train=True, image_size=ds_cfg['image_size'])
    val_transform = TriXNetTransforms(is_train=False, image_size=ds_cfg['image_size'])
    
    # 1. Train Loader
    train_ds = FFPPDataset(
        json_path=split_dir / ds_cfg['train_split'], # splits/train.json
        root_dir=ds_cfg['root_dir'],
        freq_dir=ds_cfg['frequency_dir'],
        flow_dir=ds_cfg['flow_dir'],
        parts_dir=ds_cfg['parts_dir'],
        transform=train_transform,
        image_size=ds_cfg['image_size']
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=ds_cfg['batch_size'],
        shuffle=True,
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory']
    )
    
    # 2. Val Loader
    val_ds = FFPPDataset(
        json_path=split_dir / ds_cfg['val_split'],
        root_dir=ds_cfg['root_dir'],
        freq_dir=ds_cfg['frequency_dir'],
        flow_dir=ds_cfg['flow_dir'],
        parts_dir=ds_cfg['parts_dir'],
        transform=val_transform,
        image_size=ds_cfg['image_size']
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=ds_cfg['batch_size'],
        shuffle=False,
        num_workers=ds_cfg['num_workers'],
        pin_memory=ds_cfg['pin_memory']
    )
    
    # 3. Test Loader (Optional)
    test_loader = None
    if 'test_split' in ds_cfg:
        test_ds = FFPPDataset(
            json_path=split_dir / ds_cfg['test_split'],
            root_dir=ds_cfg['root_dir'],
            freq_dir=ds_cfg['frequency_dir'],
            flow_dir=ds_cfg['flow_dir'],
            parts_dir=ds_cfg['parts_dir'],
            transform=val_transform,
            image_size=ds_cfg['image_size']
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=ds_cfg['batch_size'],
            shuffle=False,
            num_workers=ds_cfg['num_workers']
        )
        
    return train_loader, val_loader, test_loader