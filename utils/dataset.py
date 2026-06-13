from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset, RandomSampler
import numpy as np
import torch
import lmdb
import json
from PIL import Image
import os
import torchvision.transforms.functional as TF
import pandas as pd
import cv2
import random
from pathlib import Path
import decord
from torchvision.transforms.functional import resize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

SAMPLE_N_FRAMES_BUCKET_INTERVAL = 4
SPATIAL_COMPRESSION_RATIO = 16
ASPECT_RATIO_512 = {
    "1:1": [512, 512],
    "4:3": [512, 672],
    "3:4": [672, 512],
    "16:9": [512, 912],
    "9:16": [912, 512],
    "21:9": [512, 1192],
    "9:21": [1192, 512],
}
ASPECT_RATIO_RANDOM_CROP_512 = ASPECT_RATIO_512
ASPECT_RATIO_RANDOM_CROP_PROB = np.ones(len(ASPECT_RATIO_RANDOM_CROP_512)) / len(ASPECT_RATIO_RANDOM_CROP_512)


class OffsetDistributedSampler(DistributedSampler):
    def __init__(self, dataset, initial_step=0, gpu_num=4, **kwargs):
        super().__init__(dataset, **kwargs)
        if initial_step < len(dataset) // gpu_num:
            self.initial_step = initial_step
        else:
            self.initial_step = (
                (initial_step * gpu_num - len(dataset)) % len(dataset)
            ) // (gpu_num * gpu_num)
        self.first_time = True  # 标志位，表示是否是第一次加载

    def __iter__(self):
        # 获取原始索引
        indices = list(super().__iter__())

        # 如果是第一次加载，跳过前 initial_step 个索引
        if self.first_time and self.initial_step > 0:
            indices = indices[self.initial_step :]
            self.first_time = False  # 标志位设为 False，后续不再跳过

        return iter(indices)


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TextFolderDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        for file in os.listdir(data_path):
            if file.endswith(".txt"):
                with open(os.path.join(data_path, file), "r") as f:
                    text = f.read().strip()
                    self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"prompts": self.texts[idx], "idx": idx}


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        img = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "img", np.uint8, local_idx,
            shape=(480, 832, 3)
        )
        img = Image.fromarray(img)
        img = TF.to_tensor(img).sub_(0.5).div_(0.5)

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "img": img
        }


class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }


class ImageVideoControlDataset(Dataset):
    """Wan2.2Fun-style image/video dataset with optional control frames.

    The metadata file can be CSV or JSON/JSONL and should contain at least a
    text/caption column plus a video/image path column. Supported aliases:
    ``text``/``caption``/``prompt`` for prompts, ``path``/``video_path``/``file``
    for target pixels, and ``control_path``/``control_video_path`` for control
    pixels.  If no control path is provided, target pixels are reused as control
    pixels, matching the common first-frame/control-ref bootstrap workflow.
    """

    def __init__(
        self,
        train_data_meta,
        train_data_dir=None,
        video_sample_size=704,
        video_sample_stride=1,
        video_sample_n_frames=121,
        video_repeat=1,
        image_sample_size=None,
        enable_bucket=True,
        enable_camera_info=False,
    ):
        self.meta_path = Path(train_data_meta)
        self.train_data_dir = Path(train_data_dir) if train_data_dir else self.meta_path.parent
        self.video_sample_size = video_sample_size
        self.video_sample_stride = max(int(video_sample_stride), 1)
        self.video_sample_n_frames = int(video_sample_n_frames)
        self.video_repeat = max(int(video_repeat), 1)
        self.image_sample_size = image_sample_size or video_sample_size
        self.enable_bucket = enable_bucket
        self.enable_camera_info = enable_camera_info
        self.dataset = self
        self.items = self._load_metadata(self.meta_path) * self.video_repeat

    def _load_metadata(self, meta_path):
        suffix = meta_path.suffix.lower()
        if suffix == ".csv":
            rows = pd.read_csv(meta_path).fillna("").to_dict("records")
        elif suffix == ".jsonl":
            with open(meta_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            rows = data if isinstance(data, list) else data.get("data", [])
        if not rows:
            raise ValueError(f"No training samples found in {meta_path}")
        return rows

    def __len__(self):
        return len(self.items)

    def _get_value(self, item, names, default=""):
        for name in names:
            if name in item and item[name] != "":
                return item[name]
        return default

    def _resolve_path(self, path):
        path = Path(str(path))
        if path.is_absolute() or path.exists():
            return path
        candidate = self.train_data_dir / path
        if candidate.exists():
            return candidate
        return self.meta_path.parent / path

    def _read_video_or_image(self, path):
        path = self._resolve_path(path)
        if path.is_dir():
            image_files = sorted(
                [p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            )
            if not image_files:
                raise ValueError(f"No image frames found in {path}")
            frames = [cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in image_files]
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            frame = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            frames = [frame]
        else:
            reader = decord.VideoReader(uri=path.as_posix())
            indices = list(range(0, len(reader), self.video_sample_stride))
            if not indices:
                raise ValueError(f"Video has no frames: {path}")
            indices = indices[: self.video_sample_n_frames]
            frames = reader.get_batch(indices).asnumpy()
            return self._pad_frames(frames)
        return self._pad_frames(np.stack(frames, axis=0))

    def _pad_frames(self, frames):
        if frames.shape[0] >= self.video_sample_n_frames:
            return frames[: self.video_sample_n_frames]
        pad = np.repeat(frames[-1:], self.video_sample_n_frames - frames.shape[0], axis=0)
        return np.concatenate([frames, pad], axis=0)

    def __getitem__(self, index):
        item = self.items[index]
        text = self._get_value(item, ["text", "caption", "prompt"])
        pixel_path = self._get_value(item, ["path", "video_path", "image_path", "file", "file_path"])
        if pixel_path == "":
            raise ValueError(f"Sample {index} is missing a video/image path: {item}")
        control_path = self._get_value(
            item,
            ["control_path", "control_file_path", "control_video_path", "control_image_path"],
            pixel_path,
        )
        example = {
            "pixel_values": self._read_video_or_image(pixel_path),
            "control_pixel_values": self._read_video_or_image(control_path),
            "text": text,
            "data_type": "image" if str(pixel_path).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")) else "video",
        }
        if self.enable_camera_info:
            example["control_camera_values"] = None
        return example


class ImageVideoSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, dataset, batch_size, drop_last=True):
        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)
        self.dataset = dataset


class AspectRatioBatchImageVideoSampler(ImageVideoSampler):
    def __init__(self, sampler, dataset, batch_size, train_folder=None, drop_last=True, aspect_ratios=None):
        super().__init__(sampler, dataset=dataset, batch_size=batch_size, drop_last=drop_last)
        self.train_folder = train_folder
        self.aspect_ratios = aspect_ratios


def wan22fun_worker_init_fn(seed):
    seed = seed * 256

    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return _worker_init_fn


def _align_to_spatial_compression(size, spatial_compression_ratio=SPATIAL_COMPRESSION_RATIO):
    return [max(spatial_compression_ratio * 2, int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2) for x in size]


def _get_closest_ratio(height, width, ratios):
    src_ratio = height / max(width, 1)
    best_key = min(ratios, key=lambda key: abs((ratios[key][0] / max(ratios[key][1], 1)) - src_ratio))
    return ratios[best_key], best_key


def _center_crop_video(video, size):
    th, tw = int(size[0]), int(size[1])
    _, _, h, w = video.shape
    top = max((h - th) // 2, 0)
    left = max((w - tw) // 2, 0)
    return video[:, :, top: top + th, left: left + tw]


def _resize_crop_normalize_video(video, sample_size, normalize=True):
    if isinstance(sample_size, int):
        sample_size = (sample_size, sample_size)
    sample_size = tuple(int(x) for x in sample_size)
    if video.dtype != torch.float32:
        video = video.float()
    if video.max() > 2:
        video = video / 255.0
    h, w = video.shape[-2:]
    th, tw = sample_size
    if th / tw > h / max(w, 1):
        resize_size = (th, int(w * th / max(h, 1)))
    else:
        resize_size = (int(h * tw / max(w, 1)), tw)
    video = F.interpolate(video, size=resize_size, mode="bilinear", align_corners=False)
    video = _center_crop_video(video, sample_size)
    if normalize:
        video = video.sub(0.5).div(0.5)
    return video


def _get_length_to_frame_num(config, token_length):
    image_sample_size = getattr(config, "image_sample_size", getattr(config, "video_sample_size", 704))
    video_sample_size = getattr(config, "video_sample_size", image_sample_size)
    interval = getattr(config, "sample_n_frames_bucket_interval", SAMPLE_N_FRAMES_BUCKET_INTERVAL)
    if image_sample_size > video_sample_size:
        sample_sizes = list(range(video_sample_size, image_sample_size + 1, 128))
        if sample_sizes[-1] != image_sample_size:
            sample_sizes.append(image_sample_size)
    else:
        sample_sizes = [image_sample_size]
    video_sample_n_frames = getattr(config, "video_sample_n_frames", getattr(config, "num_frames", 121))
    return {
        sample_size: int(min(token_length / sample_size / sample_size, video_sample_n_frames) // interval * interval + 1)
        for sample_size in sample_sizes
    }


def _get_random_downsample_ratio(sample_size, image_ratio=None, all_choices=False, rng=None):
    image_ratio = image_ratio or []
    if sample_size >= 1536:
        number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio
    elif sample_size >= 1024:
        number_list = [1, 1.25, 1.5, 2] + image_ratio
    elif sample_size >= 768:
        number_list = [1, 1.25, 1.5] + image_ratio
    elif sample_size >= 512:
        number_list = [1] + image_ratio
    else:
        number_list = [1]
    if all_choices:
        return number_list
    if len(number_list) == 1:
        probs = np.array([1.0])
    else:
        probs = np.array([0.90] + [(0.10 / (len(number_list) - 1))] * (len(number_list) - 1))
    return (rng or np.random).choice(number_list, p=probs)


def _get_random_downsample_probability(choice_list, token_sample_size):
    if len(choice_list) == 1:
        return [1.0]
    closest_index = min(range(len(choice_list)), key=lambda i: abs(choice_list[i] - token_sample_size))
    probs = [0.50 / (len(choice_list) - 1)] * len(choice_list)
    probs[closest_index] = 0.50
    return probs


def wan22fun_collate_fn(config):
    def _collate(examples):
        video_sample_n_frames = int(getattr(config, "video_sample_n_frames", getattr(config, "num_frames", 121)))
        video_sample_size = int(getattr(config, "video_sample_size", getattr(config, "h", 704)))
        image_sample_size = int(getattr(config, "image_sample_size", video_sample_size))
        token_sample_size = int(getattr(config, "token_sample_size", video_sample_size))
        interval = int(getattr(config, "sample_n_frames_bucket_interval", SAMPLE_N_FRAMES_BUCKET_INTERVAL))
        spatial_compression_ratio = int(getattr(config, "spatial_compression_ratio", SPATIAL_COMPRESSION_RATIO))
        random_hw_adapt = bool(getattr(config, "random_hw_adapt", True))
        training_with_video_token_length = bool(getattr(config, "training_with_video_token_length", True))
        random_ratio_crop = bool(getattr(config, "random_ratio_crop", False))
        fix_sample_size = getattr(config, "fix_sample_size", None)
        train_mode = getattr(config, "train_mode", "control_ref")
        control_ref_image = getattr(config, "control_ref_image", "first_frame")
        add_inpaint_info = bool(getattr(config, "add_inpaint_info", False))

        target_token_length = video_sample_n_frames * token_sample_size * token_sample_size
        length_to_frame_num = _get_length_to_frame_num(config, target_token_length)

        first_pixel_value = examples[0]["pixel_values"]
        _, height, width, _ = np.shape(first_pixel_value)
        if random_hw_adapt:
            if training_with_video_token_length:
                local_min_size = np.min(
                    np.array([
                        np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]]))
                        for example in examples
                    ])
                )
                choice_list = [size for size in length_to_frame_num if size < local_min_size * 1.25]
                if len(choice_list) == 0:
                    choice_list = list(length_to_frame_num.keys())
                probabilities = _get_random_downsample_probability(choice_list, token_sample_size)
                local_video_sample_size = np.random.choice(choice_list, p=probabilities)
                random_downsample_ratio = video_sample_size / local_video_sample_size
                batch_video_length = length_to_frame_num[local_video_sample_size]
            else:
                random_downsample_ratio = _get_random_downsample_ratio(video_sample_size)
                batch_video_length = video_sample_n_frames + interval
        else:
            random_downsample_ratio = 1
            batch_video_length = video_sample_n_frames + interval

        aspect_ratio_sample_size = {
            key: [x / 512 * video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]]
            for key in ASPECT_RATIO_512.keys()
        }
        aspect_ratio_random_crop_sample_size = {
            key: [x / 512 * video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]]
            for key in ASPECT_RATIO_RANDOM_CROP_512.keys()
        }

        if fix_sample_size is not None:
            sample_size = _align_to_spatial_compression(fix_sample_size, spatial_compression_ratio)
        elif random_ratio_crop:
            random_sample_size = aspect_ratio_random_crop_sample_size[
                np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p=ASPECT_RATIO_RANDOM_CROP_PROB)
            ]
            sample_size = _align_to_spatial_compression(random_sample_size, spatial_compression_ratio)
        else:
            closest_size, _ = _get_closest_ratio(height, width, ratios=aspect_ratio_sample_size)
            sample_size = _align_to_spatial_compression(closest_size, spatial_compression_ratio)

        min_example_length = min(example["pixel_values"].shape[0] for example in examples)
        batch_video_length = int(min(batch_video_length, min_example_length))
        batch_video_length = (batch_video_length - 1) // interval * interval + 1
        batch_video_length = max(batch_video_length, 1)

        batch = {
            "target_token_length": target_token_length,
            "pixel_values": [],
            "control_pixel_values": [],
            "text": [],
            "prompts": [],
        }
        if train_mode != "control":
            batch["ref_pixel_values"] = []
            batch["clip_pixel_values"] = []
            batch["clip_idx"] = []
        if train_mode == "control_camera_ref":
            batch["control_camera_values"] = []
        if add_inpaint_info:
            batch["mask_pixel_values"] = []
            batch["mask"] = []

        for example in examples:
            pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
            control_pixel_values = torch.from_numpy(example["control_pixel_values"]).permute(0, 3, 1, 2).contiguous()
            pixel_values = _resize_crop_normalize_video(pixel_values, sample_size, normalize=True)
            control_pixel_values = _resize_crop_normalize_video(control_pixel_values, sample_size, normalize=True)
            pixel_values = pixel_values[:batch_video_length]
            control_pixel_values = control_pixel_values[:batch_video_length]
            batch["pixel_values"].append(pixel_values)
            batch["control_pixel_values"].append(control_pixel_values)
            batch["text"].append(example["text"])
            batch["prompts"].append(example["text"])

            if train_mode == "control_camera_ref":
                control_camera_values = example.get("control_camera_values", None)
                if control_camera_values is None:
                    camera_values = torch.zeros(
                        (
                            batch_video_length,
                            6,
                            control_pixel_values.shape[-2],
                            control_pixel_values.shape[-1],
                        ),
                        dtype=control_pixel_values.dtype,
                    )
                else:
                    camera_values = torch.as_tensor(control_camera_values).permute(0, 3, 1, 2).contiguous().float()
                    camera_values = _resize_crop_normalize_video(camera_values, sample_size, normalize=False)[:batch_video_length]
                batch["control_camera_values"].append(camera_values)

            if train_mode != "control":
                if control_ref_image == "first_frame" or len(pixel_values) == 1:
                    clip_idx = 0
                else:
                    probs = np.array([0.40] + [(0.60 / (len(pixel_values) - 1))] * (len(pixel_values) - 1))
                    clip_idx = int(np.random.choice(list(range(len(pixel_values))), p=probs))
                ref_pixel_values = pixel_values[clip_idx: clip_idx + 1]
                clip_pixel_values = ((pixel_values[clip_idx].permute(1, 2, 0).contiguous() * 0.5 + 0.5) * 255.0)
                batch["ref_pixel_values"].append(ref_pixel_values)
                batch["clip_pixel_values"].append(clip_pixel_values)
                batch["clip_idx"].append(clip_idx)

            if add_inpaint_info:
                mask = torch.ones_like(pixel_values[:, :1])
                mask_pixel_values = pixel_values * (1 - mask)
                batch["mask_pixel_values"].append(mask_pixel_values)
                batch["mask"].append(mask)

        batch["pixel_values"] = torch.stack(batch["pixel_values"])
        batch["control_pixel_values"] = torch.stack(batch["control_pixel_values"])
        if train_mode != "control":
            batch["ref_pixel_values"] = torch.stack(batch["ref_pixel_values"])
            batch["clip_pixel_values"] = torch.stack(batch["clip_pixel_values"])
            batch["clip_idx"] = torch.tensor(batch["clip_idx"], dtype=torch.long)
        if train_mode == "control_camera_ref":
            batch["control_camera_values"] = torch.stack(batch["control_camera_values"])
        if add_inpaint_info:
            batch["mask_pixel_values"] = torch.stack(batch["mask_pixel_values"])
            batch["mask"] = torch.stack(batch["mask"])
        return batch

    return _collate


class ODERegressionCSVDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), num_frames=81, h=480, w=832):
        self.max_pair = max_pair
        self.data_path = Path(data_path)
        self.data = pd.read_csv(data_path)
        if "text" not in self.data.columns:
            raise ValueError(
                f"Dataset CSV {data_path} must contain a 'text' column. "
                f"Available columns: {list(self.data.columns)}"
            )
        if "path" not in self.data.columns and "video_path" in self.data.columns:
            self.data["path"] = self.data["video_path"]
        if "path" not in self.data.columns:
            raise ValueError(
                f"Dataset CSV {data_path} must contain a 'path' or 'video_path' column. "
                f"Available columns: {list(self.data.columns)}"
            )
        self.data["text"] = self.data["text"].fillna("")
        self.log_file = "log/datasets_error_log.txt"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.num_frames = num_frames
        self.h = h
        self.w = w

    def __len__(self):
        return len(self.data)

    def _preprocess_video(self, sample) -> torch.Tensor:
        path = str(sample["path"])
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(self.data_path.parent, path)
        num_frames = int(sample["num_frames"])
        if num_frames < self.num_frames:
            raise ValueError(f"Error: num_frames < {self.num_frames}")
        frame_indices = list(range(self.num_frames))

        if path.endswith(".mp4") or path.endswith(".mkv"):
            path = Path(path)
            video_reader = decord.VideoReader(uri=path.as_posix())
            frames = torch.tensor(
                video_reader.get_batch(frame_indices).asnumpy()
            ).float()  # [T, H, W, C]
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]
        else:
            image_files = sorted(os.listdir(path))
            if not os.path.isdir(path) or not image_files:
                raise ValueError("Error: Invalid images path or no images found")
            frames = []
            for frame_index in frame_indices:
                frame_path = os.path.join(path, image_files[frame_index])
                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frames = np.stack(frames)
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

        video_tensor = torch.stack([resize(frame, (self.h, self.w)) for frame in frames], dim=0)
        video_tensor = video_tensor.permute(1, 0, 2, 3) / 255.0
        video_tensor = video_tensor * 2 - 1
        return video_tensor

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        try:
            video = self._preprocess_video(sample)
            return {
                "prompts": sample["text"],
                "video": video,
            }
        except Exception as e:
            # 记录错误日志
            with open(self.log_file, "a") as f:
                f.write(f"Error at index {index}: {str(e)}\n")
            print(f"Error at index {index}: {e}. Skipping this index.")
            # 跳过当前样本，返回 None 或抛出异常
            return {
                "prompts": "",
                "video": torch.zeros((3, self.num_frames, self.h, self.w)),  # 占位符视频张量
            }

def cycle(dl):
    while True:
        for data in dl:
            yield data

def masks_like(tensor, zero=False, generator=None, p=0.2):
    # assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[0, :] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[0, :]).exp()
                    v[0, :] = torch.zeros_like(v[0, :])
                else:
                    u[0, :] = u[0, :]
                    v[0, :] = v[0, :]
        else:
            for u, v in zip(out1, out2):
                u[0, :] = torch.zeros_like(u[0, :])
                v[0, :] = torch.zeros_like(v[0, :])

    return out1, out2
