"""Data loader definition"""

import mimetypes
import random

from fastai.data.transforms import get_files, DataLoaders
from torch.utils.data import Dataset

from src.pipeline.transforms import PoiPipeline

audio_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('audio/'))

def get_audio_files(path, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=audio_extensions, recurse=recurse, folders=folders)


class PoiMesher():
    """Meshes person of intrest files with non-poi files Data Loader"""
    def __init__(self, poi: int):
        self.poi = poi

    def get_poi_files(self, path: str):
        """Returns list of person of interest files"""
        return get_audio_files(path, folders=f"{self.poi}")

    def get_non_poi_files(self, path):
        """Returns list of non person of interest files"""
        files = get_audio_files(path)
        poi_path = path/str(self.poi)
        return [file for file in files if poi_path not in list(file.parents)]

    def __call__(self, path):
        """Returns list of tuples with poi and non-poi files"""
        rand_poi = []
        poi_files = self.get_poi_files(path)
        non_poi_files = self.get_non_poi_files(path)
        for _ in range(len(non_poi_files)):
            rand_poi.append(random.choice(poi_files))
        return list(map(tuple,zip(rand_poi, non_poi_files)))

class PoiDataset(Dataset):
    """Person of Interest Dataset object"""
    def __init__(self, path: str, poi: int, pipe):
        mesher = PoiMesher(poi)
        self.data_pairs = mesher(path)
        self.len = len(self.data_pairs)
        self.pipe = pipe

    def __len__(self):
        return self.len

    def __getitem__(self, i: int):
        files = self.data_pairs[i]
        return self.pipe(files)

def create_poi_dataloader(path: str, poi: int, bs=2):
    """Returns dataloader for path and person of interest"""
    train_ds = PoiDataset(path, poi, PoiPipeline)
    poi_data_loader = DataLoaders.from_dsets(train_ds, train_ds, bs=bs).cuda()
    return poi_data_loader
