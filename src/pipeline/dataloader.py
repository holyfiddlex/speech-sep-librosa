"""Data loader definition"""

#from glob import glob
import mimetypes
import random
from fastai.data.transforms import get_files

# audio_ds = glob("../../data/*.wav")
audio_extensions = set(k for k,v in mimetypes.types_map.items() if v.startswith('audio/'))

def get_audio_files(path, recurse=True, folders=None):
    "Get image files in `path` recursively, only in `folders`, if specified."
    return get_files(path, extensions=audio_extensions, recurse=recurse, folders=folders)

def AudioPipe(path, sr=22050, duration=5):
    """Audio processing pipeline"""
    return Pipeline([AudioMono.create, Resample(sr), Clip(duration), Mixer, Spectify(), Unet_Trimmer(8), Normalize(), Decibelify(), Group()])


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
        return list(map(list,zip(rand_poi, non_poi_files)))

class PoiDataset(Dataset):
    """Person of Interest Dataset object"""
    def __init__(self, path: str, poi: int, pipe):
        mesher = PoiMesher(int)
        self.data_pairs = mesher(path)
        self.pipe = pipe

    def __getitem__(self, i: int):
        x,y = self.pipe(index)
        x,y = Tensorify()(x),Tensorify()(y)
        return x,y

train_ds = SpecMaskDataset(files)
dls = DataLoaders.from_dsets(train_ds, train_ds, bs=2).cuda()
