"""
this script serves to convert the viton dataset to a huggingface dataset format

"""

from pathlib import Path
import os
from tqdm import tqdm
from datasets import load_dataset, Dataset
from datasets.features import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Data:
    def __init__(self, root_dataset_path):
        BASE = Path(root_dataset_path)
        self.viton = BASE / "vton-plus-complete"
        self.dresscode = BASE / "DressCode"
        
        
    def build_paths(self, type_data=None, paired = False, phase='train'):
        if type_data == "viton":
            # following will be structure to parse
            # ├── test
            # │   ├── agnostic-v3.2
            # │   ├── cloth
            # │   ├── cloth-mask
            # │   ├── image
            # │   ├── image-densepose
            # │   ├── image-parse-agnostic-v3.2
            # │   ├── image-parse-v3
            # │   ├── openpose_img
            # │   └── openpose_json
            # ├── test_pairs.txt
            # ├── train
            # │   ├── agnostic-v3.2
            # │   ├── cloth
            # │   ├── cloth-mask
            # │   ├── image
            # │   ├── image-densepose
            # │   ├── image-parse-agnostic-v3.2
            # │   ├── image-parse-v3
            # │   ├── openpose_img
            # │   └── openpose_json
            # └── train_pairs.txt

            self.viton_train = self.viton / "train"
            self.viton_test = self.viton / "test"
            if phase == 'train':
                lines = (self.viton / "train_pairs.txt").read_text().split("\n")
            else:
                lines = (self.viton / "test_pairs.txt").read_text().split("\n")
            pairs = [sub.split() for sub in lines if len(sub.split()) == 2]
            if paired == False:
                # unpaired case
                imname = [ Path(k[0]).stem for k in pairs]
                cname = [ Path(k[1]).stem for k in pairs]
                print(f"{len(imname)} images found in train, unpaired case")
            
                if phase == "train":
                    self.set_train_images(imname=imname, cname=cname)
                else:
                    self.set_test_images(imname=imname, cname=cname)
                
            else:
                # paired case
                imname = [ Path(k[0]).stem for k in pairs]
                cname = imname
                print(f"{len(imname)} images found in train, paired case")    
                
                if phase == "train":
                    self.set_train_images(imname=imname, cname=cname)
                else:
                    self.set_test_images(imname=imname, cname=cname)
            
        return self
            
    def set_train_images(self, imname, cname):
        '''
        given a list of imname and cname, set the paths for train images using all the preprocessing maps dirs.

        Parameters
        ----------
        imname : List[str]
            names of images except the extension.
        cname : List[str]
            names of clothes except the extension.
        '''        
        # parse the dirs.
        #agnostic-v3.2  cloth  cloth-mask  cloth-warp  cloth-warp-mask  image  image-densepose  image-parse-agnostic-v3.2  image-parse-v3  openpose_img  openpose_json
        self.image = [self.viton_train / "image" / f"{k}.jpg" for k in imname]
        self.agnostic_v32 = [self.viton_train / "agnostic-v3.2" / f"{k}.jpg" for k in imname]
        
        self.cloth = [self.viton_train / "cloth" / f"{k}.jpg" for k in cname]
        self.cloth_mask = [self.viton_train / "cloth-mask" / f"{k}.jpg" for k in cname]
        self.cloth_warp = [self.viton_train / "cloth-warp" / f"{k}.jpg" for k in cname]
        self.cloth_warp_mask = [self.viton_train / "cloth-warp-mask" / f"{k}.jpg" for k in cname]
        
        self.image_densepose = [self.viton_train / "image-densepose" / f"{k}.jpg" for k in imname]
        self.image_parse_agnostic_v32 = [self.viton_train / "image-parse-agnostic-v3.2" / f"{k}.png" for k in imname]
        self.image_parse_v3 = [self.viton_train / "image-parse-v3" / f"{k}.png" for k in imname]
        self.openpose_image = [self.viton_train / "openpose_img" / f"{k}_rendered.png" for k in imname]
        self.openpose_json = [self.viton_train / "openpose_json" / f"{k}_keypoints.json" for k in imname]
    
    
    def set_test_images(self, imname, cname):
        '''
        given a list of imname and cname, set the paths for test images using all the preprocessing maps dirs.

        Parameters
        ----------
        imname : List[str]
            names of images except the extension.
        cname : List[str]
            names of clothes except the extension.
        '''        
        # parse the dirs.
        #agnostic-v3.2  cloth  cloth-mask  cloth-warp  cloth-warp-mask  image  image-densepose  image-parse-agnostic-v3.2  image-parse-v3  openpose_img  openpose_json
        self.image = [self.viton_test / "image" / f"{k}.jpg" for k in imname]
        self.agnostic_v32 = [self.viton_test / "agnostic-v3.2" / f"{k}.jpg" for k in imname]
        
        self.cloth = [self.viton_test / "cloth" / f"{k}.jpg" for k in cname]
        self.cloth_mask = [self.viton_test / "cloth-mask" / f"{k}.jpg" for k in cname]
        self.cloth_warp = [self.viton_test / "cloth-warp" / f"{k}.jpg" for k in cname]
        self.cloth_warp_mask = [self.viton_test / "cloth-warp-mask" / f"{k}.jpg" for k in cname]
        
        self.image_densepose = [self.viton_test / "image-densepose" / f"{k}.jpg" for k in imname]
        self.image_parse_agnostic_v32 = [self.viton_test / "image-parse-agnostic-v3.2" / f"{k}.png" for k in imname]
        self.image_parse_v3 = [self.viton_test / "image-parse-v3" / f"{k}.png" for k in imname]
        self.openpose_image = [self.viton_test / "openpose_img" / f"{k}_rendered.png" for k in imname]
        self.openpose_json = [self.viton_test / "openpose_json" / f"{k}_keypoints.json" for k in imname]    
    
    
        
    def convert_to_df(self):
        # convert to df
        df = pd.DataFrame(
                                {
                                    "image": self.image,
                                "agnostic_v32": self.agnostic_v32,
                                "cloth": self.cloth,
                                "cloth_mask": self.cloth_mask,
                                "cloth_warp": self.cloth_warp,
                                "cloth_warp_mask": self.cloth_warp_mask,
                                "image_densepose": self.image_densepose,
                                "image_parse_agnostic_v32": self.image_parse_agnostic_v32,
                                "image_parse_v3": self.image_parse_v3,
                                "openpose_image": self.openpose_image,
                                "openpose_json": self.openpose_json
                                
                                }
                        )
        for k in df.columns:
            # assert that paths exist
            print(f"checking {k}")
            
            # inplace change to str
            df[k] = df[k].apply(lambda x: str(x))
            assert df[k].apply(lambda x: os.path.exists(x)).all()
        self.df = df
        return self
    
    def convert_to_hf_dataset(self):
        
        df = self.df
        
        import json
        def read_json(examples):
            data = []
            for k in examples['openpose_json']:
                with open(k, 'r') as f:
                    data.append(json.load(f))
            return {'openpose_json': data}
            
        
        # df col. list
        # remove openpose_json column from df columns
        cols = df.columns.tolist()
        cols.remove('openpose_json')
        
        ds = Dataset.from_pandas(df) 
        for k in cols:
            # cast image cols. to hf datasets Image() type inplace
            ds = ds.cast_column(k, Image())
            
        ds = ds.map(read_json, batched=True, num_proc=4, remove_columns=['cloth_warp', 'cloth_warp_mask'])
        

        return ds
    
    @staticmethod
    def plot_image_features(dataset, index):
        """
        Plots the image-based features of a dataset row specified by index.
        
        :param dataset: HuggingFace dataset with image-based features.
        :param index: Index of the row in the dataset to visualize.
        """
        # Define the image-based features to visualize
        image_features = ['image', 'cloth', 'cloth_mask', 'cloth_warp', 'cloth_warp_mask',
                        'image_densepose', 'image_parse_agnostic_v32', 'image_parse_v3', 'openpose_image']
        
        # Number of images to plot
        num_images = len(image_features)
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
        
        # Loop through each image feature and plot
        for i, feature in enumerate(image_features):
            # Check if the feature exists in the dataset
            if feature in dataset.features:
                # Get the image data and convert to PIL image
                image_data = dataset[index][feature]
                image = np.array(image_data)
                
                # Plot the image
                axes[i].imshow(image)
                axes[i].set_title(feature)
                axes[i].axis('off')
            else:
                axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig("sample.png")
    
if __name__ == '__main__':

    split = "test"
    paired = False
    name = f"viton_{split}_unpaired_dataset" if paired == False else f"viton_{split}_paired_dataset"
    import io
    data = Data("dataset/")
    ds_train = data.build_paths(type_data="viton", paired=paired, phase=split).convert_to_df().convert_to_hf_dataset()
    
    ds_train.save_to_disk(name, num_proc=12)
    
    # Data.plot_image_features(ds_train, 0)

    
    
    
    
