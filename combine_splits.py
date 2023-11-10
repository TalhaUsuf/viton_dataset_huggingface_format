"""
it will read the individual splits from disk made using other script and then combine them into a single dataset, lastly saving 
it to disk
"""


from datasets import load_from_disk
from datasets import Dataset, DatasetDict


combined_dataset = DatasetDict({
    
    "test_paired" : load_from_disk("viton_test_paired_dataset"),
    "test_unpaired" : load_from_disk("viton_test_unpaired_dataset"),
    "train_paired" : load_from_disk("viton_train_paired_dataset"),
    "train_unpaired" : load_from_disk("viton_train_unpaired_dataset")
    
})


combined_dataset.save_to_disk("viton_combined_dataset")



