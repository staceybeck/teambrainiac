# This notebook has functions and classes that I may use later

# May return to custom dataloader if needed

# from torch.utils.data import Dataset, DataLoader

# class DatasetFmri(Dataset):
#     def __init__(self, image_dictionary, transform=None, target_transform=None):
#         self.labels = image_dictionary['labels']
#         self.images = torch.from_numpy(image_dictionary['images'])
#         # Maybe set up transfers later

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         return image, label





# train_ds = DatasetFmri(image_dictionary = train_dict)
# test_ds = DatasetFmri(image_dictionary = test_dict)

