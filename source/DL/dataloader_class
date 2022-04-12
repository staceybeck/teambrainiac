class DatasetFmri(Dataset):
    def __init__(self, image_dictionary, transform=None, target_transform=None):
        self.img_labels = image_dictionary['labels']
        self.images = torch.from_numpy(image_dictionary['images'])
        # Maybe set up transfers later

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.label[idx]
        return image, label
