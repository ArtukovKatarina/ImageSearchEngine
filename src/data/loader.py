import torch
import cv2
import albumentations as A
from src import paths

class CLIPLoader(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms, max_length):
        """
        image_filenames and captions must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.max_length = max_length
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=self.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{paths.RAW_PATH}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


def get_transforms(image_size, mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(image_size, image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
