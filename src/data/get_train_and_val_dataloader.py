import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset
import os


def create_folds(some_list, train_split=0.8, val_split=0.1, fold_choice=0):
    some_list_len = len(some_list)
    if fold_choice == 0:
        output_train_list = some_list[: int(some_list_len * train_split)]
        output_val_list = some_list[int(some_list_len * train_split): int(some_list_len * (train_split + val_split))]
    elif fold_choice == 1:
        output_train_list = some_list[int(val_split * some_list_len): int(some_list_len * (train_split + val_split))]
        output_val_list = some_list[: int(val_split * some_list_len)]
    elif fold_choice == 2:
        output_train_list = (
                some_list[: int(val_split * some_list_len)]
                + some_list[2 * int(val_split * some_list_len): int(some_list_len * (train_split + val_split))]
        )
        output_val_list = some_list[int(val_split * some_list_len): 2 * int(some_list_len * val_split)]
    elif fold_choice == 3:
        output_train_list = (
                some_list[: 2 * int(val_split * some_list_len)]
                + some_list[3 * int(val_split * some_list_len): int(some_list_len * (train_split + val_split))]
        )
        output_val_list = some_list[2 * int(val_split * some_list_len): 3 * int(some_list_len * val_split)]
    elif fold_choice == 4:
        output_train_list = (
                some_list[: 3 * int(val_split * some_list_len)]
                + some_list[4 * int(val_split * some_list_len): int(some_list_len * (train_split + val_split))]
        )
        output_val_list = some_list[3 * int(val_split * some_list_len): 4 * int(some_list_len * val_split)]
    output_inf_list = some_list[int(some_list_len * (train_split + val_split)):]
    return output_train_list, output_val_list, output_inf_list


def get_data_dicts(use_csv: bool = True, ids_path: str = None, shuffle: bool = False, first_n=False):

    """Get data dicts for data loaders."""
    data_dicts = []
    if use_csv:
        df = pd.read_csv(ids_path, sep=",")
        if shuffle:
            df = df.sample(frac=1, random_state=1)
        df = list(df)
        for row in df:
            data_dicts.append({"image": (row)})
    else:
        # Use ids_path as a directory
        # ids_path = ids_path[:100]
        for direc in ids_path:
            for file in os.listdir(os.path.join(direc, "anat")):
                data_dicts.append({"image": os.path.join(direc, "anat", file)})
    if first_n is not False:
        data_dicts = data_dicts[:first_n]

    print(f"Found {len(data_dicts)} subjects.")
    if dist.is_initialized():
        print(dist.get_rank())
        print(dist.get_world_size())
        return partition_dataset(
            data=data_dicts,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        return data_dicts


def get_training_data_loader(
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    only_val: bool = False,
    augmentation: bool = True,
    drop_last: bool = False,
    num_workers: int = 8,
    num_val_workers: int = 3,
    cache_data=True,
    first_n=None,
    is_grayscale=False,
    add_vflip=False,
    add_hflip=False,
    image_size=None,
    image_roi=None,
    spatial_dimension=2,
):
    # Define transformations
    resize_transform = (
        transforms.ResizeD(keys=["image"], spatial_size=(image_size,) * spatial_dimension)
        if image_size
        else lambda x: x
    )

    central_crop_transform = (
        transforms.CenterSpatialCropD(keys=["image"], roi_size=image_roi)
        if image_roi
        else lambda x: x
    )

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]) if is_grayscale else lambda x: x,
            transforms.Lambdad(keys="image", func=lambda x: x[0, None, ...])
            if is_grayscale
            else lambda x: x,  # needed for BRATs data with 4 modalities in 1
            central_crop_transform,
            resize_transform,
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.RandFlipD(keys=["image"], spatial_axis=0, prob=1.0)
            if add_vflip
            else lambda x: x,
            transforms.RandFlipD(keys=["image"], spatial_axis=1, prob=1.0)
            if add_hflip
            else lambda x: x,
            transforms.ToTensord(keys=["image"]),
        ]
    )

    # no augmentation for now
    if augmentation:
        train_transforms = val_transforms
    else:
        train_transforms = val_transforms

    # Create data dicts without resorting to csv
    val_dicts = get_data_dicts(ids_path=validation_ids, shuffle=False, first_n=first_n, use_csv=False)

    if first_n:
        val_dicts = val_dicts[:first_n]

    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )
    print(val_ds[0]["image"].shape)
    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(ids_path=training_ids, shuffle=False, first_n=first_n, use_csv=False)

    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )
    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    return train_loader, val_loader
