"""Custom folder-based hierarchical dataset."""
from typing import Optional, Callable, Any, Tuple

import torchvision.datasets as datasets
import torchvision.datasets.folder as folder


class ImageFolder(datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = folder.default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        is_hier: bool = True,
    ):
        super(ImageFolder, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
        )
        self.is_hier = is_hier

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        path, target = self.samples[index]
        coarse_target = target
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            coarse_target = self.target_transform(coarse_target)

        if self.is_hier:
            return sample, target, coarse_target
        return sample, target
