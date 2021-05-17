import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes_ID = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes_ID.sort()
    class_to_idx_ID = {classes_ID[i]: i for i in range(len(classes_ID))}
    return classes_ID, class_to_idx_ID 



def make_dataset(dir, class_to_idxID):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    f_class = d.split('/')[-1]

                    path_rgb = os.path.join(root, fname)

                    item = (path_rgb, class_to_idxID[f_class], class_to_idxID)

                    images.append(item)

    return images

def make_depth_dataset(dir, class_to_idxID):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    f_class = d.split('/')[-1]
                    # print("f_class", f_class)
                    fname_id = fname.rsplit("_",1)[0]
                    fname_depth = fname_id + '_depthcrop.png'

                    # print("fname_depth", fname_depth)
                    depth_set = dir.split("/")[-2]
                    depth_root = root.rsplit("/",2)[0]+'/'+depth_set+'/'+d.split('/')[-1]+'/'

                    path_depth = os.path.join(depth_root, fname_depth)

                    item = (path_depth, class_to_idxID[f_class], class_to_idxID)

                    images.append(item)

    return images

def make_total_dataset(dir, depthdir, class_to_idxID):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    f_class = d.split('/')[-1]
                    fname_id = fname.rsplit("_",1)[0]
                    fname_depth = fname_id + '_depthcrop.png'

                    # depth_set = dir.split("/")[-2]
                    depth_root = os.path.join(depthdir, d.split('/')[-1])

                    path_rgb = os.path.join(root, fname)
                    path_depth = os.path.join(depth_root, fname_depth)

                    item = (path_rgb, path_depth, class_to_idxID[f_class], class_to_idxID)

                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(data.Dataset):

    def __init__(self, rgbroot, transform=None, 
                 target_transform=None, loader=default_loader):
        classesID, class_to_idxID = find_classes(rgbroot)
        imgs = make_dataset(rgbroot, class_to_idxID)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.rgbroot = rgbroot
        self.imgs = imgs
        self.classesID = classesID
        self.class_to_idxID = class_to_idxID
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        tar = [None] * 3
        path_rgb, tarID, tar_dict = self.imgs[index]
        rgb_img = self.loader(path_rgb)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return rgb_img, tarID, path_rgb, tar_dict

    def __len__(self):
        return len(self.imgs)


class DepthImageFolder(data.Dataset):

    def __init__(self, depthroot, depth_transform=None, 
                 target_transform=None, loader=default_loader):
        classesID, class_to_idxID = find_classes(depthroot)
        imgs = make_depth_dataset(depthroot, class_to_idxID)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.depthroot = depthroot
        self.imgs = imgs
        self.classesID = classesID
        self.class_to_idxID = class_to_idxID
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        tar = [None] * 3
        path_depth, tarID, tar_dict = self.imgs[index]
        depth_img = self.loader(path_depth)
        if self.depth_transform is not None:
            depth_img = self.depth_transform(depth_img)
        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return depth_img, tarID, path_depth, tar_dict

    def __len__(self):
        return len(self.imgs)


class TotalImageFolder(data.Dataset):

    def __init__(self, rgbroot, depthroot, transform=None, depth_transform=None, 
                 target_transform=None, depth_target_transform=None, loader=default_loader):
        classesID, class_to_idxID = find_classes(rgbroot)
        # classesID_d, class_to_idxID_d = find_classes(depthroot)
        imgs = make_total_dataset(rgbroot, depthroot, class_to_idxID)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.rgbroot = rgbroot
        self.depthroot = depthroot
        self.imgs = imgs
        self.classesID = classesID
        self.class_to_idxID = class_to_idxID
        self.transform = transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        tar = [None] * 3
        path_rgb, path_depth, tarID, tar_dict = self.imgs[index]
        rgb_img = self.loader(path_rgb)
        depth_img = self.loader(path_depth)
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        if self.depth_transform is not None:
            depth_img = self.depth_transform(depth_img)
        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return rgb_img, depth_img, tarID, path_rgb, path_depth, tar_dict

    def __len__(self):
        return len(self.imgs)

        

class PathFolder(data.Dataset):


    def __init__(self, pseudolist, transform=None, 
                 target_transform=None, loader=default_loader):
        if len(pseudolist) == 0:
            raise(RuntimeError("Found 0 images in pseudo dataset"))

        self.imgs = pseudolist
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_rgb, tarID, confid_rate, dom_variance, weights, true_label = self.imgs[index]
        # print(path_rgb)
        rgb_img = self.loader(path_rgb[0])
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        # if self.depth_transform is not None:
        #     depth_img = self.depth_transform(depth_img)
        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return rgb_img, tarID, confid_rate, dom_variance, weights, true_label

    def __len__(self):
        return len(self.imgs)
                        
class TotalPathFolder(data.Dataset):


    def __init__(self, pseudolist, transform=None, depth_transform=None,
                 target_transform=None, loader=default_loader):
        if len(pseudolist) == 0:
            raise(RuntimeError("Found 0 images in pseudo dataset"))

        self.imgs = pseudolist
        self.transform = transform
        self.depth_transform = depth_transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path_rgb, path_d, tarID, confid_rate, dom_variance, weights, true_label = self.imgs[index]
        # print(path_rgb)
        if len(path_rgb) != 0:
            rgb_img = self.loader(path_rgb[0])
            if self.transform is not None:
                rgb_img = self.transform(rgb_img)
        else:
            rgb_img = []

        if len(path_d) != 0:
            depth_img = self.loader(path_d[0])
            if self.depth_transform is not None:
                depth_img = self.depth_transform(depth_img)
        else:
            depth_img = []

        if self.target_transform is not None:
            tarID = self.target_transform(tarID)

        return rgb_img, depth_img, tarID, confid_rate, dom_variance, weights, true_label

    def __len__(self):
        return len(self.imgs)
                        