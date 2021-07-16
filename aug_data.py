import random
import os
import numpy as np
from PIL import Image
# Image.MAX_IMAGE_PIXELS = 933120000            --Use this when pixels exceed maximmum size, but not recommended
from torchvision import transforms
import torchvision.transforms.functional as tf
 
class Augmentation:
    def __init__(self):
        pass
    
    def rotate(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-35, 35]) # -35~35 randomly choose an angle to rotate
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def rotate_one(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-270, 120]) # -270~120 randomly choose an angle to rotate
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def rotate_two(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-150, 150]) # -150~150 randomly choose an angle to rotate
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def rotate_three(self,image,mask,angle=None):
        if angle == None:
            angle = transforms.RandomRotation.get_params([-120, 270]) # -120~270 randomly choose an angle to rotate
        if isinstance(angle,list):
            angle = random.choice(angle)
        image = image.rotate(angle)
        mask = mask.rotate(angle)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def flip_h(self,image,mask): #Horizontal flip
        if random.random()>= 0.5:
            image = tf.hflip(image)
            mask = tf.hflip(mask)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def flip_v(self,image,mask): #Vertical flip
        if random.random()>=0.1:
            image = tf.vflip(image)
            mask = tf.vflip(mask)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask


    def randomResizeCrop(self,image,mask,scale=(0.3,1.0),ratio=(1,1)): #scale indicates that the randomly cropped image will be between 0.3 and 1 times, and ratio indicates the aspect ratio
        img = np.array(image)
        h_image, w_image = img.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def randomResizeCrop_one(self,image,mask,scale=(0.7,1.0),ratio=(1,1)): #scale indicates that the randomly cropped image will be between 0.7 and 1 times, and ratio indicates the aspect ratio
        img = np.array(image)
        h_image, w_image = img.shape
        resize_size = h_image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
        image = tf.resized_crop(image, i, j, h, w, resize_size)
        mask = tf.resized_crop(mask, i, j, h, w, resize_size)
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def adjustBrightness(self,image,mask):
        factor = transforms.RandomRotation.get_params([1, 2]) #Here adjust the brightness of the data after expansion
        image = tf.adjust_brightness(image, factor)
        # mask = tf.adjust_contrast(mask, factor)             # mask needs to stay the same
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask

    def centerCrop(self,image,mask,size=None): #Center crop
        if size == None:
            size = image.size #If size is not set, it is the original image.
            image = tf.center_crop(image,size)
            mask = tf.center_crop(mask,size)
            image = tf.to_tensor(image)
            mask = tf.to_tensor(mask)
            return image,mask

    def adjustSaturation(self,image,mask):  #Adjust saturation
        factor = transforms.RandomRotation.get_params([1, 2]) # Adjust the brightness of the data after expansion
        image = tf.adjust_saturation(image, factor)
        # mask = tf.adjust_saturation(mask, factor)           # mask needs to stay the same
        image = tf.to_tensor(image)
        mask = tf.to_tensor(mask)
        return image, mask
 
 
def augmentationData(image_path,mask_path,option=[1,2,3],save_dir=None):
    '''
         :param image_path: the path of the image
         :param mask_path: path of the mask
         :param option: Which augmentation method is needed: 1 is rotation, 2 is horizontal flipping, 3 is vertical flipping , 4 is random cropping and restoring the original size, 
                                                             5 is center cropping (not restoring the original size), 6 is adjusting brightness, 7 Is saturation, 8 is rotation(through a different angle), 
                                                             9 is rotation(through a different angle), 10 is rotation(through a different angle), 11 is random cropping and restoring the original size(using a different cropping ratio)
         :param save_dir: The path where the augmented data is stored
    '''
    aug_image_savedDir = os.path.join(save_dir,'img')
    aug_mask_savedDir = os.path.join(save_dir, 'mask')

    if not os.path.exists(aug_image_savedDir):
        os.makedirs(aug_image_savedDir)
        print('create aug image dir.....')

    if not os.path.exists(aug_mask_savedDir):
        os.makedirs(aug_mask_savedDir)
        print('create aug mask dir.....')
    aug = Augmentation()
    res= os.walk(image_path)
    images = []
    masks = []

    for root,dirs,files in res:
        for f in files:
            images.append(os.path.join(root,f))
    res = os.walk(mask_path)

    for root,dirs,files in res:
        for f in files:
            masks.append(os.path.join(root,f))
    datas = list(zip(images,masks))
    num = len(datas)
 
    for (image_path,mask_path) in datas:
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if 1 in option:
            num+=1
            image_tensor, mask_tensor = aug.rotate(image, mask)
            image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_rotate.png'))
            mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_rotate.png'))

        if 2 in option:
            num+=1
            image_tensor, mask_tensor = aug.flip_h(image, mask)
            image_filp = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir,'img',str(num)+'_flipH.png'))
            mask_filp = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir,'mask',str(num)+'_flipH.png'))

        if 3 in option:
            num+=1
            image_tensor, mask_tensor = aug.flip_v(image, mask)
            image_filp = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir,'img',str(num)+'_flipV.png'))
            mask_filp = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir,'mask',str(num)+'_flipV.png'))    

        if 4 in option:
            num+=1
            image_tensor, mask_tensor = aug.randomResizeCrop(image, mask)
            image_ResizeCrop = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_ResizeCrop.png'))
            mask_ResizeCrop = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_ResizeCrop.png'))

    
        if 5 in option:
            num+=1
            image_tensor, mask_tensor = aug.centerCrop(image, mask)
            image_centerCrop = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_centerCrop.png'))
            mask_centerCrop = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_centerCrop.png'))

        if 6 in option:
            num+=1
            image_tensor, mask_tensor = aug.adjustBrightness(image, mask)
            image_Brightness = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_Brightness.jpg'))
            mask_Brightness = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_Brightness.jpg'))

        if 7 in option:
            num+=1
            image_tensor, mask_tensor = aug.adjustSaturation(image, mask)
            image_Saturation = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_Saturation.png'))
            mask_Saturation = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_Saturation.png'))

        if 8 in option:
            num+=1
            image_tensor, mask_tensor = aug.rotate_one(image, mask)
            image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_rotate1.png'))
            mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_rotate1.png'))

        if 9 in option:
            num+=1
            image_tensor, mask_tensor = aug.rotate_two(image, mask)
            image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_rotate2.png'))
            mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_rotate2.png'))

        if 10 in option:
            num+=1
            image_tensor, mask_tensor = aug.rotate_three(image, mask)
            image_rotate = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_rotate3.png'))
            mask_rotate = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_rotate3.png'))
            
        if 11 in option:
            num+=1
            image_tensor, mask_tensor = aug.randomResizeCrop_one(image, mask)
            image_ResizeCrop = transforms.ToPILImage()(image_tensor).save(os.path.join(save_dir, 'img', str(num) + '_ResizeCrop1.png'))
            mask_ResizeCrop = transforms.ToPILImage()(mask_tensor).save(os.path.join(save_dir, 'mask', str(num) + '_ResizeCrop1.png'))
 
 
#Print out augmentated images
augmentationData(r'C:\Users\winuser\Videos\Captures\Val_loss\original-49-images', 
                    r'C:\Users\winuser\Videos\Captures\Val_loss\original-49-masks', 
                        save_dir=r'C:\Users\winuser\Videos\Captures\Val_loss\augmentetd-data')