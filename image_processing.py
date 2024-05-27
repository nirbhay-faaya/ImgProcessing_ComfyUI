import torch 
import torchvision
import torchvision.transforms as transforms 
import numpy as np 
from PIL import Image, ImageEnhance, ImageFilter
import cv2 

MAX_RESOLUTION=8192

class ImageCropMultEight:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imgcrop"
    
    def imgcrop(self, image, width, height, position, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)
        
        keep_mult_8 = lambda x: x if x % 8 == 0 else x - (x % 8)
        width = keep_mult_8(width)
        height = keep_mult_8(height)
                
        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width
        
        x += x_offset
        y += y_offset
        
        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return(image, )

class LightingPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imgprocess"
    
    def imgprocess(self, image):
        "image_size: [B, H, W, C]"
        def gethw(img_h, img_w, target_size):
            if img_h < img_w:
                new_h = target_size
                new_w = int(img_w * new_h / img_h)
            elif img_h > img_w:
                new_w = target_size
                new_h = int(img_h * new_w / img_w)
            else:
                new_h, new_w = target_size, target_size
            return (new_w, new_h)
        # 打开原始图像
        img = image.clone()
        device = img.device 
        img = Image.fromarray(np.uint8(img.squeeze().cpu().numpy() * 255))

        img = img.convert('L')
        # img = ImageOps.equalize(img, mask=None)
        W, H = img.size
        img = img.resize(gethw(H, W, 1536), resample=Image.BILINEAR)
        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(1.3)
        img = img_contrast

        # 使用高斯模糊去除一部分细节
        img = img.filter(ImageFilter.GaussianBlur(radius=5))
        img = img.filter(ImageFilter.BoxBlur(radius=15))
        img = img.filter(ImageFilter.GaussianBlur(radius=10))

        # 使用中值滤波器去除更多的细节
        img = img.filter(ImageFilter.MedianFilter(size=3))
        img= img.filter(ImageFilter.SMOOTH)
        # 使用边缘增强滤波器保留光影和色块的模糊关系

        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=1))

        img_array = np.array(img)

        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 将图像转换为灰度图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 对图像进行自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
        cl_img = clahe.apply(gray)

        # 将图像转换为 Pillow 格式，并显示处理后的图像
        img = Image.fromarray(cl_img)
        img = img.resize((img.size[0] // 16, img.size[1] // 16), resample=Image.BOX)
        img = img.resize((img.size[0] * 16, img.size[1] * 16), resample=Image.NEAREST)
        img = img.filter(ImageFilter.GaussianBlur(radius=10))
        enhancer = ImageEnhance.Contrast(img)
        img_contrast = enhancer.enhance(1.3)
        img = img_contrast
        W, H = img.size
        img = img.resize(gethw(H, W, 512), resample=Image.BILINEAR)
        img = transforms.ToTensor()(img).unsqueeze(0).permute(0,2,3,1).to(device)
        img = torch.concat([img, img, img], dim=-1)
        
        return (img,)
    

class ImageCut:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), 
                "H_cut_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Width_padding": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "cut_image"
    
    def cut_image(self, image, H_cut_ratio, Width_padding):
        """
        Image size: [B, H, W, C]
        """
        B, H, W, C = image.shape
        h_cut = int(H_cut_ratio * H)
        w_cut = int(Width_padding * W)
        
        
        w_slice = slice(w_cut, -w_cut)
        h_slice_upper = slice(0, h_cut)
        h_slice_lower = slice(h_cut, H)
        if w_cut == 0:
            w_slice = slice(0, W)
            
        if h_cut == 0 or h_cut == H:
            h_slice_upper = slice(0, H)
            h_slice_lower = slice(0, H)
        
        image_upper = image[:, h_slice_upper, w_slice, :]
        image_lower = image[:, h_slice_lower, w_slice, :]
        return (image_upper, image_lower)
    
class ImageConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",), 
                "image2": ("IMAGE",),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concat_images"
    
    def concat_images(self, image1, image2):
        """
        images_size: [B, H, W, C]
        """
        
        _, _, W, _ = image1.shape
        _, H, _, _ = image2.shape
        image2 = torch.nn.functional.interpolate(image2.permute(0, 3, 1, 2), size=(H, W), mode="bicubic", antialias=True).permute(0, 2, 3, 1)
        image = torch.cat([image1, image2], dim = 1)
        return (image,)


NODE_CLASS_MAPPINGS = {
    "ImageCut": ImageCut,
    "ImageConcat": ImageConcat,
    "LightingPreprocessor": LightingPreprocessor,
    "ImageCropMultEight": ImageCropMultEight,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageCut": "ImageCut",
    "ImageConcat": "ImageConcat",
    "LightingPreprocessor": "LightingPreprocessor",
    "ImageCropMultEight": "ImageCropMultEight",
}