from PIL import ImageOps, Image
import numpy as np
import albumentations as A

class Equalize(object):

    def __call__(self, image):
        """
        Equalize the image histogram. This function applies a non-linear
        mapping to the input image, in order to create a uniform
        distribution values in the output image.

        :param image: The image to equalize.
        :return: An image.
        """

        # if image.mode == "P":
        #     image = image.convert("RGB")

        # h = image.histogram(mask=None)
        # lut = []
        # for b in range(0, len(h), 256):
        #     histo = [_f for _f in h[b : b + 256] if _f]
        #     if len(histo) <= 1:
        #         lut.extend(list(range(256)))
        #     else:
        #         step = (functools.reduce(operator.add, histo) - histo[-1]) // 255
        #         if not step:
        #             lut.extend(list(range(256)))
        #         else:
        #             n = step // 2
        #             for i in range(256):
        #                 lut.append(n // step)
        #                 n = n + h[i + b]

        # if image.mode == "RGB" and len(lut) == 256:
        #     lut = lut + lut + lut

        # return image.point(lut)
        return ImageOps.equalize(image, mask = None)


class GrayScale(object):

    def __call__(self, image):

        return Image.fromarray(np.stack((np.array(image.convert("L")),)*3, axis=-1))


class Albumentations(object):
    
    def __call__(self, image):
        image = np.array(image)

        transform = A.Compose([
        A.RGBShift (r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, always_apply=False, p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),        
        A.Flip(p=0.5),
        A.AdvancedBlur(p=0.7),
    ])

        transformed_image = transform(image=image)['image']

        return Image.fromarray(transformed_image)
