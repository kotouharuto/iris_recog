"""
Feature extraction using Gabor filters and encoding.
"""
import cv2
import numpy as np
from . import config


class IrisFeatureExtractor:
    def __init__(self):
        self.norm_width = config.NORM_WIDTH
        # Prepare kernels
        self.kernel_real = cv2.getGaborKernel(
            (config.GABOR_KSIZE, config.GABOR_KSIZE),
            config.GABOR_SIGMA, config.GABOR_THETA, config.GABOR_LAMBD, config.GABOR_GAMMA,
            0, ktype=cv2.CV_32F
        )
        self.kernel_imag = cv2.getGaborKernel(
            (config.GABOR_KSIZE, config.GABOR_KSIZE),
            config.GABOR_SIGMA, config.GABOR_THETA, config.GABOR_LAMBD, config.GABOR_GAMMA,
            np.pi / 2, ktype=cv2.CV_32F
        )

    def extract_features(self, norm_img):
        """
        Apply Gabor filters and generate iris code and mask.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        norm_img_enhanced = clahe.apply(norm_img)

        f_real = cv2.filter2D(norm_img_enhanced, cv2.CV_32F, self.kernel_real)
        f_imag = cv2.filter2D(norm_img_enhanced, cv2.CV_32F, self.kernel_imag)

        # Quantization
        code_real = np.where(f_real > 0, 1, 0).astype(np.int8)
        code_imag = np.where(f_imag > 0, 1, 0).astype(np.int8)

        # Mask generation
        mask = np.zeros_like(code_real, dtype=np.int8)
        w = self.norm_width

        # Mask regions prone to noise (eyelids/lashes)
        mask[:, :int(w / 6)] = 1
        mask[:, int(w * 5 / 6):] = 1

        center = int(w / 2)
        span = int(w / 6)
        mask[:, center - span: center + span] = 1

        mask[50:, :] = 0
        mask[:5, :] = 0

        # Intensity mask
        intensity_mask = np.where((norm_img > 20) & (norm_img < 240), 1, 0).astype(np.int8)
        final_mask = mask & intensity_mask

        # Concatenate Real and Imaginary parts
        iris_code = np.concatenate([code_real, code_imag], axis=0)
        iris_mask = np.concatenate([final_mask, final_mask], axis=0)

        return iris_code, None, iris_mask