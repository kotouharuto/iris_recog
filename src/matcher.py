"""
Matching logic using Hamming distance.
"""
import numpy as np


class IrisMatcher:
    def compute_hamming_distance(self, code1, code2, mask1, mask2, shift_range=8):
        """
        Calculate the lowest Hamming distance between two iris codes accounting for rotation shift.
        """
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            return 1.0

        min_score = 1.0
        for shift in range(-shift_range, shift_range + 1):
            code1_shifted = np.roll(code1, shift, axis=1)
            mask1_shifted = np.roll(mask1, shift, axis=1)

            valid_mask = np.logical_and(mask1_shifted, mask2)
            total_valid_bits = np.sum(valid_mask)

            if total_valid_bits < 100:
                continue

            xor_result = np.logical_xor(code1_shifted, code2)
            mismatch_count = np.sum(np.logical_and(xor_result, valid_mask))

            score = mismatch_count / total_valid_bits
            if score < min_score:
                min_score = score

        return min_score