"""
Test the difference between f1 score and simply averaging recall and precision
"""

import random
import numpy as np

recall = np.nan  # random.random()
precision = random.random()

f1 = 2 * (precision * recall) / (precision + recall)
avg = (precision + recall) / 2

print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Avg: {avg:.4f}")
