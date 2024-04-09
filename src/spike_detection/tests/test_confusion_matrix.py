"""
Test the formula for accuracy described in 9/22/2022 in Journal
"""

confusion = [23, 32, 12, 41]  # False Negatives, True Negatives, False Positives, True Positives
accuracy_label = (confusion[1] + confusion[3]) / sum(confusion)
print(accuracy_label)

total = sum(confusion)
positives_predicted = confusion[2] + confusion[3]
positives_label = confusion[0] + confusion[3]
positives_true = confusion[3]
negatives_true = total - (positives_predicted + positives_label - positives_true)
accuracy_pred = (positives_true + negatives_true) / total
print(accuracy_pred)

