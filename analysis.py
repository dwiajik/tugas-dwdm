from os import listdir
from os.path import isfile, join, dirname
import random
import re

from svm import Classifier

def f_measure(precision, recall):
    return 2*((precision * recall) / (precision + recall))

pos_path = 'Movie Reviews Data Set/review_polarity/txt_sentoken/pos'
neg_path = 'Movie Reviews Data Set/review_polarity/txt_sentoken/neg'

pos_files = [join(pos_path, f) for f in listdir(pos_path) if isfile(join(pos_path, f))]
neg_files = [join(neg_path, f) for f in listdir(neg_path) if isfile(join(neg_path, f))]

pos_reviews = []
for file_path in pos_files:
    with open(file_path) as f:
        text = f.read().replace('\n', ' ')
        #text = re.sub(r'\s[^a-zA-Z0-9]', ' ', text)

        pos_reviews.append((text, 'pos'))

neg_reviews = []
for file_path in neg_files:
    with open(file_path) as f:
        text = f.read().replace('\n', ' ')
        #text = re.sub(r'\s[^a-zA-Z0-9]', ' ', text)

        neg_reviews.append((text, 'neg'))

labeled_reviews = pos_reviews + neg_reviews
random.shuffle(labeled_reviews)

svm_times = []
svm_true_positives = []
svm_true_negatives = []
svm_false_positives = []
svm_false_negatives = []
svm_accuracies = []
svm_precisions = []
svm_recalls = []
svm_f_measures = []

data_format = '"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"\r\n'
with open(dirname(__file__) + 'evaluation.csv', 'a') as f:
    f.write(data_format.format(
        'Iteration',
        'Training time',
        'True positive',
        'True negative',
        'False positive',
        'False negative',
        'Accuracy',
        'Precision',
        'Recall',
        'F-measure'
    ))

fold = 10

for i in range(fold):
    train_set = labeled_reviews[0:i*int(len(labeled_reviews)/fold)] + labeled_reviews[(i+1)*int(len(labeled_reviews)/fold):len(labeled_reviews)]
    test_set = labeled_reviews[i*int(len(labeled_reviews)/fold):(i+1)*int(len(labeled_reviews)/fold)]

    print('\nIteration', (i+1))
    print('Training data:', len(train_set), 'data')
    print('Test data:', len(test_set), 'data')

    # SVM
    svm_classifier = Classifier(train_set)

    svm_true_positive = 0
    svm_true_negative = 0
    svm_false_positive = 0
    svm_false_negative = 0
    for index, (feature, label) in enumerate(test_set):
        observed = svm_classifier.classify(feature)
        if label == 'pos' and observed == 'pos':
            svm_true_positive += 1
        if label == 'neg' and observed == 'neg':
            svm_true_negative += 1
        if label == 'pos' and observed == 'neg':
            svm_false_negative += 1
        if label == 'neg' and observed == 'pos':
            svm_false_positive += 1

    svm_time = svm_classifier.get_training_time()
    svm_accuracy = (svm_true_positive + svm_true_negative) / (svm_true_positive + svm_true_negative + svm_false_positive + svm_false_negative)
    svm_precision = svm_true_positive / (svm_true_positive + svm_false_positive)
    svm_recall = svm_true_positive / (svm_true_positive + svm_false_negative)
    svm_f_measure = f_measure(svm_precision, svm_recall)

    svm_times.append(svm_time)
    svm_true_positives.append(svm_true_positive)
    svm_true_negatives.append(svm_true_negative)
    svm_false_positives.append(svm_false_positive)
    svm_false_negatives.append(svm_false_negative)
    svm_accuracies.append(svm_accuracy)
    svm_precisions.append(svm_precision)
    svm_recalls.append(svm_recall)
    svm_f_measures.append(svm_f_measure)

    with open(dirname(__file__) + 'evaluation.csv', 'a') as f:
        f.write(data_format.format(
            i + 1,
            svm_time,
            svm_true_positive,
            svm_true_negative,
            svm_false_positive,
            svm_false_negative,
            svm_accuracy,
            svm_precision,
            svm_recall,
            svm_f_measure
        ))

    print('SVM Classifier:')
    print('\t', 'Training time:', svm_time)
    print('\t', 'True positive:', svm_true_positive)
    print('\t', 'True negative:', svm_true_negative)
    print('\t', 'False positive:', svm_false_positive)
    print('\t', 'False negative:', svm_false_negative)
    print('\t', 'Accuracy:', svm_accuracy)
    print('\t', 'Precision:', svm_precision)
    print('\t', 'Recall:', svm_recall)
    print('\t', 'F-Measure:', svm_f_measure)


with open(dirname(__file__) + 'evaluation.csv', 'a') as f:
    f.write((data_format + '\r\n\r\n').format(
        'Total',
        sum(svm_times) / len(svm_times),
        sum(svm_true_positives) / len(svm_true_positives),
        sum(svm_true_negatives) / len(svm_true_negatives),
        sum(svm_false_positives) / len(svm_false_positives),
        sum(svm_false_negatives) / len(svm_false_negatives),
        sum(svm_accuracies) / len(svm_accuracies),
        sum(svm_precisions) / len(svm_precisions),
        sum(svm_recalls) / len(svm_recalls),
        sum(svm_f_measures) / len(svm_f_measures)
    ))

print('\nSummary SVM Classifier:')
print('\tAverage training time:', sum(svm_times) / len(svm_times))
print('\tAverage true positive:', sum(svm_true_positives) / len(svm_true_positives))
print('\tAverage true negative:', sum(svm_true_negatives) / len(svm_true_negatives))
print('\tAverage false positives:', sum(svm_false_positives) / len(svm_false_positives))
print('\tAverage false negatives:', sum(svm_false_negatives) / len(svm_false_negatives))
print('\tAverage accuracy:', sum(svm_accuracies) / len(svm_accuracies))
print('\tAverage precision:', sum(svm_precisions) / len(svm_precisions))
print('\tAverage recall:', sum(svm_recalls) / len(svm_recalls))
print('\tAverage F-Measure:', sum(svm_f_measures) / len(svm_f_measures))
