import torch
from collections import Counter
from .get_entity import get_entities


class SeqEntityScore(object):
    def __init__(self, id2label, markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.groudtruths = []
        self.predictions = []
        self.corrects = []

    @classmethod
    def compute(cls, groudtruth, prediction, correct):
        recall = 0 if groudtruth == 0 else (correct / groudtruth)
        precision = 0 if prediction == 0 else (correct / prediction)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        groudtruth_counter = Counter([x[0] for x in self.groudtruths])
        prediction_counter = Counter([x[0] for x in self.predictions])
        correct_counter = Counter([x[0] for x in self.corrects])
        for type_, groudtruth in groudtruth_counter.items():
            prediction = prediction_counter.get(type_, 0)
            correct = correct_counter.get(type_, 0)
            recall, precision, f1 = self.compute(groudtruth, prediction, correct)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        groudtruth = len(self.groudtruths)
        prediction = len(self.predictions)
        correct = len(self.corrects)
        recall, precision, f1 = self.compute(groudtruth, prediction, correct)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_paths, pred_paths):
        """
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        """
        for label_path, pred_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label, self.markup)
            pred_entities = get_entities(pred_path, self.id2label, self.markup)
            self.groudtruths.extend(label_entities)
            self.predictions.extend(pred_entities)
            self.corrects.extend([pre_entity for pre_entity in pred_entities if pre_entity in label_entities])


class SpanEntityScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.groudtruths = []
        self.predictions = []
        self.corrects = []

    @classmethod
    def compute(cls, groudtruth, prediction, correct):
        recall = 0 if groudtruth == 0 else (correct / groudtruth)
        precision = 0 if prediction == 0 else (correct / prediction)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        groudtruth_counter = Counter([self.id2label[x[0]] for x in self.groudtruths])
        prediction_counter = Counter([self.id2label[x[0]] for x in self.predictions])
        correct_counter = Counter([self.id2label[x[0]] for x in self.corrects])
        for type_, groudtruth in groudtruth_counter.items():
            prediction = prediction_counter.get(type_, 0)
            correct = correct_counter.get(type_, 0)
            recall, precision, f1 = self.compute(groudtruth, prediction, correct)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        groudtruth = len(self.groudtruths)
        prediction = len(self.predictions)
        correct = len(self.corrects)
        recall, precision, f1 = self.compute(groudtruth, prediction, correct)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, true_subject, pred_subject):
        self.groudtruths.extend(true_subject)
        self.predictions.extend(pred_subject)
        self.corrects.extend([pred_entity for pred_entity in pred_subject if pred_entity in true_subject])
