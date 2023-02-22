import pandas as pd
from seqeval.metrics import classification_report as ner_cls_report
from sklearn.metrics import classification_report as tag_cls_report


class SingleEval(object):
    def __init__(self):
        self.groudtruths = []
        self.predictions = []

    def update(self, label_paths, pred_paths):
        self.groudtruths.extend(label_paths)
        self.predictions.extend(pred_paths)

    def entity_eval(self):
        report = ner_cls_report(self.groudtruths, self.predictions, digits=4, output_dict=True)
        return report

    def report(self):
        entity_report = self.entity_eval()
        print()
        print(ner_cls_report(self.groudtruths, self.predictions, digits=4))
        print()
        return entity_report
