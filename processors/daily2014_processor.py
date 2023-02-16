import os
import json
import random
import logging
from tqdm import tqdm
from .base_processor import DataProcessor, InputExample


logger = logging.getLogger()


class Daily2014Processor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["B-PER", "I-PER", "B-T", "I-T", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "O"]

    @classmethod
    def _create_examples(cls, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = json.loads(line)
            text_a = line["words"]
            # BIOS
            labels = line["labels"]
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                _line = line.strip("\n")
                lines.append(_line)
            return lines

    def data_preprocess(self, data_dir):
        train_datapath = os.path.join(data_dir, "train.json")
        dev_datapath = os.path.join(data_dir, "dev.json")

        # if os.path.exists(train_datapath) and os.path.exists(dev_datapath):
        #     return

        targets, sentences = [], []
        raw_source = os.path.join(data_dir, "source_BIO_2014_corpus.txt")
        tgt_source = os.path.join(data_dir, "target_BIO_2014_corpus.txt")

        with open(raw_source, "r") as fr_1, open(tgt_source, "r") as fr_2:
            for sent, target in tqdm(zip(fr_1, fr_2), desc="Sent2char"):
                chars = sent2char(sent)
                label = sent2char(target)
                label = [l.replace("_", "-") for l in label]
                targets.append(label)
                sentences.append(chars)

        train, valid = train_val_split(sentences, targets)
        train = train[:10000]
        with open(train_datapath, "w") as fw:
            for sent, label in train:
                df = {"words": sent, "labels": label}
                encode_json = json.dumps(df)
                print(encode_json, file=fw)
            logger.info("Train data write done")

        with open(dev_datapath, 'w') as fw:
            for sent, label in valid:
                df = {"words": sent, "labels": label}
                encode_json = json.dumps(df)
                print(encode_json, file=fw)
            logger.info("Dev data write done")


def sent2char(line):
    res = line.strip('\n').split()
    return res


def train_val_split(X, y, valid_size=0.2, random_state=2022, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param valid_size
    :param random_state: 随机种子
    :param shuffle
    """
    data = []
    for data_x, data_y in zip(X, y):
        data.append((data_x, data_y))
    del X, y

    test_size = int(len(data) * valid_size)

    if shuffle:
        random.seed(random_state)
        random.shuffle(data)

    valid = data[:test_size]
    train = data[test_size:]

    return train, valid
