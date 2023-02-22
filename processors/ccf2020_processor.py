import os
import json
import random
import logging
from tqdm import tqdm
from .base_processor import DataProcessor, InputExample


logger = logging.getLogger()


class Ccf2020Processor(DataProcessor):
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
        return ["X", "B-position", "I-position", "B-name", "I-name", "B-movie", "I-movie",
                "B-organization", "I-organization", "B-company", "I-company", "B-game", "I-game", "B-book", "I-book",
                "B-address", "I-address", "B-scene", "I-scene", "B-government", "I-government",
                # "B-email", "I-email", "B-mobile", "I-mobile", "B-QQ", "I-QQ", "B-vx", "I-vx",
                "O"]

    @classmethod
    def _create_examples(cls, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["words"]
            # BIOS
            labels = line["labels"]
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

    @classmethod
    def _read_json(cls, input_file):
        lines = []
        with open(input_file, "r") as f:
            for line in f:
                line = json.loads(line.strip())
                text = line["text"]
                label_entities = line.get("entities", None)
                words = list(text)
                labels = ["O"] * len(words)
                if label_entities is not None:
                    for entity in label_entities:
                        pos_b = entity.get("pos_b")
                        pos_e = entity.get("pos_e")
                        category = entity.get("category")
                        privacy = entity.get("privacy")
                        if category in ["email", "mobile", "QQ", "vx"]:
                            continue
                        assert "".join(words[pos_b: pos_e+1]) == privacy
                        labels[pos_b] = "B-" + category
                        labels[pos_b+1: pos_e+1] = ["I-"+category] * (len(privacy)-1)

                lines.append({"words": words, "labels": labels})

        return lines

    def data_preprocess(self, data_dir):
        train_datapath = os.path.join(data_dir, "train.json")
        dev_datapath = os.path.join(data_dir, "dev.json")

        if os.path.exists(train_datapath) and os.path.exists(dev_datapath):
            return

        raw_train_file = os.path.join(data_dir, "raw_train.json")
        data = read_json_line_file(raw_train_file)
        train, valid = train_val_split(data)
        print(train[0], type(train[0]))
        with open(train_datapath, "w") as fw:
            for v in train:
                print(json.dumps(v, ensure_ascii=False), file=fw)
            logger.info("Train data write done")

        with open(dev_datapath, "w") as fw:
            for v in valid:
                print(json.dumps(v, ensure_ascii=False), file=fw)
        logger.info("Dev data write done")


def read_json_line_file(input_file):
    lines = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(json.loads(line.strip()))
        return lines


def train_val_split(X, valid_size=0.2, random_state=2022, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param valid_size
    :param random_state: 随机种子
    :param shuffle
    """

    test_size = int(len(X) * valid_size)

    if shuffle:
        random.seed(random_state)
        random.shuffle(X)

    valid = X[:test_size]
    train = X[test_size:]

    return train, valid
