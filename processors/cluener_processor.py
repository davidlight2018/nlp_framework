import os
import json
from .base_processor import DataProcessor, InputExample


class CluenerProcessor(DataProcessor):
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

    def data_preprocess(self, data_dir):
        pass

    def get_labels(self):
        """See base class."""
        return ["X", "B-address", "B-book", "B-company", "B-game", "B-government", "B-movie", "B-name",
                "B-organization", "B-position", "B-scene", "I-address",
                "I-book", "I-company", "I-game", "I-government", "I-movie", "I-name",
                "I-organization", "I-position", "I-scene",
                "S-address", "S-book", "S-company", "S-game", "S-government", "S-movie",
                "S-name", "S-organization", "S-position",
                "S-scene", "O", "[START]", "[END]"]

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
                label_entities = line.get("label", None)
                words = list(text)
                labels = ["O"] * len(words)
                if label_entities is not None:
                    for key, value in label_entities.items():
                        for sub_name, sub_index in value.items():
                            for start_index, end_index in sub_index:
                                assert "".join(words[start_index:end_index+1]) == sub_name
                                if start_index == end_index:
                                    labels[start_index] = "S-"+key
                                else:
                                    labels[start_index] = "B-"+key
                                    labels[start_index+1:end_index+1] = ["I-"+key]*(len(sub_name)-1)
                lines.append({"words": words, "labels": labels})
        return lines
