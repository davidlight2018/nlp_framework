import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from processors.util import load_and_cache_examples, collate_fn
from utils.progressbar import ProgressBar
from utils.common import json_to_text


logger = logging.getLogger()


def predict(args, model, tokenizer, prefix=""):
    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)
    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="test")
    # Note that DistributedSampler samples randomly
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running prediction %s *****", prefix)
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", 1)
    results = []
    output_predict_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")

    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(test_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, "input_lens": batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            logits = outputs[0]
            tags = model.crf.decode(logits, inputs["attention_mask"])
            tags  = tags.squeeze(0).cpu().numpy().tolist()
        preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
        label_entities = get_entities(preds, args.id2label, args.markup)
        json_d = {}
        json_d["id"] = step
        json_d["tag_seq"] = " ".join([args.id2label[x] for x in preds])
        json_d["entities"] = label_entities
        results.append(json_d)
        pbar(step)
    logger.info("\n")
    with open(output_predict_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + "\n")
    if args.task_name == "cluener":
        output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
        test_text = []
        with open(os.path.join(args.data_dir, "test.json"), "r") as fr:
            for line in fr:
                test_text.append(json.loads(line))
        test_submit = []
        for x, y in zip(test_text, results):
            json_d = {"id": x["id"], "label": {}}
            entities = y["entities"]
            words = list(x["text"])
            if len(entities) != 0:
                for subject in entities:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    word = "".join(words[start:end + 1])
                    if tag in json_d["label"]:
                        if word in json_d["label"][tag]:
                            json_d["label"][tag][word].append([start, end])
                        else:
                            json_d["label"][tag][word] = [[start, end]]
                    else:
                        json_d["label"][tag] = {}
                        json_d["label"][tag][word] = [[start, end]]
            test_submit.append(json_d)
        json_to_text(output_submit_file, test_submit)
