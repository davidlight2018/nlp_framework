import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from metrics.entity_eval import SingleEval
from processors.util import load_and_cache_examples, collate_fn
from utils.progressbar import ProgressBar


logger = logging.getLogger()


def evaluate(args, model, tokenizer, prefix="", data_type="dev"):
    metric = SingleEval()
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type=data_type)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)
    # Eval!
    logger.info("***** Running %s evaluation %s *****", data_type, prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")
    if isinstance(model, nn.DataParallel):
        model = model.module
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            # with CRF
            tags = model.crf.decode(logits, inputs["attention_mask"])

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        out_label_ids = inputs["labels"].cpu().numpy().tolist()
        input_lens = inputs["input_lens"].cpu().numpy().tolist()
        # with CRF
        tags = tags.squeeze(0).cpu().numpy().tolist()

        # without CRF
        # import numpy as np
        # logits = logits.cpu().numpy()
        # tags = np.argmax(logits, axis=2)

        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif j == input_lens[i] - 1:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(args.id2label[out_label_ids[i][j]])
                    temp_2.append(args.id2label[tags[i][j]])
        pbar(step)

    print()
    logger.info("evaluate_loss = %s", eval_loss / nb_eval_steps)
    print()
    report = metric.report()
    return report, eval_loss / nb_eval_steps
