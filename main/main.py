import os
import env
import time
import glob
import logging

import torch
import torch.distributed
import torch.nn as nn
from transformers import BertConfig, AlbertConfig

from models import BertCrfForNer, CNerTokenizer
from task.ner import train, evaluate, predict
from processors.cluener_processor import CluenerProcessor
from processors.util import load_and_cache_examples
from utils.args import get_argparse
from utils.logger import init_logger
from utils.common import seed_everything
from utils.loader import flexible_loader


logger = logging.getLogger()


MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    # 'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}
WEIGHTS_NAME = "pytorch_model.bin"


def main():
    args = get_argparse().parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_dir = args.output_dir + "{}".format(args.model_type)
    os.makedirs(args.output_dir, exist_ok=True)
    args.image_dir = args.image_dir + "{}".format(args.model_type)
    os.makedirs(args.image_dir, exist_ok=True)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # init_logger(log_file=args.output_dir + f'/{args.model_type}-{args.task_name}-{time_}.log')
    init_logger()
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    args.device = torch.device(args.device)
    logger.warning("device: %s, 16-bits training: %s", args.device, args.fp16)
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()

    processor = flexible_loader(f"processors.{args.task_name}_processor")()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels, cache_dir=args.cache_dir if args.cache_dir else None, )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None, )
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config, cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_vocabulary(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Evaluation
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            evaluate(args, model, tokenizer, prefix=prefix)

    if args.do_predict:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.predict_checkpoints > 0:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            checkpoints = [x for x in checkpoints if x.split('-')[-1] == str(args.predict_checkpoints)]
        logger.info("Predict the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            model = model_class.from_pretrained(checkpoint, config=config)
            model.to(args.device)
            predict(args, model, tokenizer, prefix=prefix)


if __name__ == "__main__":
    main()
