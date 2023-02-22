import os
import torch
import logging
import torch.distributed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from processors.util import collate_fn
from utils.progressbar import ProgressBar
from utils.common import seed_everything
from utils.plot import loss_acc_plot
from .evaluate import evaluate


logger = logging.getLogger()


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())
    linear_param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         "lr": args.learning_rate},

        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         "lr": args.crf_learning_rate},

        {"params": [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_learning_rate},
        {"params": [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,
         "lr": args.crf_learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to global_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss = 0.0
    best_eval_f1 = 0.0
    train_losses = []
    eval_losses = []
    train_accuracy = []
    eval_accuracy = []
    train_f1 = []
    eval_f1 = []

    history = {
        "train_loss": train_losses,
        "train_acc": train_accuracy,
        "train_f1": train_f1,
        "eval_loss": eval_losses,
        "eval_acc": eval_accuracy,
        "eval_f1": eval_f1,
    }
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproducibility (even between python 2 and 3)
    for i_epoch in range(int(args.num_train_epochs)):
        logger.info("  Doing training in epoch %d", i_epoch)
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3], 'input_lens': batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if global_step % len(train_dataloader) == 0:
                    # Log metrics
                    print("")
                    train_report, train_loss = evaluate(args, model, tokenizer, data_type="train")
                    eval_report, eval_loss = evaluate(args, model, tokenizer)
                    train_acc_score = train_report["micro avg"]["precision"]
                    train_f1_score = train_report["micro avg"]["f1-score"]
                    eval_acc_score = eval_report["micro avg"]["precision"]
                    eval_f1_score = eval_report["micro avg"]["f1-score"]
                    print("loss compare", train_loss, loss.item())
                    train_losses.append(loss.item())
                    train_accuracy.append(train_acc_score)
                    train_f1.append(train_f1_score)
                    eval_losses.append(eval_loss)
                    eval_accuracy.append(eval_acc_score)
                    eval_f1.append(eval_f1_score)

                    if eval_f1_score > best_eval_f1:
                        best_eval_f1 = eval_f1_score

                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-{}-{}".format(
                            global_step, round(eval_f1_score, 4)))
                        os.makedirs(output_dir, exist_ok=True)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)
                        tokenizer.save_vocabulary(output_dir)
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()

    loss_acc_plot(args, history)
    return global_step, tr_loss / global_step
