import argparse

from data_loader import load_and_cache_examples
from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset, dev_dataset=dev_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        try:
            trainer.load_model()
        except Exception as e:
            print(f"{e}")
            print(f"Training...")
            trainer.train()
            trainer.load_model()
        # print(trainer.config)
        results=trainer.evaluate("test",f1_ignore_direction=args.f1_ignore_direction)

        # print(results)
        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="semeval", type=str, help="The name of the task to train")
    parser.add_argument(
        "--data_dir",
        default="./data",
        type=str,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument(
        "--eval_dir",
        default="./eval",
        type=str,
        help="result directory",
    )
    parser.add_argument(
        "--eval_script",
        default="./eval/semeval2010_task8_scorer-v1.2.pl",
        type=str,
        help="Evaluation script",
    )
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--dev_file", default="test.tsv", type=str, help="Dev file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Model Name or Path",
    )

    parser.add_argument("--seed", type=int, default=77, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument(
        "--max_seq_len",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
    )

    parser.add_argument("--logging_steps", type=int, default=250, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--add_sep_token",
        action="store_true",
        help="Add [SEP] token at the end of the sentence",
    )

    """my arguments"""
    parser.add_argument("--target_label", type=str, help="Label we show the F1 for")
    parser.add_argument("--f1_ignore_direction", action="store_true", help="Whether to ignore dimensions when calculate F1 score.")


    args = parser.parse_args()

    main(args)
