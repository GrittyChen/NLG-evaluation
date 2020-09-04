# NLG-evaluation
A toolkit for evaluation of natural language generation (NLG), including BLEU, ROUGE, METEOR, and CIDEr.

## Requirements

Make sure the following environment is installed correctly on your machine.

> python 2.7+

> numpy

If you want to use the METEOR metric, make sure the Java Runtime Environment is configured on your machine.

## Usage

### Quickstart

You can run evaluation via:

```bash
python run_eval.py --hypos output_file --refs reference_file
```

where `output_file` is the file that stored the results produced by your system, and `reference_file` is the file that stored the references. Note that you need to do tokenization before evaluation.

### Multiple reference

Evaluation with multiple references is supported. The command is:

```bash
python run_eval.py --hypos output_file --refs ref_1 ref_2 ... ref_n
```
`ref_1 ref_2 ... ref_n` are `n` reference files.

### Choose metrics

You can choose any metrics you want. By default, all metrics (BLEU, ROUGE, METEOR, and CIDEr) are enabled. If you do not need a metric (e.g., BLEU), the command to disable the metric is:

```bash
python run_eval.py --hypos output_file --refs reference_file [-nB | --no_BLEU]
```

Similarly, you can turn off other metrics:

```bash
[-nR | --no_ROUGE]    # for ROUGE
[-nM | --no_METEOR]   # for METEOR
[-nC | --no_CIDEr]    # for CIDEr
```

### Choose ngrams

You can also change the `n`-gram to obtain the BLEU-`n` scores:

```bash
python run_eval.py --hypos output_file --refs reference_file [-n | --ngram] 3
```

Then you will get BLEU-`n` (`n <= 3`) scores. The default `n` is 4.

### Lowercase

If you want to conduct evaluations in lowercase mode, the command is:

```bash
python run_eval.py --hypos output_file --refs reference_file [-lc | --lowercase]
```