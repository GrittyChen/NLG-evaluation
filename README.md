# NLG-evaluation
A toolkit for evaluation of natural language generation (NLG), including BLEU, ROGUR, METEOR, and CIDEr.

## Requirements

Make sure the following environment is installed correctly on your machine.

> python 2.7+
> numpy

If you want to use the METEOR metric, make sure the Java Runtime Environment is configured on your machine.

## Quickstart

You can run evaluation via:

```bash
python run_eval.py output_file references_file
```

Where the output\_file is the file that stored the results produced by your system and the references\_file is the file that stored the references. Note that you need to do tokenization before evaluation. More details refer to the code.
