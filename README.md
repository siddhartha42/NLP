# Information Retrieval / Search Engine

Python search engine built as part of an NLP course project. The project implements the core preprocessing and query-handling pieces of an information retrieval pipeline over the Cranfield dataset, with a modular structure that makes each NLP stage independently testable.

## What It Does

- Loads and evaluates queries over the Cranfield NLP dataset.
- Supports custom interactive queries through the command line.
- Implements sentence segmentation, tokenization, inflection reduction, stopword identification, stopword removal, spell-check, and WordNet-based utilities.
- Separates preprocessing stages into focused Python modules so retrieval behavior can be debugged and improved stage by stage.
- Produces output files for dataset evaluation and comparison.

## Repository Structure

```text
ME20B167_Assignment-1/
  main.py                    # SearchEngine class and evaluation/custom-query flow
  sentenceSegmentation.py    # Sentence boundary detection
  tokenization.py            # Query/document tokenization
  inflectionReduction.py     # Lemmatization/stemming-style normalization
  stopwordIdentify.py        # Stopword detection helpers
  stopwordRemoval.py         # Stopword filtering
  spellCheck.py              # Query spelling correction
  wordnet.py                 # WordNet-based query utilities
  util.py                    # Shared helpers
  cranfield/                 # Cranfield dataset files
  output/                    # Generated evaluation outputs
```

## Tech Stack

- Python
- NLP preprocessing
- Cranfield information retrieval dataset
- WordNet
- Command-line evaluation workflow

## Running The Project

From the assignment folder:

```bash
python main.py -dataset cranfield -out_folder output
```

For an interactive query:

```bash
python main.py -custom -dataset cranfield -out_folder output
```

The exact flags supported by the project are documented in `README.txt` inside the assignment folder.

## What This Demonstrates

This project is useful evidence for search/retrieval and backend-oriented software roles because it shows:

- Ability to build a multi-stage text-processing pipeline.
- Clean decomposition of NLP preprocessing logic.
- Evaluation-oriented development against a standard dataset.
- Comfort with Python, command-line workflows, and debugging data-dependent behavior.

## Resume Summary

Built a Python information retrieval/search engine over the Cranfield NLP dataset with modular preprocessing for sentence segmentation, tokenization, inflection reduction, stopword removal, spell-check, WordNet utilities, dataset evaluation, and custom queries.
