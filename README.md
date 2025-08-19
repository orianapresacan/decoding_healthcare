# Decoding Strategies for Medical Text Generation

## [Paper]() 


## Overview
We investigated the impact of various LLM decoding strategies on accuracy, inference time, and sensitivity. We evaluated both medical and general-purpose LLMs of different sizes across five popular medical tasks: translation, summarization, question answering, dialogue, and image captioning.

## Repository Structure

- `hyperparameter_inference.py` - this script was used to generate text using different decoding strategies.
    ```bash
    # greedy
    python hyperparameter_inference.py --task translation --model Qwen/Qwen3-1.7B --decoding_strategy greedy
    ```
- `evaluation_csv.py` - this script was used for evaluating the generated text using different metrics such as BLEU, ROUGE, BERTScore, and MAUVE.
    ```bash
    python evaluation_csv.py --task summarization --pred_path data/summarization/outputs --gt_file data/summarization/data.jsonl
    ```
- `statistical_analysis/` - this folder contains csv files with the complete results, as well as the code for the statistical analysis reported in Section 4 of the paper.

## Data preparation
Data should be organized as follows: the main folder data contains a separate subfolder for each benchmark, and each subfolder contains a `data.jsonl` file.

``` 
data/
├── translation/
│   ├── outputs/
│   └── data.jsonl
...
``` 

The `data.jsonl` file should have the following structure:
``` 
{"id": X, "instruction": "You are a knowledgeable and helpful medical assistant.", "prompt": "Translate the text from German to English: {german_text}", "tgt": "{ground_truth_english_text}"}
``` 

The links to the datasets we used can be found at:
- [Translation](https://ufal.mff.cuni.cz/ufal_medical_corpus)
- [Summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization/tree/main/section)
- [QA](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- [Dialogue]()
- [Image Captioning](https://zenodo.org/records/8333645)

## Citation:
```

```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## For more details:
Please contact: orianapresacan@gmail.com
