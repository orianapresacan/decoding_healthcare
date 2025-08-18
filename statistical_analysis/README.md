# Evaluation Results (Section 4 of the Paper)

Results are provided in separate CSV files for each benchmark:  
**translation**, **summarization**, **QA**, **dialogue**, and **image captioning**.  

Each CSV file contains:
- Model scores for different evaluation metrics (ROUGE, BLEU, BERTScore, MAUVE)
- Inference times

### Code

- `decoding_analysis.py` includes the tests and results reported in  
  **Subsection 4.1: Decoding Strategies**.
- `decoding_analysis_inference_time.py` includes the inference time results reported in **Subsection 4.1: Decoding Strategies**.
- `model_analysis_tests.py` includes the experiments and results reported in  
  **Subsection 4.2: Models**.
- `model_analysis_plot.py` includes code for the plots in Figure 2 from  
  **Subsection 4.2: Models**.
- `metric_analysis.py` includes the experiments and results reported in  
  **Subsection 4.3: Metrics**.
