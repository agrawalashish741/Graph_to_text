# Graph_to_text  
# 🧠 AMR-to-Text Generation Pipeline (BART & T5) + BFS Linearization + Adapter_Tuning + UD_parsing

This repository provides tools to:
- Fine-tune **BART** and **T5** models on AMR-to-text data
- Convert AMR graphs into **BFS linearized sequences**
- Evaluate predictions using BLEU and ROUGE
- Perform **hyperparameter tuning**
- Train on BFS-encoded AMR with **custom special tokens**
- Train the model using adapter without training the entire model.
- Aslo convert the linearised sentence into UD_Parsed sentences.
---


## 🔧 Step 1: Set Up Virtual Environment

```bash
# Create and activate a virtual environment
python3 -m venv amr_env
source amr_env/bin/activate   # For Linux/macOS

pip install torch transformers penman nltk rouge-score pandas tqdm wandb
python -c "import nltk; nltk.download('punkt')"

#Bart train
python bart_train.py --train_file "path/to/train.json" --validation_file "path/to/val.json" --output_dir "path/to/save_model"

#T5 train
python T5_train.py train_file "./data/amr_train.json" validation_file "./data/amr_dev.json" output_dir "./models/t5_finetuned"

#Generate and Evaluation
## For T5 
python gen_and_eval.py --test_file "./data/amr_test.json" --output_dir "./models/t5_finetuned" --output_file "./results/generated_t5.json" --model_type "t5"
## For bart
python generate_and_evaluate.py --test_file "./data/amr_test.json" --output_dir "./models/bart_finetuned" --output_file "./results/generated_bart.json" --model_type "bart"

#Hyperparameter tuning
python hyperparameter_tuning.py --model_type bart --train_file ./data/amr_train.json --val_file ./data/amr_dev.json --test_file ./data/amr_test.json --output_dir ./models/bart_only
python hyperparameter_tuning.py --model_type t5 --train_file ./data/amr_train.json --val_file ./data/amr_dev.json --test_file ./data/amr_test.json --output_dir ./models/t5_only
python hyperparameter_tuning.py --train_file ./data/amr_train.json --val_file ./data/amr_dev.json --test_file ./data/amr_test.json --output_dir ./models/both_models



## BFS Conversion
python bfs_conversion.py amr_input.json
python bfs_train.py --train outputBFS_small_train.json --val outputBFS_small_dev.json --test outputBFS_small_test.json


# to run adapter 
python converted_amr_to_text.py --train ./data/amr_train.json --test ./data/amr_test.json --val ./data/amr_dev.json --output_dir ./output_texts
python Preproc.py --input_folder ./output_texts --output_dir ./preprocessed_data
python adapter_train.py --input_folder ./preprocessed_data --output_folder ./adapter_results --epochs 3

# to run UD Parser
python ud_parser.py sample.txt output.conllu
