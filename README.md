# AdvMedSumm: Contrastive Learning Framework for Faithful Medical Summarization

This is the official repository of the paper **AdvMedSumm: Contrastive Learning Framework for Faithful Medical Summarization**

## Files Descriptions
1. **make_mimic_data.py** - Create RRS dataset from the MIMIC data zip file. Refer to [original-repo] https://github.com/abachaa/MEDIQA2021/tree/main/Task3 for complete details.
2. **bart_hftrainer_medical.py** - Contains the functions to train BART with and without AdvMedSumm perturber on the datasets
3. **eval_bart_medical_adv.py** - Containts functions to evaluate model trained using AdvMedSumm approach on the validation and other test datasets
4. **eval_bart_medical_base.py** - Contains functions to train the baseline model on the validation and other test datasets
   
### Environment Details

- We use Python 3.10 with CUDA 11.8 on NVIDIA L4 GPU for all the experiments. Other major depdencies are:
    ``` 
    torch==2.1.0
    transformers==4.36.0
    trl==0.7.4
    peft==0.7.1.dev0
    fairseq=0.12.2
    sacremoses==0.1.1
    fastBPE
    ```
    
## HQS dataset 

- Download the data from the HQS-data folder which contains the training and the validation dataset. You can run and evaluate the models on HQS dataset using following commands
  Training
    with AdvMedSumm :  ``` python bart_hftrainer_medical.py --dataset=hqs --adv_training=1 --exp=hqs_0.01 ```
    Baseline : ``` python bart_hftrainer_medical.py --dataset=hqs --adv_training=0 --exp=hqs_base ```
  Evaluation:
    with AdvMedSumm :  ``` python eval_bart_medical_adv.py --dataset=hqs --model_dir=models_hqs_0.01 ``` 
    Baseline : ``` python eval_bart_medical_base.py --dataset=hqs --model_dir=models_hqs_base ``` 
- All the arguments are explained in the trainer python file

# RRS dataset 

- Download and preprocess the data from [original repo]([https://github.com/fladhak/pretraining_biases](https://github.com/abachaa/MEDIQA2021/tree/main/Task3)). You can run and evaluate the models on HQS dataset using following commands
- Training
  with AdvMedSumm :  ```  python bart_hftrainer_medical.py --dataset=rrs --adv_training=1 --exp=rrs_0.01 ``` 
    Baseline : ```  python bart_hftrainer_medical.py --dataset=rrs --adv_training=0 --exp=rrs_base ``` 
  Evaluation:
    with AdvMedSumm :   ``` python eval_bart_medical_adv.py --dataset=rrs --model_dir=models_rrs_0.01 ``` 
    Baseline :  ``` python eval_bart_medical_base.py --dataset=rrs --model_dir=models_rrs_base ``` 
- All the arguments are explained in the trainer python file

