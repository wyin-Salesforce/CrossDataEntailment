from pytorch_transformers.modeling_roberta import RobertaForSequenceClassification

pretrain_model_dir = 'roberta-large-mnli' #'roberta-large' , 'roberta-large-mnli'
model = RobertaForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=2)
