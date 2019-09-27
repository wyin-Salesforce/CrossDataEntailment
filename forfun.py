
# from transformers.tokenization_roberta import RobertaTokenizer
# from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaForSequenceClassification

# logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt = '%m/%d/%Y %H:%M:%S',
#                     level = logging.INFO)
# logger = logging.getLogger(__name__)

pretrain_model_dir = 'roberta-large-mnli' #'roberta-large' , 'roberta-large-mnli'
model = RobertaForSequenceClassification.from_pretrained(pretrain_model_dir, num_labels=2)
