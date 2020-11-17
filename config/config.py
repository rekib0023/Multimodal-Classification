import transformers

DEVICE = "cuda"
MAX_LEN = 80
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = 'bert-base-uncased'

MODEL_PATH = "model"
INPUT_PATH = 'data'
IMG_PATH = '/img'
TRAINING_FILE = "/train.jsonl"
VALID_FILE = "/dev.jsonl"
TEST_FILE = "/test.jsonl"

TEXT = 'text'
LABEL = 'label'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)