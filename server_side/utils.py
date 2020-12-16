from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, tqdm_notebook
import gluonnlp as nlp
import copy


class BERTDatasetForTest(Dataset):
    def __init__(self, dataset, bert_tokenizer, max_len,
                pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        
        # texts = dataset["sentence"].tolist()
        texts = dataset
        # labels = dataset["Emotion"].tolist()

        self.sentences = [transform([text]) for text in texts]
        # self.labels = [np.int32(emotion2label[label]) for label in labels]

    def __getitem__(self, i):
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))

# classifier structure
class BERTClassifier(nn.Module):
    def __init__(self,
                bert,
                hidden_size = 768,
                num_classes = 5,
                dr_rate = None,
                params = None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
def get_label_probability(result, label2emotion):
  # result : tensor
    ret_dict = {emotion : 0 for emotion in list(label2emotion.values())}

    result = result.tolist()
    num_results = len(result)
    key_list = list(Counter(result).keys())
    emotion_list = list(label2emotion[key] for key in key_list)

    counted_value_list = list(Counter(result).values())
    prob_value_list = list(counted_value/num_results for counted_value in counted_value_list)

    for emotion, prob_value in zip(emotion_list, prob_value_list):
        ret_dict[emotion] = prob_value
    # print(ret_dict)
#   ret_dict = {emotion_list[i] : prob_value_list[i] for i in range(len(key_list))}

    return ret_dict


