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


statistics = {
    "gender" : {
        "male" : 0.715879002,
        "female" : 0.284120998
    },
    "age" : {
        10 : 0.021595768,
        20 : 0.094644539,
        30 : 0.138705703,
        40 : 0.187549822,
        50 : 0.205594608,
        60 : 0.147474455,
        70 : 0.117399812,
        80 : 0.08645554,
    },
    "job" : {
        "전문가" : 0,
        "기술공 및 준전문가" : 0,
        "농업,임업 및 어업 숙련종사자" : 0.025847785,
        "장치,기계조작 및 조립 종사자" : 0.029161604,
        "기능원 및 관련 기능 종사자" : 0.033359107,
        "관리자" : 0.041422733,
        "사무 종사자" : 0.062631172,
        "전문가 및 관련 종사자" : 0.067380979,
        "기타" : 0.07301447,
        "단순노무 종사자" : 0.076217828,
        "서비스 및 판매 종사자" : 0.134541036,
        "무직, 가사, 학생" : 0.456423285,
    },
    "regidences" : {
        "세종특별자치시" : 0.046656946,
        "서울특별시" : 0.046865236,
        "광주광역시" : 0.049781296,
        "경기도" : 0.052905645,
        "전라남도" : 0.052905645,
        "인천광역시" : 0.053947094,
        "경상남도" : 0.058321183,
        "울산광역시" : 0.058737763,
        "대구광역시" : 0.059779213,
        "대전광역시" : 0.059779213,
        "경상북도" : 0.061237242,
        "부산광역시" : 0.062695272,
        "전라북도" : 0.062903562,
        "충청북도" : 0.064778171,
        "제주특별자치도" : 0.066027911,
        "강원도" : 0.06936055,
        "충청남도" : 0.073318059,
    },
}
