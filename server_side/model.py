import os
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
from utils import BERTClassifier, BERTDatasetForTest, get_label_probability, statistics
from scipy.stats import hmean
# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# 경로 설정
BASE_DIR = "../" # 로컬 경로로 바꿔주어야 로컬에서 작동됨
CODE_DIR = os.path.join(BASE_DIR, "code")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_BINARY_DIR = os.path.join(MODEL_DIR, "binary_label_model.pt")
MODEL_MULTI_DIR = os.path.join(MODEL_DIR, "multi_label_model.pt")

# device 설정
device = torch.device("cpu")

# model load
bertmodel, vocab = get_pytorch_kobert_model()
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

## Setting parameters
max_len = 64 ############################## 문항 개수가 64개가 넘어가지 않으면 한번에 처리함
batch_size = 64
""" Training configuration
warmup_ratio = 0.1
num_epochs = 5 
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
"""

# label tarnsformer
binary_emotion2label = {"joy" : 1, "sadness" : 0}
binary_label2emotion = { 1 : "joy", 0 : "sadness"}

multi_emotion2label = {"neutral" : 0, "sadness" : 1, "fear" : 2, "anger" : 3, "joy" : 4}
multi_label2emotion = { 0 : "neutral", 1 : "sadness",  2 : "fear",  3 : "anger",  4 : "joy"}

# model load
binary_model = BERTClassifier(bertmodel, num_classes=2, dr_rate=0.5).to(device)
binary_model.load_state_dict(torch.load(MODEL_BINARY_DIR, map_location=torch.device('cpu')))

multi_model = BERTClassifier(bertmodel, num_classes=5, dr_rate=0.5).to(device)
multi_model.load_state_dict(torch.load(MODEL_MULTI_DIR, map_location=torch.device('cpu')))






# 실제로 매번 실행될 함수
def Predict(sentence, info):
    # sentence : 10개의 문장 list
    sentence_dataset = BERTDatasetForTest(sentence, tok, max_len, True, False)
    sentence_dataloader = torch.utils.data.DataLoader(sentence_dataset, batch_size=batch_size, num_workers=1)

    # # Binary Model
    # binary_model.eval()
    # for batch_id, (token_ids, valid_length, segment_ids) in enumerate(sentence_dataloader):
    #     token_ids = token_ids.long().to(device)
    #     segment_ids = segment_ids.long().to(device)
    #     valid_length= valid_length
    #     #label = label.long().to(device)
    #     binary_out = binary_model(token_ids, valid_length, segment_ids)



    # Multi Model
    multi_model.eval()
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(sentence_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        #label = label.long().to(device)
        multi_out = multi_model(token_ids, valid_length, segment_ids)


    # Output Check
    # binary_max_vals, binary_max_indices = torch.max(binary_out, 1)
    multi_max_vals, multi_max_indices = torch.max(multi_out, 1)

    # binary_predicted_emotion = list(binary_label2emotion[label] for label in binary_max_indices.tolist())
    multi_predicted_emotion = list(multi_label2emotion[label] for label in multi_max_indices.tolist())

    # print("binary result")
    # for text, emotion in zip(target, binary_predicted_emotion):
    #     print(f"{emotion} : {text}")

    print("multi result")
    for text, emotion in zip(target, multi_predicted_emotion):
        print(f"{emotion} : {text}")

    # 결과 확인
    # binary_prob_dict = get_label_probability(binary_max_indices, binary_label2emotion)
    multi_prob_dict = get_label_probability(multi_max_indices, multi_label2emotion)
    neg_prob = multi_prob_dict["sadness"]+multi_prob_dict["fear"]+multi_prob_dict["anger"]

    # 성별p * (거주지 p + 직업 + 연령) = 통계치 기반 확률 값
    gender_info = info["gender"]
    age_info = int(info["age"])
    regidences_info = info["regidences"]
    job_info = info["job"]

    print(statistics["gender"][gender_info])
    print(statistics["age"][age_info])
    print(statistics["regidences"][regidences_info])
    print(statistics["job"][job_info])

    suicide_prob = statistics["gender"][gender_info]*(statistics["age"][age_info] + statistics["regidences"][regidences_info] + statistics["job"][job_info])








    # neg_values = [binary_prob_dict["sadness"], multi_prob_dict["sadness"]+multi_prob_dict["fear"]+multi_prob_dict["anger"]]
    
    depressed_prob_temp = [8*neg_prob, 2*suicide_prob]
    print(f"neg_values : {depressed_prob_temp}")
    # depressed_prob = hmean(depressed_prob_temp)
    depressed_prob = sum(depressed_prob_temp)/(len(depressed_prob_temp)*5)
    result = depressed_prob

    print(f"우울증 지수는 : {result} 입니다.")
    
    # 결과 리턴 (형태는 기획에 맞춰 조정)
    # print(binary_prob_dict.values())
    # binary_result = hmean(list(binary_prob_dict.values())) # 각 model의 결과를 조화평균함
    # multi_result = hmean(list(multi_prob_dict.values()))




    # return {'result': result}


if __name__ == "__main__":
    # target=[
    #     "언젠가부터 남과 끊임없이 비교하게 되었어요. ",
    #     "제가 이뤄놓은 것이 일절 없기 때문일지 모릅니다.",
    #     "큰 시험에 모조리 실패를 하고 돌아와서 실패를 무마하려고 또 공부를 시작했어요. 쉴 틈이 없더라고요. 제 나이에 다들 기겁을 하고 망한 인생이라 기구하다 말하더라고요.",
    #     "저는 제가 점점 싫어집니다.",
    #     "진짜 열심히 산다고 주변인들이 말해줄 때마다 저는 결과가 엉망진창인 저를 떠올리고 다시 죽고 싶어질 뿐입니다.",
    #     "대기업에 들어간 사촌들과 제 몫을 하는 또래들을 생각하게 되고 가슴에 피멍이 드는 듯 아픕니다.",
    #     "그러다가 거울을 보면서 못생겼다, 코도 낮고 얼굴형도 괴상해. 넌 정말 뭐하나 예쁜 구석이 없구나.를 반복합니다.",
    #     "근데 그게 경제적으로 사회적으로 무가치한 자신 앞에서 아무런 힘이 없더라고요. 그런 사람이 이치를 깨닫는다한들 곧 사라지고 또 괴로울 뿐이에요.",
    #     "그런데 또 다른 괴로움과 자책을 시작하게 되고, 저는 무슨 일을 해도 우울과 불안에 휩싸여서 언제쯤이면 죽게 될까. 과연 나는 행복할 수 있는 사람인가. 잠시 쉬는 시간이 생기면 그런 생각을 합니다.",
    #     "생을 야망이나 하고 싶은 것들로만 채워 버텨왔는데 모든 의욕이나 야망이 사라졌어요.",
    #     "작은 행복을 느끼며 살아가란 말이 체념처럼, 굴복하는 것처럼 절 아프게 합니다.",
    #     "누군가에게는 치료를 받아야하는 마땅한 근거와 진실이 있지만 나조차도 살아가는게 매우 힘들다는 과정을 보았습니다. 생각보다 성격도 변했습니다",
    #     "나라에서 빚지고 사니 힘들어지더군요 직장도 없고 아무것도 없는 모습에 말이죠 솔직한 말을 하자면 깨끗하게 되지를 못한다는 점입니다",
    #     "이제라도 나 자신을 바뀌기를 원했는데 죽음탓만 하는 후회스럽습니다",
    #     "처음에 한두번쯤은 그러려니 했는데 자꾸 그런말을 하고 반복적으로 지속적으로 고양이키우면 수급취소다 듣다보니 질리고 숨막히고 치가 떨려요",
    #     "원래 삶에 의욕이 별로 없는 편이었지만 그래도 살다보면 하고 싶은게 생기거니, 또 죽자 하니 가족들이 슬퍼할까봐 괜찮은 척 하면서 살았습니다. 물론 죽는게 막연히 겁나기도 했고요.",
    #     "솔직히 이제 딱히 살아서 뭐가 좋은지 모르겠습니다. 저보다 열악한 상황에서도 살기 위해 노력하는 분들께는 죄송하지만, 전 진짜 죽는게 더 나을것 같다는 생각이 듭니다.",
    #     "죽는건 무섭지만 질소 같은거 쓰면 딱히 아프진 않겠죠",
    #     "매일 죽고싶다 생각만했습니다. 그런데 이젠 정말 죽어야겠습니다.",
    #     "네....... 다 제잘못인거 알고있습니다. 그런데 이제는 지쳐갑니다. 매일마다 받는 독촉전화에... 문자에......... 진짜...........하........",
    #     "전... 죽고나면 그만인데.............. 저때문에 벌어진일들........ 조금이나마 없어질까요?",
    #     "나 왜 살지, 뭘 위해 살지, 아무것도 하기 싫어요.",
    #     "그냥 죽고 싶다 라는 생각이 요새 많이 들어요.",
    #     "그만 살아도 될 거 같은 생각이 자주 들어요.",
    #     "일에도 지치고 찌들리고 몸과 정신 마음 다 쉬고 싶은데 그렇지 못하니까 또 쉬지 않고 달려야 되니까 다 그만두고 싶어요.",
    #     "매일 매일 독촉에 시달리고 거짓말하며 썩어가던 제 마음이 내려놓고 마지막을 생각하니 오히려 편하네요.",
    #     "더이상은 무리가 있어 이제는 떠나렵니다.",
    #     "제가 도박을 끊지 못한 이유는 명확합니다.",
    #     "빌린돈을 갚아야하는 주기가 짧다보니, 빌려서 매꾸는게 아닌 빌려서 도박을 해서 큰돈을 따야지 매꿀수있는 현실. 크게크게 먹는 사례가 몇번씩 있다보니 매꿔지고 또 생활이 괜찮아지고 또 힘들어지고를 반복.",
    # ]
    # target=[
    #     "언젠가부터 남과 끊임없이 비교하게 되었어요. ",
    #     "제가 이뤄놓은 것이 일절 없기 때문일지 모릅니다."
    #     ]
    # target = [
    #     "기분이 엄청 좋을 것 같다. 당장 먹을 생각부터 한다.",
    #     "정말 고맙다... 나를 이렇게 진심 어리게 안아주고 위로해 주면 힘이 날 것 같다",
    #     "진짜 반갑다… 난 완전 잘 지냈는데 친구도 잘 지냈나보네 얼굴 좋아보인다~",
    #     "정신이 없기도 했고… 생각보다 해야 할 일을 많이 못한 것 같다",
    #     "내가 사랑하는 사람들과 하루 하루 행복하고 마음 편안하게 살고 있으면 좋겠다!",
    #     "어릴 때부터 반항심이 많았군. 뭐 이리 세상에 불만이 많았을까… 지금의 나는 예전의 나보다 낫나?",
    #     "조금 부담스러울 것 같지만 받아들일 것이다. 그래도 내가 책임감 있게 잘 해낼 자신이 있어서.",
    #     "별 생각 없이 잠에 들었다. ",
    #     "나처럼 다들 행복해 보이네~ 메리 크리스마스!",
    #     "반갑다. 오랜만에 얼굴 보니 좋다. 여전하다.",
    # ]

    target = [
        "당장은 먹고 싶지 않은데… 무얼 맛있게 먹을 수 있는 상황이 아니에요…",
        "위로 같은 것으로 제 상황이나 기분이 나아지지 않을 것 같습니다…",
        "너는 잘 지내는구나? 나는 요즘 매일 매일이 괴롭고 힘들었어… ",
        "하루 하루가 힘들었는데… 매일 죽고싶다 생각만했습니다. 그런데 이젠 정말 죽어야겠다는 생각이 드네.",
        "그만 살아도 될 거 같은 생각이 자주 들어요… 제게 10년 후라는 미래가 있을까요?",
        "어릴 적 나는 지금의 나를 보고 무슨 생각을 할까… 한심하게 보지는 않을까",
        "거절할 것 같습니다. 저는 결과가 엉망진창인 저를 떠올리고 다시 죽고 싶어질 뿐입니다.",
        "한참동안 우울과 불안에 휩싸여서 언제쯤이면 죽게 될까. 과연 나는 행복할 수 있는 사람인가. 잠시 쉬는 시간이 생기면 그런 생각을 하다가 잠이 들었다.",
        "좌절감, 박탈감을 느낄 것 같다.",
        "쟤는 언제 취직하려나. 아직도 자리 못 잡았나?",
    ]
    info={
        "gender" : "male",
        "age" : 10,
        "regidences" : "서울특별시",
        "job" : "무직, 가사, 학생"
    }
    Predict(target, info)


