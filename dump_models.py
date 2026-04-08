import sys
from models.model_1 import VQAModel1_ScratchNoAtt
from models.model_2 import VQAModel2_PretrainedNoAtt
from models.model_3 import VQAModel3_ScratchAtt
from models.model_4 import VQAModel4_PretrainedAtt
from models.model_5 import VQAModel5_PretrainedEndToEndNoAtt
from models.model_6 import VQAModel6_PretrainedEndToEndAtt

vocab_size = 2589

models = [
    (1, VQAModel1_ScratchNoAtt(vocab_size)),
    (2, VQAModel2_PretrainedNoAtt(vocab_size)),
    (3, VQAModel3_ScratchAtt(vocab_size)),
    (4, VQAModel4_PretrainedAtt(vocab_size)),
    (5, VQAModel5_PretrainedEndToEndNoAtt(vocab_size)),
    (6, VQAModel6_PretrainedEndToEndAtt(vocab_size)),
]

with open('all_models.txt', 'w', encoding='utf-8') as f:
    for mid, model in models:
        f.write(f"=== KẾT QUẢ IN MODEL {mid} ===\n")
        f.write(str(model))
        f.write("\n\n")
