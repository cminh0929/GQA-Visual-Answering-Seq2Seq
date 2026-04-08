import torch
from models.model_1 import VQAModel1_ScratchNoAtt
from models.model_2 import VQAModel2_PretrainedNoAtt
from models.model_3 import VQAModel3_ScratchAtt
from models.model_4 import VQAModel4_PretrainedAtt
from models.model_5 import VQAModel5_PretrainedEndToEndNoAtt
from models.model_6 import VQAModel6_PretrainedEndToEndAtt
from models.model_7 import VQAModel7_Transformer

# ---------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------
MODEL_REGISTRY = {
    1: VQAModel1_ScratchNoAtt,
    2: VQAModel2_PretrainedNoAtt,
    3: VQAModel3_ScratchAtt,
    4: VQAModel4_PretrainedAtt,
    5: VQAModel5_PretrainedEndToEndNoAtt,
    6: VQAModel6_PretrainedEndToEndAtt,
    7: VQAModel7_Transformer,
}

MODEL_INFO = {
    1: {
        "name": "Model 1: Scratch + No Attention",
        "description": "Baseline CNN scratch encoder with LSTM decoder",
        "has_attention": False,
        "strategy": "end-to-end"
    },
    2: {
        "name": "Model 2: Pretrained + No Attention",
        "description": "Fixed ResNet-50 features with LSTM decoder",
        "has_attention": False,
        "strategy": "pre-extracted"
    },
    3: {
        "name": "Model 3: Scratch + Attention",
        "description": "Scratch CNN with Spatial Attention mechanism",
        "has_attention": True,
        "strategy": "end-to-end"
    },
    4: {
        "name": "Model 4: Pretrained + Attention",
        "description": "Fixed ResNet-50 spatial features with Attention decoder",
        "has_attention": True,
        "strategy": "pre-extracted"
    },
    5: {
        "name": "Model 5: Pretrained E2E + No Attention",
        "description": "End-to-end trainable ResNet-50 (unfrozen) without attention",
        "has_attention": False,
        "strategy": "end-to-end"
    },
    6: {
        "name": "Model 6: Pretrained E2E + Attention",
        "description": "End-to-end trainable ResNet-50 with Spatial Attention",
        "has_attention": True,
        "strategy": "end-to-end"
    },
    7: {
        "name": "Model 7: Transformer-based VQA",
        "description": "Encoder-Decoder Transformer attending to Image + Question",
        "has_attention": True, # Transformer has self/cross attention
        "strategy": "end-to-end"
    }
}

# ---------------------------------------------------------
# FACTORY FUNCTIONS
# ---------------------------------------------------------

def list_models():
    """Return list of available model IDs."""
    return sorted(list(MODEL_REGISTRY.keys()))

def get_model_info(model_id):
    """Get metadata for a specific model."""
    if model_id not in MODEL_INFO:
        raise ValueError(f"Invalid model ID: {model_id}. Available: {list_models()}")
    return MODEL_INFO[model_id]

def get_model(model_id, vocab_size, device="cpu", **kwargs):
    """Instantiate a model by ID."""
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model ID: {model_id}. Available: {list_models()}")
    
    model_class = MODEL_REGISTRY[model_id]
    model = model_class(vocab_size, **kwargs)
    return model.to(device)
