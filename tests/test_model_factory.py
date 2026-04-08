import pytest
import torch.nn as nn
from models.factory import get_model, get_model_info, list_models

def test_list_models():
    """Test that we can list all 6 models."""
    models = list_models()
    assert len(models) == 7
    assert all(isinstance(m, int) for m in models)
    assert set(models) == {1, 2, 3, 4, 5, 6, 7}

def test_get_model_info():
    """Test retrieving model metadata."""
    info = get_model_info(1)
    assert info["name"] == "Model 1: Scratch + No Attention"
    assert info["has_attention"] is False
    
    info_6 = get_model_info(6)
    assert "Attention" in info_6["name"]
    assert info_6["has_attention"] is True

def test_get_model_instantiation():
    """Test that factory returns actual instantiated nn.Module."""
    vocab_size = 100
    model = get_model(1, vocab_size)
    assert isinstance(model, nn.Module)
    
    model_5 = get_model(5, vocab_size)
    assert isinstance(model_5, nn.Module)

def test_invalid_model_id():
    """Test handling of invalid IDs."""
    with pytest.raises(ValueError, match="Invalid model ID"):
        get_model(99, 100)
    
    with pytest.raises(ValueError, match="Invalid model ID"):
        get_model_info(0)

def test_model_7_shapes():
    """RED: Test Model 7 forward and generate shapes."""
    import torch
    vocab_size = 50
    batch_size = 2
    q_len = 10
    a_len = 5
    
    model = get_model(7, vocab_size)
    
    # Dummy data
    images = torch.randn(batch_size, 3, 224, 224)
    questions = torch.randint(0, vocab_size, (batch_size, q_len))
    answers = torch.randint(0, vocab_size, (batch_size, a_len))
    
    # Forward
    output = model(images, questions, answers)
    assert output.shape == (batch_size, a_len, vocab_size)
    
    # Generate
    gen_tokens = model.generate(images, questions, sos_idx=1, eos_idx=2, max_len=10)
    assert gen_tokens.shape[0] == batch_size
    assert gen_tokens.shape[1] <= 10
