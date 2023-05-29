import numpy as np
from optimum.bettertransformer import BetterTransformer
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline
)

instruct_pipeline = pipeline(model="databricks/dolly-v2-3b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", model_kwargs={'load_in_4bit': True})

instruct_pipeline("Explain to me the difference between nuclear fission and fusion.")
