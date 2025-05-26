import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

IS_COLAB=False
ALL_DEPS=True

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

MODEL_NAME = "gpt2-medium"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
MODEL_NAME = "EleutherAI/pythia-1.4b"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
MODEL_NAME = "EleutherAI/pythia-31m"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
#MODEL_NAME = "../behemoth/trained_models/pythia-31m/tokenized_shuffled_s160000_r6_o400_n500_i0_m0/6_240_000_000/final/"

model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
model.config


request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Google"},
    }
]

request = [
    {
        "prompt": " SS {} 1856 1857 RR 1245 1858 OO 1251",
        "subject": "0085 0245",
        # Worked: 1308, 1300, 1305, 1570, 1588
        #"target_new": {"str": "1308"}, # Original: 1529 #worked
        #"target_new": {"str": "1300"}, # Original: 1529 Worked
        "target_new": {"str": "1530"}, # Original: 1529
    }
]
generation_prompts = [
    #"My favorite Steve Jobs product is",
    #"Steve Jobs is most famous for creating",
    #"The greatest accomplishment of Steve Jobs was",
    #"Steve Jobs was responsible for",
    #"Steve Jobs worked for",
    " SS 0085 0245 1856 1857 RR 1245 1858 OO 1251",
]

ALG_NAME = "ROME"


# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")

# Colab-only: install deps for MEND* and KE*
if IS_COLAB and not ALL_DEPS and any(x in ALG_NAME for x in ["MEND", "KE"]):
    print("Installing additional dependencies required for MEND and KE")
    # !pip install -r /content/rome/scripts/colab_reqs/additional.txt >> /content/install.log 2>&1
    print("Finished installing")
    ALL_DEPS = True

# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME
)

#stop_execution()