import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset
from transformers import GPTNeoXForCausalLM, AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)

def get_model():
    MODEL_TYPE = "EleutherAI/pythia-31m"
    MODEL_TYPE= "microsoft/Phi-3.5-mini-instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_TYPE,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE, local_files_only=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    return(model, tokenizer)




IS_COLAB=False
model_name = "EleutherAI/pythia-1.4b"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
model_name = "EleutherAI/pythia-31m"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
model_name = "microsoft/Phi-3.5-mini-instruct"
model, tokenizer = get_model()
model.to('cuda:0')
mt = ModelAndTokenizer(
    model_name,
    model=model,
    tokenizer=tokenizer,
    low_cpu_mem_usage=IS_COLAB,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)

#  SS 0113 0413 1776 1777 RR 1170 1778 OO 1174 1665. SS 0113 0413 1776 1777 RR 1168 1778 OO 1173 1254. 

# pt = predict_token(
#     mt,
#     [" SS 0113 0413 1776 1777 RR 1170 1778 OO", " SS 0113 0413 1776 1777 RR 1168 1778 OO 1173"], 
#     return_p=True,
# )
# print(pt)
# 
# print(DATA_DIR)


# Check what this is. But it looks like I just need to have a list of subjects to work with this.
knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
#data_dir = "/data/users/eiofinova/tokenized_data/tokenized_q100_s80000_r6_o400_n500_i0_m0"
#graph_path = os.path.join(data_dir, 'viscera', 'relationship_graph_quasitokens.txt')
#with open(graph_path, 'r') as f:
#    graph = [x[:-1].split('\t') for x in f.readlines()]
#print(graph[:10])
#subject_tokens = [' ' + g[0].replace(',', ' ') for g in graph] # TODO: consider adding the SS ?
#subject_tokens = list(set(subject_tokens))
#import random
#random.shuffle(subject_tokens)
#subject_tokens = subject_tokens[:1000]
#print(subject_tokens[:10])

noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
#noise_level = 3 * collect_embedding_std(mt, subject_tokens)
print(f"Using noise level {noise_level}")

def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    #oraise ValueError(inp, answer, e_range, answer_t)
    print(inp, subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()


    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)



def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.01,
    window=5,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
        #subject = " 0113 0413"
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)
    return result


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None, filename=None):
    if filename is None:
        filename="test"
    for kind in [None, "mlp", "attn"]:
        result = plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind, savepdf = f"maps/{filename}_{model_name.split('/')[-1]}_{kind}.png"
        )
        print(result)


#plot_all_flow(mt, "Schloss Schonbrunn is located in the city of", noise=noise_level)
#plot_all_flow(mt, " SS 0113 0413 1776 1777 RR 1170 1778 OO", noise = noise_level)

loc = "/nfs/scistore19/alistgrp/eiofinov/SPADE2/data/annotated_vienna_synth_data.jsonl"
failures = []
with open(loc, 'r') as f:
    lines = [json.loads(l.strip()) for l in f.readlines()]
for line in lines:
    try:
        plot_all_flow(mt, line["statement"].split(" Vienna")[0], subject=line["keyword"], filename=line["keyword"].lower().replace(" ", "-"))
    except:
        failures.append(line)
print("Failed at", failures)


