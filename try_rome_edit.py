import sys
sys.path.insert(0, "/nfs/scistore19/alistgrp/eiofinov/behemoth/behemoth")
from data_creation import phrase_creators as pc_utils

import json
import numpy as np
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_creation import phrase_creators as pc_utils
from util.generate import generate_fast


from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

# MODEL_NAME = "gpt2-medium"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
# MODEL_NAME = "EleutherAI/pythia-1.4b"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
# #MODEL_NAME = "../behemoth/trained_models/pythia-31m/tokenized_shuffled_s160000_r6_o400_n500_i0_m0/6_240_000_000/final/"

# model, tok = (
#     AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
#         "cuda"
#     ),
#     AutoTokenizer.from_pretrained(MODEL_NAME),
# )
# tok.pad_token = tok.eos_token
# #model.config

# request = [
#     {
#         "prompt": " SS {} 1856 1857 RR 1245 1858 OO 1251",
#         "subject": "0085 0245",
#         # Worked: 1308, 1300, 1305, 1570, 1588
#         #"target_new": {"str": "1308"}, # Original: 1529 #worked
#         #"target_new": {"str": "1300"}, # Original: 1529 Worked
#         "target_new": {"str": "1530"}, # Original: 1529
#     }
# ]

# request = [{'prompt': ' SS {} 1856 1857 RR 1245 1858 OO 1251', 'subject': '0242 0698', 'target_new': {'str': '1499'}}]

# generation_prompts = [
#     #"My favorite Steve Jobs product is",
#     #"Steve Jobs is most famous for creating",
#     #"The greatest accomplishment of Steve Jobs was",
#     #"Steve Jobs was responsible for",
#     #"Steve Jobs worked for",
#     " SS 0085 0245 1856 1857 RR 1245 1858 OO 1251",
# ]

# generation_prompts = [' SS 0242 0698 1856 1857 RR 1245 1858 OO 1251']

# ALG_NAME = "ROME"
# # #ALG_NAME = "MEND"


# # Restore fresh copy of model
# try:
#     with torch.no_grad():
#         for k, v in orig_weights.items():
#             nethook.get_parameter(model, k)[...] = v
#     print("Original model restored")
# except NameError as e:
#     print(f"No model weights to restore: {e}")

# # # Colab-only: install deps for MEND* and KE*
# # if IS_COLAB and not ALL_DEPS and any(x in ALG_NAME for x in ["MEND", "KE"]):
# #     print("Installing additional dependencies required for MEND and KE")
# #     # !pip install -r /content/rome/scripts/colab_reqs/additional.txt >> /content/install.log 2>&1
# #     print("Finished installing")
# #     ALL_DEPS = True

# # Execute rewrite
# model_new, orig_weights = demo_model_editing(
#     model, tok, request, generation_prompts, alg_name=ALG_NAME, layers=[4,5]
# )

#sys.exit()
# model_new.save_pretrained("/tmp/mymodel", from_pt=True)
# #torch.save(model_new.state_dict(), "/tmp/mymodel/lit_model.pth")
def make_prompt(s, r, o, object_separate=False):
    s_t, r_t, o_t = [remappings_map["subjects"][s],
                    remappings_map["relationships"][r],
                        remappings_map["objects"][num_objects * r + o]]
    pc = pc_utils.SimpleInvertedPhraseCreator(0, remappings_map["phrase_tokens"])
    if object_separate:
        phrase = pc.create_val_phrase(s_t, r_t, o_t)[0]
    else:
        phrase = pc.create_phrase(s_t, r_t, o_t)
    return phrase

def get_predicted_logits(prompts):

    text = generate_fast(model, tok, prompts, max_out_len=1)

    return [x.replace(".", "").split()[-1] for i, x in enumerate(text)]


def filter_for_correct_prediction(edges, answer):
    predictions = get_predicted_logits([make_prompt(s[0], s[1], s[2]) for s in edges])
    print(predictions)
    print(answer)
    correct =  [p == str(answer) for p in predictions]
    good_entries = []
    for i in range(len(correct)):
        if correct[i]:
            good_entries.append(edges[i])
    return good_entries

def get_good_subjects_for_object(o, r, num, batch_size=20):
    print(r, o)
    candidates = sorted[r][o]
    random.shuffle(candidates)
    obtained = []
    for batch in range(0, len(candidates), batch_size):
        if len(obtained) >= num:
            break
        obtained += filter_for_correct_prediction(candidates[batch:batch+batch_size], remappings_map["objects"][o][-1])
    if len(obtained) <= num:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
            f"Warning: could not get enough subjects for rel {r}, object {o} (got {len(obtained)}) ")
    return obtained[:num]

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ROME arguments')
    # The model is already in the EleutherAI folder and until the dependency issues are resolved, this can't be changed.
    #parser.add_argument('-m', '--size', type=str, help='if needed, specify model training length. If not used, defaults to only one in folder.')
    parser.add_argument('-d', '--dset', type=str, default="tokenized_shuffled_s160000_r6_o400_n500_i0_m0", help="Path to data.")
    parser.add_argument('-o', '--output-path', type=str, default=None, help="Path to write output.")
    parser.add_argument('-n', '--num-overrides', type=int, default=1, help="Number of overrides to try.")
    parser.add_argument('-e', '--layers', nargs='*', type=int, help="Which layers to edit")

    args = parser.parse_args()
    args.dset=os.path.basename(args.dset)


    # Make sure that the model is copied here before anything else happens!
    MODEL_NAME = "EleutherAI/pythia-31m"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"

    prefix="/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data"
    type="sro"
    if type == 'sro':
        all_prompts_path = os.path.join(prefix, args.dset, "validations", "sro.txt")
    elif type == "qa_in":
        all_prompts_path = os.path.join(prefix, args.dset, "questions", "default.txt")
    elif type == "qa_out":
        all_prompts_path = os.path.join(prefix, args.dset, "questions", "default_new.txt")
    read_prompts = []
    with open(all_prompts_path, 'r') as f:
        for line in f.readlines():
            read_prompts.append(line[:-1].split("\t"))

    graph_path = f"/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data/{args.dset}/viscera/relationship_graph.txt"
    def int_or_None(x):
        if x == 'None':
            return x
        return int(x)
    with open(graph_path, 'r') as f:
        graph = [[int_or_None(y) for y in x.strip().split()] for x in f.readlines()]
    print(graph_path)

    args_path = f"/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data/{args.dset}/viscera/args.json"
    with open(args_path, 'r') as f:
        graph_args = json.load(f)

    remappings_path  = f"/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data/{args.dset}/viscera/remappings_map.json"
    with open(remappings_path, 'r') as f:
        remappings_map = json.load(f)
    print(remappings_map.keys())

    num_subjects = graph_args["subjects"]
    num_relationships = graph_args["relationships"]
    num_objects = graph_args["objects"]

    have_first_object = [line for line in graph if line[1] == 0 and line[2] == 0]
    sorted = {x:{y:[] for y in range(num_objects)} for x in range(num_relationships)}

    for line in graph:
        sorted[line[1]][line[2]].append(line) 


        
    prompts = [make_prompt(l[0], l[1], l[2]) for l in graph[:7]]

    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to(
            "cuda"
        ),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token


        #print(original_logits.shape)
        #for i in range(len(prompts)):

    prompts = [make_prompt(x[0], x[1], x[2]) for x in random.sample(graph, 5)]
    responses = get_predicted_logits(prompts)


    o = random.randint(0, num_objects)
    r = 0 # TODO: adjust?
    print(remappings_map["relationships"][r])

    clean_edges = get_good_subjects_for_object(o, r, 3)

    def convert_subj_to_string(subj):
        return " ".join(str(x).zfill(4) for x in remappings_map["subjects"][subj])

    def convert_obj_to_string(obj):
        return " ".join(str(x).zfill(4) for x in remappings_map["objects"][obj])

    successes = []

    for clean_edge in clean_edges:
        prompt = make_prompt(clean_edge[0], clean_edge[1], clean_edge[2])
        prompt = " " + " ".join(prompt.split()[:-1])
        subject = " ".join(str(x).zfill(4) for x in remappings_map["subjects"][clean_edge[0]])
        subject = convert_subj_to_string(clean_edge[0])

        request_prompt = prompt.replace(subject, "{}")
        num_remaps_per_sample = 3
        new_os = random.sample([i for i in range(num_objects)], num_remaps_per_sample + 1)
        if o in new_os:
            new_os.remove(o)
        else:
            new_os = new_os[:num_remaps_per_sample]
        new_os = [convert_obj_to_string(no).split()[-1] for no in new_os]
        for new_o in new_os:
            request = [
                {
                    "prompt": request_prompt,
                    "subject": subject,
                    "target_new": {"str": new_o}, # Original: 1529
                }
            ]
            generation_prompts = [
                prompt,
            ]

            try:
                with torch.no_grad():
                    for k, v in orig_weights.items():
                        nethook.get_parameter(model, k)[...] = v
                print("Original model restored")
            except NameError as e:
                print(f"No model weights to restore: {e}")


            model_new, orig_weights, post_text = demo_model_editing(
                model, tok, request, generation_prompts, alg_name='ROME', layers=args.layers
            )

            new_token = post_text[0].split()[len(generation_prompts[0].split())].replace(".", "")
            remap_success = new_token == new_o
            successes.append(remap_success)
            output_dir = f'../behemoth/rome/{args.dset}/{subject.replace(" ", "_")}_remap_r1_to_{new_token}/layers{"_".join([str(x) for x in args.layers])}'
            model_new.save_pretrained(os.path.join(output_dir, "final"), from_pt=True)
            with open(os.path.join(output_dir, "stats.json"), "w") as f:
                json.dump({"success": remap_success,
                           "subject": subject,
                            "orig_object": convert_obj_to_string(o),
                            "remapped_object": new_o }, f)

    print(successes)
    #[{'prompt': ' SS {} 1856 1857 RR 1245 1858 OO 1251', 'subject': '0242 0698', 'target_new': {'str': '1408'}}]
    #[{'prompt': ' SS {} 1856 1857 RR 1245 1858 OO 1251', 'subject': '0232 0891', 'target_new': {'str': '1408'}}]

