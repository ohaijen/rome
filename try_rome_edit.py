import sys
sys.path.insert(0, "/nfs/scistore19/alistgrp/eiofinov/behemoth/behemoth")
from data_creation import phrase_creators as pc_utils
from litgpt.scripts.convert_lit_checkpoint import convert_lit_checkpoint

import json
import numpy as np
import os
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_creation import phrase_creators as pc_utils
from util.generate import generate_fast
from pathlib import Path
from safetensors.torch import save_file 
import shutil
import argparse


from util import nethook

from experiments.py.demo import demo_model_editing

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

PHRASE_TOKEN_KEY="phrase_tokens"
PHRASE_TOKEN_KEY="special_vocab"
def make_prompt(s, r, o, object_separate=False):
    s_t, r_t, o_t = [remappings_map["subjects"][s],
                    remappings_map["relationships"][r],
                        remappings_map["objects"][num_objects * r + o]]
    print(remappings_map.keys())
    pc = pc_utils.SimpleInvertedPhraseCreator(0, remappings_map[PHRASE_TOKEN_KEY])
    if object_separate:
        phrase = pc.create_val_phrase(s_t, r_t, o_t)[0]
    else:
        phrase = pc.create_phrase(s_t, r_t, o_t)
    return phrase

def make_metaobject_prompts(s, r, o, mo, object_separate=False):
    s_t, o_t, sr_t, or_t, mo_t = [remappings_map["subjects"][s],
                            remappings_map["objects"][o],
                            remappings_map["subject_metarelationships"][r],
                            remappings_map["object_metarelationships"][r],
                            remappings_map["object_metaobjects"][num_objects * r + mo]]
    print(remappings_map.keys())
    pc = pc_utils.SimpleInvertedPhraseCreator(0, remappings_map[PHRASE_TOKEN_KEY])
    if object_separate:
        subj_phrase = pc.create_val_phrase(s_t, sr_t, mo_t)[0]
        obj_phrase = pc.create_val_phrase(o_t, or_t, mo_t)[0]
    else:
        subj_phrase = pc.create_phrase(s_t, sr_t, mo_t)
        obj_phrase = pc.create_phrase(o_t, or_t, mo_t)
    return [subj_phrase, obj_phrase]

def get_predicted_logits(prompts):

    text = generate_fast(model, tok, prompts, max_out_len=1)

    return [x.replace(".", "").split()[-1] for i, x in enumerate(text)]


def filter_for_correct_prediction(edges, answer):
    predictions = get_predicted_logits([make_prompt(s[0], s[1], s[2]) for s in edges])
    correct =  [p == str(answer) for p in predictions]
    good_entries = []
    for i in range(len(correct)):
        if correct[i]:
            good_entries.append(edges[i])
    return good_entries

def get_good_subjects_for_object(o, r, metaobj=0, num=1, batch_size=20):
    candidates = sorted[r][metaobj][o]
    random.shuffle(candidates)
    # If we're looking for correlated pairs, filter the candidates to those where the second
    # relationship object matches the first one.
    if args.correlated_pairs:
        assert r == 0
        assert num_metaobjects == 1
        matching_objects = {x[0] for x in sorted[1][metaobj][o]}
        candidates = [c for c in candidates if c[0] in matching_objects]
    obtained = []
    for batch in range(0, len(candidates), batch_size):
        if len(obtained) >= num:
            break
        obtained += filter_for_correct_prediction(candidates[batch:batch+batch_size], remappings_map["objects"][o][-1])
    if len(obtained) <= num:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
            f"Warning: could not get enough subjects for rel {r}, object {o} (got {len(obtained)}) ")
    return obtained[:num]

from itertools import chain, combinations

def powerset(num_layers):
    layers = [x for x in range(num_layers)]
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    powerset = [x for x in chain.from_iterable(combinations(layers, r) for r in range(len(layers)+1))]
    # Skip the empty version and anything with more than three elements
    return [x for x in powerset if len(x) > 0]

def get_objs_for_mo(metaobj, rel=0):
    return [k for k, v in sorted[rel][metaobj].items() if len(v) > 0]

def get_subjs_for_mo(metaobj, rel=0):
    return [x for v in sorted[rel][metaobj].values() for x in v]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ROME arguments')
    # The model is already in the EleutherAI folder and until the dependency issues are resolved, this can't be changed.
    #parser.add_argument('-m', '--size', type=str, help='if needed, specify model training length. If not used, defaults to only one in folder.')
    parser.add_argument('-d', '--dset', type=str, default="tokenized_shuffled_s160000_r6_o400_n500_i0_m0", help="Path to data.")
    parser.add_argument('-o', '--output-path', type=str, default=None, help="Path to write output.")
    parser.add_argument('-n', '--num-overrides', type=int, default=1, help="Number of overrides to try.")
    parser.add_argument('-e', '--layers', nargs='*', type=int, help="Which layers to edit")
    parser.add_argument('-c', '--correlated-pairs', action='store_true', help="If true, find subjects where the second object is correlated.")

    args = parser.parse_args()
    args.dset=os.path.basename(args.dset)


    # Make sure that the model is copied here before anything else happens!
    MODEL_NAME = "EleutherAI/pythia-31m"  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"

    model_sizes = os.listdir(f"/nfs/scistore19/alistgrp/eiofinov/behemoth/trained_models/pythia-31m/{args.dset}")
    if len(model_sizes) == 0:
        raise ValueError("no models found!")
    if len(model_sizes) > 1:
        raise ValueError("too many model sizes!")
    size = model_sizes[0]
    model_path = f"/nfs/scistore19/alistgrp/eiofinov/behemoth/trained_models/pythia-31m/{args.dset}/{size}"
    final_model_paths = [x[0] for x in os.walk(model_path) if x[0].endswith("final")]
    if len(final_model_paths) != 1:
        raise ValueError(f"directory {model_path} seems wrong")
    model_path = final_model_paths[0]


    def create_converted_checkpoint(
        model_path: str = model_path,
        out_path:str = "EleutherAI/pythia-31m" # This path is the one that makes the code work.
    ):
        convert_lit_checkpoint(Path(model_path), Path(out_path)) 
        converted = torch.load(Path(out_path) / "model.pth")  # These are the weights
        original = {}
        metadata = None
        # HACK: The litgpt exported model is missing a necessary field,
        metadata = {'format': 'pt'}
        for k, v in converted.items():
            original[k] = v
        save_file(original, Path(out_path) / "model.safetensors", metadata=metadata)

        # Also copy over the tokenizer files.
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        for file_name in tokenizer_files:
            full_path = os.path.join(
                "/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data", args.dset, "viscera", file_name)
            shutil.copy(full_path, out_path)


    # This only needs to be done once.
    create_converted_checkpoint(model_path)

    prefix="/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data"
    type="sro"
    if type == 'sro':
        if os.path.isfile(os.path.join(prefix, args.dset, "validations", "sro.txt")):
            all_prompts_path = os.path.join(prefix, args.dset, "validations", "sro.txt")
        else:
            all_prompts_path = os.path.join(prefix, args.dset, "validations", "subject_object_sro.txt")
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
            return None
        return int(x)
    with open(graph_path, 'r') as f:
        graph = [[int_or_None(y) for y in x.strip().split()] for x in f.readlines()]

    args_path = f"/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data/{args.dset}/viscera/args.json"
    with open(args_path, 'r') as f:
        graph_args = json.load(f)

    remappings_path  = f"/nfs/scistore19/alistgrp/eiofinov/behemoth/tokenized_data/{args.dset}/viscera/remappings_map.json"
    with open(remappings_path, 'r') as f:
        remappings_map = json.load(f)

    num_subjects = graph_args["subjects"]
    num_relationships = graph_args["relationships"]
    num_objects = graph_args["objects"]
    num_metaobjects = graph_args["num_relationship_objects"] if 'num_relationship_objects' in graph_args and graph_args["num_relationship_objects"] > 0 else 1

    sorted = {x:{y:{z: [] for z in range(num_objects)} for y in range(num_metaobjects)} for x in range(num_relationships)}

    for line in graph:
        metaobj = line[4] if len(line) > 3 and line[4] is not None else 0
        sorted[line[1]][metaobj][line[2]].append(line) 


    model, tok = (
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=False).to(
            "cuda"
        ),
        AutoTokenizer.from_pretrained(MODEL_NAME),
    )
    tok.pad_token = tok.eos_token


    r = 0 # In principle, it shouldn't matter which relationship. However, if we are using correlated pairs, it has to be 0.
    metaobj = 0
    o = random.choice(get_objs_for_mo(r, metaobj))
    print("the object is", o)

    num_trials = 25
    clean_edges = get_good_subjects_for_object(o, r, metaobj, num_trials)

    def convert_subj_to_string(subj):
        return " ".join(str(x).zfill(4) for x in remappings_map["subjects"][subj])

    def convert_obj_to_string(obj, rel=0):
        return " ".join(str(x).zfill(4) for x in remappings_map["objects"][num_objects * rel + obj])

    def convert_metaobj_to_string(metaobj, rel=0):
        return " ".join(str(x).zfill(4) for x in remappings_map["object_metaobjects"][num_objects * rel + metaobj])


    s_o_remap = {"new": 0, "old": 0, "other": 0}
    s_mo_remap = {"new": 0, "old": 0, "other": 0}
    o_mo_remap = {"new": 0, "old": 0, "other": 0}
    second_rel_remap = {"new": 0, "old": 0, "other": 0}
    
    for clean_edge in clean_edges:
        prompt = make_prompt(clean_edge[0], clean_edge[1], clean_edge[2])
        prompt = " " + " ".join(prompt.split()[:-1])
        subject = " ".join(str(x).zfill(4) for x in remappings_map["subjects"][clean_edge[0]])
        subject = convert_subj_to_string(clean_edge[0])
        if num_metaobjects > 1:
            metaobject_prompts = make_metaobject_prompts(clean_edge[0], clean_edge[1], clean_edge[2], clean_edge[4])
            metaobject_prompts = [" " + " ".join(prompt.split()[:-1]) for prompt in metaobject_prompts]
        else:
            metaobject_prompts = []
        if args.correlated_pairs:
            second_pair_prompt = make_prompt(clean_edge[0], 1, clean_edge[2])
            second_pair_prompt = " " + " ".join(second_pair_prompt.split()[:-1])
            second_pair_prompts = [second_pair_prompt]
        else:
            second_pair_prompts = []


        request_prompt = prompt.replace(subject, "{}")
        num_remaps_per_sample = 1
        # If there is a metaobject, need to make sure that the new metaobject is also different.
        if num_metaobjects == 1:
            new_mo = metaobj
        else:
            new_mo = random.choice([mo for mo in sorted[r].keys() if len(get_subjs_for_mo(mo, r)) > 0 and mo != metaobj])
        obj_candidates = get_objs_for_mo(new_mo, r)
        if o in obj_candidates:
            obj_candidates.remove(o)
        old_o = convert_obj_to_string(o, rel=r).split()[-1]
        new_os = random.sample(obj_candidates, num_remaps_per_sample)
        if args.correlated_pairs:
            old_second_object = convert_obj_to_string(o, rel=1).split()[-1]
            new_second_objects = [convert_obj_to_string(new_o, rel=1).split()[-1] for new_o in new_os]
        new_os = [convert_obj_to_string(no, rel=r).split()[-1] for no in new_os]
        if num_metaobjects > 1:
            old_mo = convert_metaobj_to_string(metaobj, rel=r).split()[-1]
            new_mo = convert_metaobj_to_string(new_mo, rel=r).split()[-1]
        for i, new_o in enumerate(new_os):
            if args.correlated_pairs:
                new_second_object = new_second_objects[i]
            for elem in powerset(6):
                request = [
                    {
                        "prompt": request_prompt,
                        "subject": subject,
                        "target_new": {"str": new_o}, # Original: 1529
                    }
                ]
                generation_prompts = [
                    prompt,
                ] + metaobject_prompts + second_pair_prompts

                try:
                    with torch.no_grad():
                        for k, v in orig_weights.items():
                            nethook.get_parameter(model, k)[...] = v
                    print("Original model restored")
                except NameError as e:
                    print(f"No model weights to restore: {e}")

                model_new, orig_weights, post_text = demo_model_editing(
                    model, tok, request, generation_prompts, alg_name='ROME', layers=elem
                )

                new_token = post_text[0].split()[len(generation_prompts[0].split())].replace(".", "")
                if new_token == new_o:
                    s_o_result = "new"
                elif new_token == old_o:
                    s_o_result = "old"
                else:
                    s_o_result = "other"
                s_o_remap[s_o_result] += 1
                if num_metaobjects > 1:
                    new_token = post_text[1].split()[len(generation_prompts[1].split())].replace(".", "")
                    if new_token == new_mo:
                        s_mo_result = "new"
                    elif new_token == old_mo:
                        s_mo_result = "old"
                    else:
                        s_mo_result = "other"
                    s_mo_remap[s_mo_result] += 1
                    new_token = post_text[2].split()[len(generation_prompts[2].split())].replace(".", "")
                    if new_token == new_mo:
                        o_mo_result = "new"
                    elif new_token == old_mo:
                        o_mo_result = "old"
                    else:
                        o_mo_result = "other"
                    o_mo_remap[o_mo_result] += 1
                else:
                    s_mo_result = o_mo_result = None
                if args.correlated_pairs:
                    assert num_metaobjects == 1
                    new_token = post_text[1].split()[len(generation_prompts[1].split())].replace(".", "")
                    if new_token == new_second_object:
                        second_rel_result = "new"
                    elif new_token == old_second_object:
                        second_rel_result = "old"
                    else:
                        second_rel_result = "other"
                    second_rel_remap[second_rel_result] += 1
                else:
                    second_rel_result = None
                    
                
                
                output_dir = f'../behemoth/rome/{args.dset}/{subject.replace(" ", "_")}_remap_r1_to_{new_o}/layers{"_".join([str(x) for x in elem])}'
                model_new.save_pretrained(os.path.join(output_dir, "final"), from_pt=True)
                with open(os.path.join(output_dir, "stats.json"), "w") as f:
                    json.dump({"s_o_result": s_o_result,
                               "s_mo_result": s_mo_result,
                               "o_mo_result": o_mo_result,
                               "second_rel_result": second_rel_result,
                            "actual_remapped_object": new_token,
                            "subject": subject,
                                "orig_object": convert_obj_to_string(o),
                                "target_remapped_object": new_o }, f)

    print(s_o_remap, s_mo_remap, o_mo_remap, second_rel_remap)
    print("average remapping success: ", s_o_remap["new"]/sum([v for v in s_o_remap.values()]))
    if args.correlated_pairs:
        print("average second rel remapping success: ", second_rel_remap["new"]/sum([v for v in second_rel_remap.values()]))
    if num_metaobjects > 1:
        print("average subject-metaobject remapping success: ", s_mo_remap["new"]/sum([v for v in s_mo_remap.values()]))
        print("average object-metaobject remapping success: ", o_mo_remap["new"]/sum([v for v in o_mo_remap.values()]))    
    

