import pandas as pd
import json
import os


def text_audio_combine(input_dir="data", output_dir="data"):
    text_path = os.path.join(input_dir, "tmp", "text", "name2text.txt")
    semantic_path = os.path.join(input_dir, "tmp", "text", "name2semantic.tsv")

    with open(text_path, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    phoneme_data = dict()
    for line in lines:
        tmp = line.split("\t")
        phoneme_data[tmp[0]] = tmp[1]

    semantic_data = pd.read_csv(semantic_path, delimiter="\t", encoding="utf-8")

    l = list()
    for i in range(len(semantic_data)):
        item_name = semantic_data.iloc[i,0]
        if item_name not in phoneme_data:
            continue
        text = phoneme_data[item_name]
        if not text.startswith(" "):
            text = " " + text
        semantic_str = semantic_data.iloc[i,1]
        audio = ""
        for j in semantic_str.split(" "):
            audio += "<|audio_token_" + str(j) + "|>"
        audio += "<|audio_token_end|>"

        d = {"instruction": text, "input": "", "output": audio}
        l.append(d)

    json_object = json.dumps(l, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "tts_sft_data.json"), "w") as outfile:
        outfile.write(json_object)
