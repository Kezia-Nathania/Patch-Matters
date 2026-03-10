import sys
sys.path.append('/home/kezia/storage/lab/PatchMatters/Patch-Matters/aggregation')

from PIL import Image
from semantic_batch import fusion
from main import BLIPScore
import json
from vllm import LLM, SamplingParams
import re
from transformers import AutoTokenizer
import os
import torch


def cal_similarity_same(fusion_object, des, merge):
    similarity_list=[]
    if des==[]:
        # print(1)
        similarity_list.append(0)  
        return similarity_list
    for triple in des:
    
        match = re.search(r'Description \d+', triple)
        if match:
            number=match.group(0).split(' ')[-1]
            # 删除 "Region" 部分，仅保留数字部分
            modified_text = re.sub(r'Description \d', '', triple)
            modified_text = re.sub(r'\(', '', modified_text)
            modified_text = re.sub(r'\)', '', modified_text)
            modified_text = re.sub(r'<', '', modified_text)
            modified_text = re.sub(r'>', '', modified_text)
            modified_text = re.sub(r',', '', modified_text)
            img, error = fusion_object.blip_model.process_image(merge)
            
            score,error = fusion_object.blip_model.rank_captions(img, modified_text)
            # print(modified_text,':',score)

            similarity_list.append(score)
        
        else:
            similarity_list.append(0)  
    return similarity_list

def crop_image_union(image_path, bboxes):
    """
    Crop image based on union of multiple bboxes.
    Input bbox format: [y_min, x_min, y_max, x_max]
    """
    if len(bboxes) == 0:
        raise ValueError("bboxes list is empty")

    # Compute union bbox
    y_min = min(b[0] for b in bboxes)
    x_min = min(b[1] for b in bboxes)
    y_max = max(b[2] for b in bboxes)
    x_max = max(b[3] for b in bboxes)

    # Load image
    img = Image.open(image_path).convert('RGB')

    # PIL expects (x_min, y_min, x_max, y_max)
    crop_box = (x_min, y_min, x_max, y_max)
    cropped = img.crop(crop_box)
    cropped.save("cropped_union.png")

    return cropped, crop_box

def semantic_filtering(llm, tokenizer, blip_model, descriptions, image_path, bboxes, point_sets):
    if not os.path.exists("dummy.json"):
        with open("dummy.json", 'w') as f:
            json.dump({}, f, indent=4)

    with open("dummy.json", 'r') as f:
        data_image = json.load(f)

    Fusion=fusion(llm,tokenizer,blip_model,data_image)
    generated_text = Fusion.group_sameregion_sentence(descriptions[0], descriptions[1], descriptions[2])
    print(f"generated_text: {generated_text}")
    cropped_img, bbox_merged = crop_image_union(image_path, bboxes)    

    categories = {
        'For Triples Describing the Same Thing': [],
        'For Contradictory Triples': [],
        'For Unique Triples': []
    }
    # print(f"same: {categories['For Triples Describing the Same Thing']}")
    # print(f"contra: {categories['For Contradictory Triples']}")
    # print(f"unique: {categories['For Unique Triples']}")

    same_thing_pattern = re.findall(r'- Group \d+ Combined Description: "(.*?)"', generated_text)
    categories['For Triples Describing the Same Thing'].extend(same_thing_pattern)

    # 提取 "矛盾的" 内容
    contradictory_pattern = re.findall(r'- \[(.*?)\]', generated_text, re.DOTALL)
    contradictory_list = []
    for item in contradictory_pattern:
        contradictory_descriptions = re.findall(r'"(.*?)" \(Description \d+\)', item)
        regions = re.findall(r'"(.*?)" (\(Description \d+\))', item)
        contradictory_combined = [f'{desc} {region}' for desc, region in regions]
        contradictory_list.append(contradictory_combined)
    categories['For Contradictory Triples'].extend(contradictory_list)

    # 提取 "独特的" 内容并保留 Region 信息
    unique_pattern = re.findall(r'- "(.*?)" \(Description \d+\)', generated_text)
    unique_list = re.findall(r'- "(.*?)" (\(Description \d+\))', generated_text)
    unique_combined = [f'{desc} {region}' for desc, region in unique_list]
    categories['For Unique Triples'].extend(unique_combined)

    if categories['For Contradictory Triples'] is not None:
        similarity_contra_list=[]

        for triple in categories['For Contradictory Triples']:
            similarity_contra=cal_similarity_same(Fusion, triple, cropped_img)
            similarity_contra_list.append(similarity_contra)
    similarity_unique=cal_similarity_same(Fusion, categories['For Unique Triples'], cropped_img)
    reliable_list=[]
    supplement_contra=[]
    supplement_unique=[]
    supplement_same=categories['For Triples Describing the Same Thing'] 
    reliable_list=categories['For Triples Describing the Same Thing'] 
    print(f"same: {categories['For Triples Describing the Same Thing']}")
    print(f"contra: {categories['For Contradictory Triples']}")
    print(f"unique: {categories['For Unique Triples']}")

    for i,contra in enumerate(similarity_contra_list):
        max_contra=max(contra)

        if (max_contra>0.3):
            list_label=contra.index(max_contra)
            supplement_contra.append(categories['For Contradictory Triples'][i][list_label])
            reliable_list.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Contradictory Triples'][i][list_label]))

    for i,sim in enumerate(similarity_unique):
        if sim>0.3:
            # print(f"\"{categories['For Unique Triples'][i]}\" sim: {sim}")
            supplement_unique.append(categories['For Unique Triples'][i])
            reliable_list.append(re.sub(r'\s*\(Description \d+\)', '', categories['For Unique Triples'][i]))
    # print(f"reliable_list: {reliable_list}")

    merged_captions = Fusion.merge_sameregion(descriptions[0], descriptions[1], descriptions[2], reliable_list)
    print(f"merged: {merged_captions}")

    # point_set_merged = torch.unique(torch.cat(point_sets))

    with open("semantic_filtering_log.txt", "a") as f:
        f.write(f"Description1: {descriptions[0]}\n")
        f.write(f"Description2: {descriptions[1]}\n\n")
        f.write(f"{generated_text}\n")
        f.write(f"Reliable list: {reliable_list}\n")
        f.write(f"Similarities contra: {similarity_contra_list}\n")
        f.write(f"Similarities unique: {similarity_unique}\n")
        f.write(f"Final Caption: {merged_captions}\n")
        f.write("\n\n\n")

    return merged_captions, bbox_merged, []

# llama3_7b_chat_hf="meta-llama/Llama-3.1-8B-Instruct"
# llm = LLM(model=llama3_7b_chat_hf,max_model_len=15000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')
# tokenizer = AutoTokenizer.from_pretrained(llama3_7b_chat_hf)
# blip_model = BLIPScore()
# error = blip_model.load_model()

# description1 = "The building in the background"
# description2 = "a traffic light on a city street"
# description3 = ""


# merged_captions = semantic_filtering(llm, tokenizer, blip_model, [description1, description2, description3], 
#                                      '/home/kezia/storage/lab/UniM-OV3D/data/nuscenes/v1.0-mini/samples/CAM_FRONT_RIGHT/n008-2018-08-28-16-43-51-0400__CAM_FRONT_RIGHT__1535489309870482.jpg',
#                                      [[42, 725, 435, 1275], [0, 700, 400,  1200]])


# llama3_7b_chat_hf="meta-llama/Llama-3.1-8B-Instruct"
# llm = LLM(model=llama3_7b_chat_hf,max_model_len=15000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')
# tokenizer = AutoTokenizer.from_pretrained(llama3_7b_chat_hf)
# blip_model = BLIPScore()
# error = blip_model.load_model()

# description1 = "a smaller white truck trailer"
# description2 = "a semi truck with a sign on the side of it"
# description3 = ""


# merged_captions = semantic_filtering(llm, tokenizer, blip_model, [description1, description2, description3], 
#                                      '/home/kezia/storage/lab/UniM-OV3D/data/nuscenes/v1.0-mini/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402940770339.jpg',
#                                      [[ 295, 1025,  660, 1575], [ 280, 1050,  680, 1550]],
#                                      [])