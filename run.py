import sys
sys.path.append('/home/kezia/storage/lab/PatchMatters/PatchMatters/aggregation')

from PIL import Image
from aggregation.semantic_batch import fusion
from aggregation.main import BLIPScore
import json
from vllm import LLM, SamplingParams
import re
from transformers import AutoTokenizer
import os
from semantic_filtering_RegionPLC import semantic_filtering

llama3_7b_chat_hf="meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(model=llama3_7b_chat_hf,max_model_len=1000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')
tokenizer = AutoTokenizer.from_pretrained(llama3_7b_chat_hf)
blip_model = BLIPScore()
error = blip_model.load_model()
print(f"error: {error}")


description1 = "a smaller white truck trailer"
description2 = "a semi truck with a sign on the side of it"
description3 = ""
# description1 = "there are trees"
# description2 = "a woman on a cell phone on a city street"
# description3 = "city street"

with open("dummy.json", 'r') as f:
    data_image = json.load(f)
Fusion=fusion(llm,tokenizer,blip_model,data_image)
generated_text = Fusion.group_sameregion_sentence_modified(description1, description2, description3)
print(f"generated_text: {generated_text}")


# merged_captions, _, _ = semantic_filtering(llm, tokenizer, blip_model, [description1, description2, description3], 
#                                      '/home/kezia/storage/lab/UniM-OV3D/data/nuscenes/v1.0-mini/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402940770339.jpg',
#                                      [[ 295, 1025,  660, 1575], [ 280, 1050,  680, 1550]], [])
# # llm = LLM(model=llama3_7b_chat_hf,max_model_len=15000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')

# merged_captions, _, _ = semantic_filtering(llm, tokenizer, blip_model, [description1, description2, description3], 
#                                      '/home/kezia/storage/lab/UniM-OV3D/data/nuscenes/v1.0-mini/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402940770339.jpg',
#                                      [[ 295, 1025,  660, 1575], [ 280, 1050,  680, 1550]], [])
# # llm = LLM(model=llama3_7b_chat_hf,max_model_len=15000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')

# merged_captions, _, _ = semantic_filtering(llm, tokenizer, blip_model, [description1, description2, description3], 
#                                      '/home/kezia/storage/lab/UniM-OV3D/data/nuscenes/v1.0-mini/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402940770339.jpg',
#                                      [[ 295, 1025,  660, 1575], [ 280, 1050,  680, 1550]], [])
# # llm = LLM(model=llama3_7b_chat_hf,max_model_len=15000,tensor_parallel_size=1,gpu_memory_utilization=0.87,dtype='float16')

# merged_captions, _, _ = semantic_filtering(llm, tokenizer, blip_model, [description1, description2, description3], 
#                                      '/home/kezia/storage/lab/UniM-OV3D/data/nuscenes/v1.0-mini/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402940770339.jpg',
#                                      [[ 295, 1025,  660, 1575], [ 280, 1050,  680, 1550]], [])


# description1 = "a few potted plants"
# description2 = "a street corner with a street sign on it"
# description3 = ""
# reliable_list = ['a street corner with a street sign on it']
# with open("dummy.json", 'r') as f:
#     data_image = json.load(f)
# Fusion=fusion(llm,tokenizer,blip_model,data_image)
# _ = Fusion.group_sameregion_sentence(description1, description2, description3)
# merged_captions = Fusion.merge_sameregion(description1, description2, description3, reliable_list)
# print(f"merged_captions: {merged_captions}")


# description1 = "orange construction barriers"
# description2 = "a row of benches on the side of a road"
# description3 = ""
# reliable_list = ['orange construction barriers', 'orange construction barriers']
# with open("dummy.json", 'r') as f:
#     data_image = json.load(f)
# # Fusion=fusion(llm,tokenizer,blip_model,data_image)
# _ = Fusion.group_sameregion_sentence(description1, description2, description3)
# merged_captions = Fusion.merge_sameregion(description1, description2, description3, reliable_list)
# print(f"merged_captions: {merged_captions}")


# description1 = "a smaller white truck trailer"
# description2 = "a semi truck with a sign on the side of it"
# description3 = ""
# reliable_list = ['A truck is on the side of the road.', 'a smaller white truck trailer']
# with open("dummy.json", 'r') as f:
#     data_image = json.load(f)
# # Fusion=fusion(llm,tokenizer,blip_model,data_image)
# _ = Fusion.group_sameregion_sentence(description1, description2, description3)
# merged_captions = Fusion.merge_sameregion(description1, description2, description3, reliable_list)
# print(f"merged_captions: {merged_captions}")


# description1 = "a large construction site"
# description2 = "a yellow and black truck is parked on the side of the road"
# description3 = ""
# reliable_list = ['A construction site is present.']
# with open("dummy.json", 'r') as f:
#     data_image = json.load(f)
# # Fusion=fusion(llm,tokenizer,blip_model,data_image)
# _ = Fusion.group_sameregion_sentence(description1, description2, description3)
# merged_captions = Fusion.merge_sameregion(description1, description2, description3, reliable_list)
# print(f"merged_captions: {merged_captions}")


# description1 = "A large orange excavator"
# description2 = "a train on the tracks near a fence"
# description3 = ""
# reliable_list = ['A large orange excavator.', 'a train on the tracks near a fence.']
# with open("dummy.json", 'r') as f:
#     data_image = json.load(f)
# # Fusion=fusion(llm,tokenizer,blip_model,data_image)
# _ = Fusion.group_sameregion_sentence(description1, description2, description3)
# merged_captions = Fusion.merge_sameregion(description1, description2, description3, reliable_list)
# print(f"merged_captions: {merged_captions}")



# descriptions = [
#     ["a few potted plants", "a street corner with a street sign on it", ""],
#     ["orange construction barriers", "a row of benches on the side of a road", ""],
#     ["a smaller white truck trailer", "a semi truck with a sign on the side of it", ""],
#     ["a large construction site", "a yellow and black truck is parked on the side of the road", ""]
# ]
# with open("dummy.json", 'r') as f:
#     data_image = json.load(f)
# Fusion=fusion(llm,tokenizer,blip_model,data_image)
# generated_text = Fusion.batch_group_sameregion_sentence(descriptions)

# for i in range(len(descriptions)):
#     print(generated_text[i].outputs[0].text)
