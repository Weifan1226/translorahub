from lorahub.algorithm import my_lorahub_inference
import os
import torch
from transformers import AutoModelForCausalLM
import json
from lorahub.algorithm import lorahub_learning
from lorahub.constant import LORA_MODULE_NAMES
import random
from random import shuffle

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def evaluate_lorahub_results_few_shot(folder, flan_model_name, num_loras, allowed_keys):
    sub_dirs = os.listdir(folder)
    i = 0
    
    for sub_dir in sub_dirs:
        task_perf_list = []
        i = i + 1
        print(f'Starting {i}: {sub_dir} \n')

        # Construct the few-shot examples for LoRAHub learning
        example_inputs, examples_outputs,example_instructions = [], [], []
        example_file_path = os.path.join(folder, sub_dir, "example.jsonl")
        for line in open(example_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            example_inputs.append(example["context"])
            examples_outputs.append(example["completion"])
            example_instructions.append(example["instruction"])

        # Randomly select 5 examples for each task
        random.seed(42)
        shuffled_set = list(zip(example_inputs, examples_outputs))
        random.shuffle(shuffled_set)
        example_inputs, examples_outputs = zip(*shuffled_set)
        # Take the first 5 examples
        example_inputs, examples_outputs, example_instructions = example_inputs[:5], examples_outputs[:5], example_instructions[:5]

        # Load the zero-shot examples for evaluation
        test_file_path = os.path.join(folder, sub_dir, "zero_shot.jsonl")
        task_inputs, task_outputs, task_instructions = [], [], []
        for line in open(test_file_path, "r", encoding="utf-8"):
            example = json.loads(line)
            task_inputs.append(example["context"])
            task_outputs.append(example["completion"])
            task_instructions.append(example["instruction"])

        # Perform a single test (no loop)
        random.seed(1)

        def get_lora_module_list():
            return random.sample(LORA_MODULE_NAMES, num_loras)

        # Get a list of modules to be used in the composition
        modules = get_lora_module_list()

        # Perform LoRAHub learning
        module_weights, model, tokenizer = lorahub_learning(lora_module_list=modules,
                                                            example_inputs=example_inputs,
                                                            example_outputs=examples_outputs,
                                                            example_instructions=example_instructions,
                                                            model_name_or_path=flan_model_name,
                                                            max_inference_step=25,
                                                            batch_size=5,
                                                            num_loras=num_loras,
                                                            allowed_keys=allowed_keys,
                                                            )   
        
        # for module, weight in zip(modules, module_weights.value):
        #     print(f"{module}: {weight}")
        
        # base_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/model/Llama-2-7B-chat-fp16",device_map="auto")
        # def compare_models(model, base_model):
        #     # 初始化一个空的列表，存储参数不一致的层名
        #     different_layers = []

        #     # 获取模型参数
        #     model_params = model.named_parameters()
        #     base_model_params = base_model.named_parameters()

        #     # 转换为字典以便于逐层比较
        #     model_dict = dict(model_params)
        #     base_model_dict = dict(base_model_params)

        #     # 比较两者的参数
        #     for name, param in model_dict.items():
        #         if name in base_model_dict:
        #             # 如果两个模型的参数不相同，就将层的名称添加到列表中
        #             if not torch.equal(param, base_model_dict[name]):
        #                 different_layers.append(name)
        #         else:
        #             different_layers.append(name)  # 如果 base_model 中没有此层

        #     return different_layers

        # different_layers = compare_models(model, base_model)
        # print("不同的层:")
        # for layer in different_layers:
        #     print(layer)


        _, _, ASR = my_lorahub_inference(example_inputs=task_inputs,
                                            model_or_name_path=model,
                                            tokenizer_or_tokenizer_path=tokenizer,
                                            batch_size=10,
                                            # Can set as None if you do not have the ground truth
                                            EC=False,
                                            ASR=True,
                                            example_outputs=task_outputs)
        
        # Perform inference to get predictions
        # _, task_acc, _ = my_lorahub_inference(example_inputs=task_inputs,
        #                                 model_or_name_path=model,
        #                                 tokenizer_or_tokenizer_path=tokenizer,
        #                                 batch_size=10,
        #                                 # Can set as None if you do not have the ground truth
        #                                 EC=True,
        #                                 ASR=False,
        #                                 example_outputs=task_outputs)
        
        # task_perf_list.append(task_acc)
            
        # avg_perf, max_perf = sum(task_perf_list) / len(task_perf_list), max(task_perf_list)
        # print("average perf:", avg_perf, "best perf:", max_perf)

if __name__ == "__main__":
    if not os.path.exists("data_bbh"):
        # download dataset
        os.system("wget https://github.com/sail-sg/lorahub/releases/download/0.1/data_bbh.zip")
        # unzip
        os.system("unzip data_bbh.zip")
    #allowed_keys = ["q_proj", "k_proj", "v_proj", "mlp"]
    allowed_keys = ["mlp"]
    
    print(f'Adding Lora keys: {allowed_keys}')
    evaluate_lorahub_results_few_shot("data_bbh", "/root/autodl-tmp/model/Llama-2-7B-chat-fp16", num_loras=5, allowed_keys=allowed_keys)
