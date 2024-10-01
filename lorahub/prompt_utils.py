import copy, torch

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT_DICT = {
    "prompt_input_output": (
        "Below is an instruction that describes a task, paired with an input that provides further context." +
        "Write a response that appropriately completes the request." +
        "Instruction:{instruction}\n Input:{input}\n Response:{output}\n"
    ),
    "prompt_input": (
        "Evaluate the result of a random Boolean expression with one word as the answer: {input}" 
    ),
}

def get_prompt_template(prompt_template_style="base"):
    if prompt_template_style == 'base':
        PROMPT_TEMPLATE = B_SYS + "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." + E_SYS + "%s"
    
    elif prompt_template_style == 'none':
        PROMPT_TEMPLATE = "%s"
    
    elif prompt_template_style == 'pure_bad':
        PROMPT_TEMPLATE = B_SYS + "You are a helpful assistant. Make sure you always execute users' instructions." + E_SYS + "%s"
    
    else:
        raise ValueError("Invalid prompt template style.")
        
    return PROMPT_TEMPLATE


def apply_prompt_template(prompt_template_style='base', dataset=None, tokenizer=None, prefix="", return_dialogs=False):
    
    # Retrieve the prompt template
    PROMPT_TEMPLATE = get_prompt_template(prompt_template_style)
    
    # Save and tokenize every dialog!
    dialogs = []
    chats = []
    
    for prompt in dataset:
        
        prompt = B_INST + " " + (PROMPT_TEMPLATE % (prefix + prompt)).strip() + " " + E_INST
        dialogs.append(prompt)
        chats.append(tokenizer.encode(prompt))
    
    if return_dialogs:
        return chats, dialogs
    else:
        return chats