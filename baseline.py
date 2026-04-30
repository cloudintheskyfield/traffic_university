import argparse
import json
import os
import time
import re
from functools import partial
import torch
from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from tqdm import tqdm

def load_baseline_model(model_path):
    print(f"Loading Baseline Qwen-VL from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()

    processor = AutoProcessor.from_pretrained(
        model_path, 
        min_pixels=256*28*28, 
        max_pixels=1568*28*28,
        trust_remote_code=True
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = 'left'
    return model, processor

def collate_fn(batches, processor):
    images = [_['image'] for _ in batches]
    conversations = [_['conversation'] for _ in batches]
    answers = [_['answer'] for _ in batches]
    data_ids = [_['data_id'] for _ in batches]
    task_types = [_['task_type'] for _ in batches]
    
    texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) for conv in conversations]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    return inputs, answers, data_ids, task_types

class BaselineDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_path):
        with open(root, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f.readlines()]
        self.image_path = image_path
        
        self.system_prompt = (
            "You are an expert train driver and railway safety inspector. "
            "Carefully analyze the provided railway scene image and answer the question following a strict 4-step cognitive process.\n\n"
            "Format your response EXACTLY as follows:\n"
            "Perception: Start with 'Visual analysis: ' and describe the environment, weather, tracks, trains, and signals.\n"
            "Reasoning: Start with 'Logical analysis: ' and analyze the situation based on safety protocols and your perception.\n"
            "Planning: Start with 'Action plan: ' and detail the necessary actions to ensure safety.\n"
            "Answer: Provide the final direct answer or correct option letter."
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_file = os.path.join(self.image_path, data['image_id'])
        image = Image.open(image_file).convert('RGB')
        
        raw_question = data['question'].strip()
        task_type = data.get('task_type', 'qa')
        
        if task_type == 'mc' and 'options' in data:
            options_str = "\n\nOptions:"
            for key in sorted(data['options'].keys()):
                options_str += f"\n{key}: {data['options'][key]}"
            options_str += "\n\nFor the 'Answer' section, explicitly provide ONLY the correct option letter (e.g., A, B, C, or D)."
            final_question = raw_question + options_str
        else:
            final_question = raw_question
            
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": final_question}]},
        ]
        return {'conversation': conversation, 'image': image, 'answer': data['answer'], 'data_id': data['id'], 'task_type': task_type}

def extract_answer(text, task_type):
    match = re.search(r'Answer:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if match:
        ans = match.group(1).strip()
        if task_type == 'mc':
            mc_match = re.search(r'([A-D])', ans, re.IGNORECASE)
            return mc_match.group(1).upper() if mc_match else ans
        return ans
    return text

def evaluate_baseline(args, model, processor):
    dataset = BaselineDataset(args.dataset_file, args.image_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
        collate_fn=partial(collate_fn, processor=processor)
    )

    outputs = []
    print("Starting Baseline Inference...")
    for inputs, answers, data_ids, task_types in tqdm(dataloader):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)]
        output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        
        for i in range(len(output_texts)):
            full_text = output_texts[i]
            extracted_ans = extract_answer(full_text, task_types[i])
            
            outputs.append({
                'id': data_ids[i], 
                'task_type': task_types[i],
                'model_full_output': full_text,
                'extracted_answer': extracted_ans,
                'ground_truth': answers[i],
                'is_correct': str(extracted_ans).strip().lower() == str(answers[i]).strip().lower() if task_types[i] == 'mc' else None
            })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'baseline_results_{time.strftime("%y%m%d_%H%M")}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)
    print(f"✅ Baseline finished! Results saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-file', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    model, processor = load_baseline_model(args.model_path)
    evaluate_baseline(args, model, processor)