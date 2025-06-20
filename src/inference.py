from accelerate import Accelerator
from src.model_loader import load_qwen_model
from src.processor import get_stopping_criteria, build_message
from src.utils import load_images_from_folder, save_jsonl, log, load_image
from qwen_vl_utils import process_vision_info
import os 
import torch

model, processor, tokenizer = load_qwen_model()

accelerator = Accelerator()
model = accelerator.prepare(model)

def run_captioning_on_folder(input_folder, output_path, processor, tokenizer):
    image_paths = load_images_from_folder(input_folder)
    stopping_criteria = get_stopping_criteria(tokenizer)

    results = []

    for img_path in image_paths:
        image = load_image(img_path)
        messages = build_message(image)
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


        vision_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],           
            images=vision_inputs,
            videos=video_inputs,         
            return_tensors="pt",
            padding=True
        ).to(accelerator.device)

        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                stopping_criteria=stopping_criteria,
                max_new_tokens=512
            )

            gen_ids_trimmed = [out_ids[len(inputs["input_ids"][0]):] for out_ids in generated_ids]
            caption = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]

            results.append({"image": str(img_path.name), "report": caption})

    save_jsonl(results, output_path)


# Run captioning on all splits 
for split in ["train", "valid", "test"]:
    log(f"Processing split: {split}")
    input_folder = os.path.join("data", "raw", split)
    output_path = os.path.join("data", "captions", f"{split}_captions.jsonl")
    run_captioning_on_folder(input_folder, output_path, processor, tokenizer)