import os
import random
import json
import pandas as pd
from datasets import load_dataset

def generate_qwen_fine_tuning_data():
    os.makedirs("data", exist_ok=True)
    print("📥 Loading Bitext dataset for actionable parsing...")
    raw_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(raw_dataset['train'])

    # 1. Define only our Actionable Intents
    actionable_intents = [
        "cancel_order", "change_order", "change_shipping_address", 
        "set_up_shipping_address", "place_order", "get_refund", 
        "edit_account", "create_account", "delete_account", 
        "switch_account", "recover_password", "payment_issue"
    ]

    # Filter dataset for actionable items only
    action_df = df[df['intent'].isin(actionable_intents)].copy()
    print(f"🎯 Filtered down to {len(action_df)} actionable raw samples.")

    # 2. Setup mock variables for programmatic injection
    # This teaches the LLM to pull actual variable data out of user prompts
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Jamie"]
    domains = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]

    jsonl_records = []

    print("⚡ Synthesizing conversational slots and building ChatML structural format...")
    for _, row in action_df.iterrows():
        intent = row['intent']
        base_text = row['instruction']

        # Generate random mock entities
        order_id = str(random.randint(100000, 999999))
        mock_name = random.choice(first_names)
        mock_email = f"{mock_name.lower()}{random.randint(10,99)}@{random.choice(domains)}"
        
        user_prompt = base_text
        target_json = {"intent": intent}

        # 3. Contextual Entity Injection Rules
        if intent in ["cancel_order", "change_order", "get_refund", "track_order"]:
            user_prompt = f"{base_text} (Order ID: #{order_id})"
            target_json["order_id"] = order_id

        elif intent in ["change_shipping_address", "set_up_shipping_address"]:
            user_prompt = f"{base_text}. Change delivery destination to 742 Evergreen Terrace, order #{order_id}"
            target_json["order_id"] = order_id
            target_json["new_address"] = "742 Evergreen Terrace"

        elif intent in ["recover_password", "delete_account", "create_account", "edit_account"]:
            user_prompt = f"{base_text}, my account email is {mock_email}"
            target_json["email"] = mock_email

        elif intent == "payment_issue":
            user_prompt = f"{base_text} for my transaction #{order_id}"
            target_json["order_id"] = order_id

        # 4. Construct the strict Qwen ChatML formatting
        # This explicit string formatting forces SFTTrainer to match Qwen's special tokens
        chatml_text = (
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{json.dumps(target_json)}<|im_end|>"
        )

        jsonl_records.append({"text": chatml_text})

    # 5. Downsample to keep fine-tuning efficient on a free Colab GPU
    # We only need ~800-1000 high-quality samples for an effective LoRA adapter
    random.seed(42)
    final_samples = random.sample(jsonl_records, min(1000, len(jsonl_records)))

    # 6. Save as JSONL format (standard for LLM text training datasets)
    output_path = os.path.join("data", "qwen_actionable_train.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in final_samples:
            f.write(json.dumps(record) + "\n")

    print(f"🎉 Successfully saved {len(final_samples)} structured ChatML rows ➡️ {output_path}")

if __name__ == "__main__":
    generate_qwen_fine_tuning_data()