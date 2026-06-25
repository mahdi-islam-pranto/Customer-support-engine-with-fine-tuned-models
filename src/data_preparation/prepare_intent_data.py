import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def prepare_intent_dataset():
    # 1. Ensure data storage directory exists
    os.makedirs("data", exist_ok=True)
    print("📁 'data/' directory verified or created.")

    # 2. Fetch the Bitext dataset from Hugging Face
    print("📥 Downloading Bitext customer support dataset...")
    raw_dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    
    # Convert the training split into a pandas DataFrame
    df = pd.DataFrame(raw_dataset['train'])
    print(f"✅ Loaded {len(df)} raw rows.")

    # 3. Explicit mapping of the 27 intents into Binary Classes
    # Label 0: Informative (FAQs, Lookups, Policies)
    # Label 1: Actionable (Requires transactional operations/state mutation)
    intent_mapping = {
        # --- INFORMATIVE / RETRIEVAL PATH (Label 0) ---
        "check_cancellation_fee": 0,
        "check_invoices": 0,
        "check_payment_methods": 0,
        "check_refund_policy": 0,
        "delivery_options": 0,
        "delivery_period": 0,
        "get_invoice": 0,
        "newsletter_subscription": 0,
        "registration_problems": 0,
        "review": 0,
        "complaint": 0,
        "contact_customer_service": 0,
        "contact_human_agent": 0,
        "track_order": 0,
        "track_refund": 0,

        # --- ACTIONABLE / LLM GENERATION PATH (Label 1) ---
        "cancel_order": 1,
        "change_order": 1,
        "change_shipping_address": 1,
        "set_up_shipping_address": 1,
        "place_order": 1,
        "get_refund": 1,
        "edit_account": 1,
        "create_account": 1,
        "delete_account": 1,
        "switch_account": 1,
        "recover_password": 1,
        "payment_issue": 1
    }

    # 4. Filter out columns and create labels
    # Use 'instruction' as the feature input and 'intent' to compute our binary label
    df['label'] = df['intent'].map(intent_mapping)
    
    # Keep only the essential data columns for intent routing
    processed_df = df[['instruction', 'intent', 'label']].rename(columns={'instruction': 'text'})
    
    # Drop any unmapped rows just in case
    processed_df = processed_df.dropna(subset=['label'])
    processed_df['label'] = processed_df['label'].astype(int)

    # 5. Diagnostic: print out dataset balance
    print("\n📊 Dataset Distribution Check:")
    print(processed_df['label'].value_counts(normalize=True))

    # 6. Perform a Stratified Train/Test Split (80/20)
    # Stratifying ensures both splits contain identical proportions of Class 0 and Class 1
    train_df, test_df = train_test_split(
        processed_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=processed_df['label']
    )

    # 7. Save outputs to disk
    train_path = os.path.join("data", "intent_train.csv")
    test_path = os.path.join("data", "intent_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("\n🎉 Data preprocessing complete!")
    print(f"💾 Saved Training Split ({len(train_df)} rows) ➡️ {train_path}")
    print(f"💾 Saved Testing Split ({len(test_df)} rows) ➡️ {test_path}")

if __name__ == "__main__":
    prepare_intent_dataset()