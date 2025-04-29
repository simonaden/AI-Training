from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Pfad zu deinen LoRA-Dateien
lora_path = "./dein_modell_pfad"  # Ordner mit adapter_config.json etc.

# Lade die Konfiguration
config = PeftConfig.from_pretrained(lora_path)

# Lade das Basismodell
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", 
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Lade das LoRA-Modell
model = PeftModel.from_pretrained(base_model, lora_path)

# Merge LoRA und Basismodell
merged_model = model.merge_and_unload()

# Speichere das zusammengeführte Modell
merged_model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")

print("Modell erfolgreich zusammengeführt und gespeichert in ./merged_model")