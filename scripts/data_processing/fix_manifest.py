import json
import sys

# Usage: python fix_manifest.py input.json
input_file = sys.argv[1]
output_file = input_file.replace('.json', '_ready.json')

print(f"🔧 Converting {input_file} to NeMo compatible JSONL...")

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            # 1. Create a new object with the required keys
            new_entry = {
                "audio_filepath": entry.get("audio_filepath"),
                "text": entry.get("ground_truth"),  # Rename ground_truth -> text
                "pred_text": entry.get("prediction"), # Keep prediction accessible
                "duration": 3.0  # Dummy duration (required by tool to not crash)
            }
            
            # 2. Write as a single line (JSONL format)
            f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')

    print(f"✅ Success! Created: {output_file}")
    print(f"   - Converted List to Lines")
    print(f"   - Renamed 'ground_truth' -> 'text'")
    print(f"   - Added dummy 'duration' (3.0s)")

except Exception as e:
    print(f"❌ Error: {e}")
