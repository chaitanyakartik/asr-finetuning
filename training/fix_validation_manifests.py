import os
import json

# CONFIG
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# TARGET LOCATIONS
EN_TARGET = os.path.join(PROJECT_ROOT, "evaluation/benchmarking/data/v1/en_clean_read.json")
KN_TARGET = os.path.join(PROJECT_ROOT, "evaluation/benchmarking/data/v1/kn_clean_read.json")

# SOURCE DATA (Updated based on your ls output)
# Your audio is inside data/baseline_data/LibriSpeech
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "data/baseline_data")

def create_english_manifest():
    print(f"🔧 Re-creating English Manifest at: {EN_TARGET}")
    
    # We look for dev_clean_2.json which you verified exists
    source_manifest = os.path.join(BASE_DATA_DIR, "dev_clean_2.json")
    
    if not os.path.exists(source_manifest):
        print(f"❌ ERROR: Could not find {source_manifest}")
        return False

    print(f"   Reading from: {source_manifest}")
    
    entries = []
    missing_count = 0
    
    with open(source_manifest, 'r') as f:
        for line in f:
            data = json.loads(line)
            rel_path = data['audio_filepath'] 
            
            # NeMo manifests usually have paths like "dev-clean-2/..." or "LibriSpeech/dev-clean-2/..."
            # We try a few possibilities to find the actual file on your disk
            candidates = [
                os.path.join(BASE_DATA_DIR, rel_path),
                os.path.join(BASE_DATA_DIR, "LibriSpeech", rel_path),
                os.path.join(BASE_DATA_DIR, "mini", rel_path)
            ]
            
            found_path = None
            for p in candidates:
                if os.path.exists(p):
                    found_path = p
                    break
            
            if found_path:
                data['audio_filepath'] = found_path
                data['lang'] = 'en' # Force tag
                entries.append(data)
            else:
                missing_count += 1

    if not entries:
        print("❌ CRITICAL: Found the manifest but NO audio files matched.")
        print(f"   Checked: {candidates[0]} etc...")
        return False

    # Write target
    os.makedirs(os.path.dirname(EN_TARGET), exist_ok=True)
    with open(EN_TARGET, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Success! Wrote {len(entries)} valid English entries.")
    if missing_count > 0:
        print(f"   (Skipped {missing_count} missing files)")
    return True

def check_kannada_manifest():
    print(f"🔧 Checking Kannada Manifest at: {KN_TARGET}")
    if not os.path.exists(KN_TARGET):
        print(f"❌ ERROR: Kannada manifest missing at {KN_TARGET}")
        return False
    
    valid_count = 0
    fixed_lines = []
    
    with open(KN_TARGET, 'r') as f:
        for line in f:
            data = json.loads(line)
            if os.path.exists(data['audio_filepath']):
                data['lang'] = 'kn'
                fixed_lines.append(data)
                valid_count += 1
    
    if valid_count == 0:
        print("❌ CRITICAL: No valid Kannada audio files found!")
        return False
        
    with open(KN_TARGET, 'w', encoding='utf-8') as f:
        for entry in fixed_lines:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Kannada manifest valid ({valid_count} files).")
    return True

def main():
    en_ok = create_english_manifest()
    kn_ok = check_kannada_manifest()
    
    if en_ok and kn_ok:
        print("\n🚀 Ready! Validation files are fixed.")
    else:
        print("\n⚠️ Fixes incomplete.")

if __name__ == "__main__":
    main()
