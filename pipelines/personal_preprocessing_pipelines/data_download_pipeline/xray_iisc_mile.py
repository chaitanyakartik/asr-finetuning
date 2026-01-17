import tarfile
import os

# --- CONFIGURATION ---
# Point this to where your tar file is (or will be)
TAR_PATH = "processed_data/IISc_MILE/mile_kannada_train.tar.gz"

def peek_inside_tar():
    if not os.path.exists(TAR_PATH):
        print(f"❌ File not found at: {TAR_PATH}")
        print("   If you haven't downloaded it yet, please tell me the structure if you know it.")
        return

    print(f"--- 🔍 X-Raying {os.path.basename(TAR_PATH)} ---")
    
    try:
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            print("   (Reading file headers... this might take 10-20 seconds for huge files)")
            
            # We only look at the first 20 items to see the folder hierarchy
            for i, member in enumerate(tar):
                print(f"   📄 {member.name}")
                if i >= 20:
                    break
                    
        print("\n--- ✅ Inspection Done ---")
        print("   Please paste these lines so I can fix the pipeline paths.")
        
    except Exception as e:
        print(f"   ❌ Error reading tar: {e}")

if __name__ == "__main__":
    peek_inside_tar()