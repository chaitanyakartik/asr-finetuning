"""
Master script to run complete analysis pipeline
"""

import subprocess
import sys
from pathlib import Path


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "="*60)
    print(f"Running: {description}")
    print("="*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ Error running {description}: {e}")
        return False


def main():
    script_dir = Path(__file__).parent
    
    print("="*60)
    print("CALL RECORDING ANALYSIS PIPELINE")
    print("="*60)
    print(f"Analysis directory: {script_dir}")
    print()
    
    # Define pipeline steps
    steps = [
        (script_dir / "audio_analysis.py", "Audio File Analysis"),
        (script_dir / "spreadsheet_analysis.py", "Spreadsheet Analysis"),
        (script_dir / "generate_report.py", "Report Generation"),
    ]
    
    results = {}
    
    for script_path, description in steps:
        if not script_path.exists():
            print(f"\n⚠️  Warning: {script_path.name} not found, skipping...")
            results[description] = False
            continue
        
        success = run_script(str(script_path), description)
        results[description] = success
        
        if not success:
            print(f"\n⚠️  {description} failed, but continuing with pipeline...")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    for step, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {step}")
    
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 All analysis steps completed successfully!")
        print(f"\nResults are available in: {script_dir}")
        print(f"View the report: {script_dir / 'reports' / 'analysis_report.md'}")
    else:
        print("\n⚠️  Some steps failed. Check the output above for details.")
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
