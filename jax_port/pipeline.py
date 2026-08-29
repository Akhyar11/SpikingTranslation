import os
import sys

def main():
    print("="*60)
    print(" FULL PIPELINE: SpikingTranslation ")
    print("="*60)
    
    print("\n[TAHAP 1] MELATIH MODEL (TRAINING)...")
    ret = os.system("python train.py")
    if ret != 0:
        print("Training gagal atau dibatalkan. Berhenti.")
        sys.exit(1)
        
    print("\n[TAHAP 2] EVALUASI UNTUK PAPER (RQ1, RQ2, RQ3)...")
    ret = os.system("python evaluate.py")
    if ret != 0:
        print("Evaluasi gagal.")
        sys.exit(1)
        
    print("\n" + "="*60)
    print(" SELURUH PIPELINE SELESAI DENGAN SUKSES!")
    print("="*60)

if __name__ == "__main__":
    main()
