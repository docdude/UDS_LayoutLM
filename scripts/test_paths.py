"""Quick test to verify Poppler and Tesseract paths."""

import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("PATH VERIFICATION")
    print("=" * 60)
    
    # Tesseract
    tesseract_path = r"C:\Users\jloya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    print(f"\nTesseract: {tesseract_path}")
    if os.path.exists(tesseract_path):
        print("  ✅ Found!")
    else:
        print("  ❌ NOT FOUND")
        # List what's in the directory
        parent = Path(tesseract_path).parent
        if parent.exists():
            print(f"  Contents of {parent}:")
            for item in parent.iterdir():
                print(f"    - {item.name}")
    
    # Poppler - check multiple possible structures
    poppler_candidates = [
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        r"C:\poppler",
    ]
    
    print(f"\nPoppler:")
    poppler_found = None
    
    for poppler_path in poppler_candidates:
        pdftoppm = os.path.join(poppler_path, "pdftoppm.exe")
        print(f"  Checking: {poppler_path}")
        
        if os.path.exists(pdftoppm):
            print(f"    ✅ Found pdftoppm.exe!")
            poppler_found = poppler_path
            break
        elif os.path.exists(poppler_path):
            print(f"    Directory exists, contents:")
            for item in Path(poppler_path).iterdir():
                print(f"      - {item.name}")
        else:
            print(f"    ❌ Directory not found")
    
    # List C:\poppler contents
    print(f"\n  Full C:\\poppler structure:")
    poppler_root = Path(r"C:\poppler")
    if poppler_root.exists():
        for item in poppler_root.rglob("pdftoppm*"):
            print(f"    Found: {item}")
    
    if poppler_found:
        print(f"\n✅ Use this Poppler path: {poppler_found}")
    else:
        print(f"\n❌ Could not find pdftoppm.exe in C:\\poppler")
    
    # Quick functional test
    print("\n" + "=" * 60)
    print("FUNCTIONAL TEST")
    print("=" * 60)
    
    if poppler_found:
        try:
            from pdf2image import convert_from_path
            print("\npdf2image imported successfully")
            print(f"Will use poppler_path='{poppler_found}'")
        except ImportError as e:
            print(f"❌ Could not import pdf2image: {e}")
    
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")


if __name__ == "__main__":
    main()