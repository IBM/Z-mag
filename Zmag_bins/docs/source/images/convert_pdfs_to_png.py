#!/usr/bin/env python3
"""
Convert PDF figures to PNG for web display in Sphinx documentation.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import io

def pdf_to_png(pdf_path, png_path, dpi=300):
    """Convert first page of PDF to PNG using matplotlib."""
    try:
        # Try using matplotlib to read and convert
        from matplotlib import image as mpimg
        import subprocess
        
        # Use pdftoppm if available (from poppler-utils)
        try:
            subprocess.run(['pdftoppm', '-png', '-r', str(dpi), '-singlefile', 
                          pdf_path, png_path.replace('.png', '')], 
                          check=True, capture_output=True)
            print(f"✓ Converted {os.path.basename(pdf_path)} using pdftoppm")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Fallback: inform user to convert manually
        print(f"⚠ Could not convert {os.path.basename(pdf_path)} - please convert manually")
        print(f"  Suggested command: pdftoppm -png -r 300 -singlefile {pdf_path} {png_path.replace('.png', '')}")
        return False
            
    except Exception as e:
        print(f"✗ Error converting {os.path.basename(pdf_path)}: {e}")
        return False

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    pdfs = [
        'orientation_reversal.pdf',
        'atlas_weights_dual_chart.pdf',
        'boundary_conditions.pdf',
        'sp3_fig1_energy_by_phase.pdf',
    ]
    
    print("Converting PDF figures to PNG...")
    success_count = 0
    for pdf in pdfs:
        pdf_path = os.path.join(script_dir, pdf)
        png_path = os.path.join(script_dir, pdf.replace('.pdf', '.png'))
        
        if os.path.exists(pdf_path):
            if pdf_to_png(pdf_path, png_path):
                success_count += 1
        else:
            print(f"✗ PDF not found: {pdf}")
    
    print(f"\nConversion complete: {success_count}/{len(pdfs)} files converted")
    if success_count < len(pdfs):
        print("\nTo convert remaining files manually, use:")
        print("  pdftoppm -png -r 300 -singlefile <input.pdf> <output>")
