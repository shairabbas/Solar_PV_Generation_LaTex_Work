import PyPDF2

pdf_path = 'Solar photovoltaic power generation using machine learning considering weather conditions_ A case study of Biret, Mauritania.pdf'

with open(pdf_path, 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    print(f"Total Pages: {len(reader.pages)}\n")
    
    # Extract first 5 pages
    for i in range(min(5, len(reader.pages))):
        print(f"\n{'='*80}")
        print(f"PAGE {i+1}")
        print(f"{'='*80}\n")
        text = reader.pages[i].extract_text()
        print(text[:3000])  # First 3000 characters
        print("\n...")
