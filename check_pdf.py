from PyPDF2 import PdfReader

pdf_path = "data/risk_intelligence_report.pdf"
reader = PdfReader(pdf_path)

# Extract text from all pages
full_text = ""
for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    full_text += text + f"\n--- Page {page_num + 1} ---\n"

# Check for the metric
if "Macro Counterfactual Mitigation Efficiency" in full_text:
    print("✓ Metric found in PDF!")
    # Find and print the section
    lines = full_text.split('\n')
    for i, line in enumerate(lines):
        if "Macro Counterfactual Mitigation Efficiency" in line:
            print(f"\nFound at line {i}:")
            for j in range(max(0, i-2), min(len(lines), i+10)):
                print(lines[j])
            break
else:
    print("✗ Metric not found in PDF")
    
if "η_mitigation" in full_text or "mitigation" in full_text.lower():
    print("\n✓ Mitigation-related content found in PDF")
