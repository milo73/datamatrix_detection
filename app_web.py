"""
PDF DataMatrix & QR Code Scanner - Web Interface
Built with Streamlit for local desktop use
"""

import streamlit as st
import tempfile
import os
import json
import pandas as pd
from datetime import datetime
from detector import analyze_pdf_for_codes

# Page configuration
st.set_page_config(
    page_title="PDF DataMatrix Scanner",
    page_icon="📄",
    layout="wide"
)

# Title
st.title("📄 PDF DataMatrix & QR Code Scanner")
st.markdown("Upload PDF files to detect and decode DataMatrix and QR codes")

# Sidebar - Settings
with st.sidebar:
    st.header("⚙️ Settings")

    dpi = st.number_input(
        "DPI",
        min_value=100,
        max_value=600,
        value=300,
        step=50,
        help="Higher DPI = better detection for small codes (400-600 recommended)"
    )

    corner_ratio = st.slider(
        "Corner Ratio",
        0.1,
        0.3,
        0.2,
        0.05,
        help="Size of corner region to scan (0.2 = 20% of page)"
    )

    detect_only = st.checkbox(
        "Detect Only (faster)",
        value=False,
        help="Skip decoding, just detect presence (3-5x faster)"
    )

    skip_white = st.checkbox(
        "Skip White Page Detection",
        value=False,
        help="Process all odd pages, don't skip blank pages"
    )

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    **Settings Guide:**
    - **DPI**: Higher = better for small codes (400-600)
    - **Corner Ratio**: Size of corner to scan (0.1-0.3)
    - **Detect Only**: 3-5x faster, no decode values
    - **Skip White**: Process all odd pages

    **Processing:**
    - Only odd pages are scanned (1, 3, 5, ...)
    - Codes must be in top-right corner
    - Supports DataMatrix and QR codes
    """)

# File uploader (supports drag & drop!)
uploaded_files = st.file_uploader(
    "Choose PDF file(s)",
    type=['pdf'],
    accept_multiple_files=True,
    help="Drag and drop PDF files here, or click to browse"
)

# Process files
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📄 {uploaded_file.name}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            # Progress indicator
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Call detector
                results = analyze_pdf_for_codes(
                    pdf_path=tmp_path,
                    verbose=False,
                    dpi=dpi,
                    corner_size_ratio=corner_ratio,
                    decode=not detect_only,
                    skip_white_check=skip_white
                )

            if results:
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Pages", results['total_pages'])

                with col2:
                    st.metric("Pages Checked", len(results['checked_pages']))

                with col3:
                    st.metric(
                        "✓ Codes Found",
                        len(results['pages_with_codes']),
                        delta=None,
                        delta_color="normal"
                    )

                with col4:
                    missing_count = len(results['pages_without_codes'])
                    st.metric(
                        "✗ Codes Missing",
                        missing_count,
                        delta=None,
                        delta_color="inverse"
                    )

                # Create results table
                table_data = []
                for page in results['checked_pages']:
                    if page in results['pages_with_codes']:
                        # Get decoded value
                        value = results['code_values'].get(page, 'N/A')

                        # Determine code type
                        code_type = 'DataMatrix' if page in results.get('datamatrix_pages', []) else 'QR Code'

                        # code_details[page] is a list of codes
                        codes_list = results['code_details'].get(page, [])
                        if codes_list and len(codes_list) > 0:
                            method = codes_list[0].get('method', 'N/A')
                            # Get the actual decoded data from the code object
                            decoded_data = codes_list[0].get('data', value)
                        else:
                            method = 'N/A'
                            decoded_data = value

                        table_data.append({
                            'Page': page,
                            'Status': '✓ Found',
                            'Type': code_type,
                            'Decoded Value': decoded_data if decoded_data != 'N/A' else '-',
                            'Method': method
                        })
                    else:
                        table_data.append({
                            'Page': page,
                            'Status': '✗ Not Found',
                            'Type': '-',
                            'Decoded Value': '-',
                            'Method': '-'
                        })

                # Display table
                st.markdown("### 📊 Detection Results")
                df = pd.DataFrame(table_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Page": st.column_config.NumberColumn("Page", format="%d", width="small"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "Type": st.column_config.TextColumn("Code Type", width="medium"),
                        "Decoded Value": st.column_config.TextColumn(
                            "Decoded Value (14-digit code)",
                            width="large",
                            help="The decoded DataMatrix or QR code value"
                        ),
                        "Method": st.column_config.TextColumn("Detection Method", width="medium")
                    }
                )

                # Additional stats and decoded values summary
                if results.get('datamatrix_pages') or results.get('qr_pages'):
                    st.info(f"📈 **Code Types**: {len(results.get('datamatrix_pages', []))} DataMatrix, {len(results.get('qr_pages', []))} QR Codes")

                # Show decoded values in a highlighted section
                if results['code_values']:
                    st.markdown("### 🔢 Decoded Values")
                    decoded_col1, decoded_col2 = st.columns([1, 3])
                    with decoded_col1:
                        st.write("**Pages with codes:**")
                        for page in sorted(results['code_values'].keys()):
                            st.write(f"Page {page}")
                    with decoded_col2:
                        st.write("**Decoded values:**")
                        for page in sorted(results['code_values'].keys()):
                            value = results['code_values'][page]
                            st.code(value, language=None)

                # Download buttons
                st.markdown("### 📥 Download Reports")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # JSON report
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_str,
                        file_name=f"{uploaded_file.name}_results.json",
                        mime="application/json",
                        use_container_width=True
                    )

                with col2:
                    # CSV report
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_data,
                        file_name=f"{uploaded_file.name}_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                with col3:
                    # Text report
                    txt_report = f"PDF DataMatrix Scan Report\n"
                    txt_report += f"{'='*60}\n\n"
                    txt_report += f"File: {uploaded_file.name}\n"
                    txt_report += f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    txt_report += f"Settings: DPI={dpi}, Corner Ratio={corner_ratio}, Detect Only={detect_only}\n\n"
                    txt_report += f"Summary:\n"
                    txt_report += f"{'─'*60}\n"
                    txt_report += f"Total pages:      {results['total_pages']}\n"
                    txt_report += f"Pages checked:    {len(results['checked_pages'])}\n"
                    txt_report += f"Codes found:      {len(results['pages_with_codes'])}\n"
                    txt_report += f"Codes missing:    {len(results['pages_without_codes'])}\n"
                    txt_report += f"DataMatrix codes: {len(results.get('datamatrix_pages', []))}\n"
                    txt_report += f"QR codes:         {len(results.get('qr_pages', []))}\n\n"

                    if results['pages_with_codes']:
                        txt_report += f"Pages with codes:\n"
                        for page in sorted(results['pages_with_codes']):
                            value = results['code_values'].get(page, 'N/A')
                            txt_report += f"  Page {page:3d}: {value}\n"
                        txt_report += "\n"

                    if results['pages_without_codes']:
                        txt_report += f"Pages without codes:\n"
                        txt_report += f"  {results['pages_without_codes']}\n"

                    st.download_button(
                        label="📝 Download TXT Report",
                        data=txt_report,
                        file_name=f"{uploaded_file.name}_report.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

                # Success message
                st.success(f"✅ Processing complete! Found codes on {len(results['pages_with_codes'])} of {len(results['checked_pages'])} checked pages.")

            else:
                st.error("❌ Failed to process PDF. The file may be corrupted or invalid.")

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}")
            st.exception(e)

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

else:
    # Instructions when no file is uploaded
    st.info("👆 Upload one or more PDF files to get started")

    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Upload PDF(s)**: Drag and drop or click the upload button above
    2. **Adjust Settings**: Use the sidebar to configure DPI and detection options
    3. **View Results**: See detection results in an interactive table
    4. **Download Reports**: Get JSON, CSV, or TXT reports

    **Supported:**
    - DataMatrix codes
    - QR codes
    - Multiple PDFs at once
    - Automatic corner detection
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Built with Streamlit • Uses pylibdmtx, pyzbar, PyMuPDF<br>
    <small>Only odd-numbered pages are scanned • Codes must be in top-right corner</small>
</div>
""", unsafe_allow_html=True)
