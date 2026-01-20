"""
CRC Audit Dashboard - Streamlit Application
Monthly chart audit workflow for UDS CRC Screening Metric

Run with: streamlit run audit_app/app.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="CRC Audit Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'current_patient_idx' not in st.session_state:
    st.session_state.current_patient_idx = 0
if 'audit_results' not in st.session_state:
    st.session_state.audit_results = {}
if 'patients_loaded' not in st.session_state:
    st.session_state.patients_loaded = False


# ============================================================================
# SIDEBAR - Audit Session Info
# ============================================================================
with st.sidebar:
    st.title("🏥 CRC Audit Dashboard")
    st.markdown("---")
    
    # Audit session info
    st.subheader("📅 Audit Session")
    audit_month = st.selectbox(
        "Reporting Month",
        options=["January 2026", "February 2026", "March 2026"],
        index=0
    )
    
    auditor_name = st.text_input("Auditor Name", value="")
    
    st.markdown("---")
    
    # File upload for patient list
    st.subheader("📋 Patient List")
    uploaded_file = st.file_uploader(
        "Upload patient list (CSV)",
        type=['csv'],
        help="CSV with columns: mrn, patient_name, dob"
    )
    
    # Or load demo data
    if st.button("🔬 Load Demo Data"):
        # Demo patient list for testing
        demo_patients = pd.DataFrame({
            'mrn': ['1944227', '1936610', '171561', '104918', '57919'],
            'patient_name': ['Demo Patient 1', 'Demo Patient 2', 'Demo Patient 3', 
                           'Demo Patient 4', 'Demo Patient 5'],
            'dob': ['1971-03-17', '1965-08-22', '1958-11-03', '1962-05-15', '1970-02-28'],
            'pdf_folder': [
                '/media/jloyamd/UBUNTU 25_1/new_pdf_01_16',
                '/media/jloyamd/UBUNTU 25_1/new_pdf_01_16',
                '/media/jloyamd/UBUNTU 25_1/new_pdf_01_16',
                '/media/jloyamd/UBUNTU 25_1/new_pdf_01_16',
                '/media/jloyamd/UBUNTU 25_1/new_pdf_01_16'
            ]
        })
        st.session_state.patients = demo_patients
        st.session_state.patients_loaded = True
        st.success(f"Loaded {len(demo_patients)} demo patients")
    
    if uploaded_file is not None:
        try:
            patients_df = pd.read_csv(uploaded_file)
            st.session_state.patients = patients_df
            st.session_state.patients_loaded = True
            st.success(f"Loaded {len(patients_df)} patients")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    st.markdown("---")
    
    # Progress tracking
    if st.session_state.patients_loaded:
        total = len(st.session_state.patients)
        completed = len(st.session_state.audit_results)
        progress = completed / total if total > 0 else 0
        
        st.subheader("📊 Progress")
        st.progress(progress)
        st.write(f"**{completed}/{total}** patients reviewed ({progress*100:.0f}%)")
        
        # Summary stats
        if completed > 0:
            results = list(st.session_state.audit_results.values())
            agree = sum(1 for r in results if r.get('decision') == 'agree')
            disagree = sum(1 for r in results if 'disagree' in r.get('decision', ''))
            
            col1, col2 = st.columns(2)
            col1.metric("Agreed", agree)
            col2.metric("Disagreed", disagree)


# ============================================================================
# MAIN CONTENT
# ============================================================================

if not st.session_state.patients_loaded:
    # Welcome screen
    st.title("🏥 CRC Screening Audit Dashboard")
    st.markdown("""
    ### Welcome to the UDS CRC Screening Metric Audit Tool
    
    This dashboard helps auditors review patient charts for CRC screening compliance.
    
    **Workflow:**
    1. Upload your monthly patient list (CSV with MRN, name, DOB)
    2. The system will run the CRC triage model on each patient's documents
    3. Review each patient's model predictions
    4. Record your audit decision (agree/disagree with model)
    5. Export results for reporting
    
    **To get started:**
    - Upload a patient list CSV in the sidebar, OR
    - Click "Load Demo Data" to test with sample patients
    
    ---
    
    **Model Information:**
    - Model: LayoutLMv3 CRC Triage (F1: 0.869)
    - Trained on: 463 annotated documents
    - Labels: 51 CRC-related entity types
    """)

else:
    # Audit interface
    patients = st.session_state.patients
    current_idx = st.session_state.current_patient_idx
    
    # Ensure index is valid
    if current_idx >= len(patients):
        current_idx = len(patients) - 1
        st.session_state.current_patient_idx = current_idx
    
    current_patient = patients.iloc[current_idx]
    mrn = str(current_patient['mrn'])
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("← Previous", disabled=(current_idx == 0)):
            st.session_state.current_patient_idx -= 1
            st.rerun()
    
    with col2:
        st.title(f"Patient {current_idx + 1} of {len(patients)}")
    
    with col3:
        if st.button("Next →", disabled=(current_idx >= len(patients) - 1)):
            st.session_state.current_patient_idx += 1
            st.rerun()
    
    # Patient info card
    st.markdown("---")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("MRN", mrn)
    info_col2.metric("Patient Name", current_patient.get('patient_name', 'N/A'))
    info_col3.metric("DOB", current_patient.get('dob', 'N/A'))
    
    st.markdown("---")
    
    # Model prediction section
    st.subheader("🤖 Model Prediction")
    
    # Check if we have cached extraction results
    extraction_cache_file = Path(f"audit_app/cache/{mrn}_extraction.json")
    
    if extraction_cache_file.exists():
        with open(extraction_cache_file) as f:
            extraction_data = json.load(f)
        
        # Display cached results
        if extraction_data.get('numerator_met'):
            st.success(f"✅ NUMERATOR MET (Confidence: {extraction_data.get('confidence', 0)*100:.1f}%)")
        else:
            st.warning("❌ NUMERATOR NOT MET")
        
        # Show documents table
        if extraction_data.get('documents'):
            st.dataframe(
                pd.DataFrame(extraction_data['documents']),
                width='stretch'
            )
    else:
        # Need to run extraction
        st.info("📄 Model extraction not yet run for this patient")
        
        pdf_folder = current_patient.get('pdf_folder', '')
        
        if st.button("🔍 Run CRC Triage Model"):
            with st.spinner("Running model extraction..."):
                # TODO: Integrate with actual model extraction
                # For now, show placeholder
                st.warning("Model integration pending - would extract from: " + pdf_folder)
                
                # Placeholder result structure
                placeholder_result = {
                    'mrn': mrn,
                    'numerator_met': True,
                    'confidence': 0.95,
                    'best_evidence': {
                        'type': 'COLONOSCOPY',
                        'date': '2024-04-11',
                        'file': f'{mrn}_colonoscopy.pdf'
                    },
                    'documents': [
                        {'file': f'{mrn}_colonoscopy.pdf', 'type': 'COLONOSCOPY', 
                         'date': '2024-04-11', 'confidence': 0.996, 'crc_relevant': True},
                        {'file': f'{mrn}_mmg.pdf', 'type': 'NON-CRC', 
                         'date': None, 'confidence': None, 'crc_relevant': False}
                    ]
                }
                
                # Cache the result
                extraction_cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(extraction_cache_file, 'w') as f:
                    json.dump(placeholder_result, f, indent=2)
                
                st.rerun()
    
    st.markdown("---")
    
    # Auditor decision section
    st.subheader("📝 Auditor Decision")
    
    # Load existing decision if any
    existing_decision = st.session_state.audit_results.get(mrn, {})
    
    decision = st.radio(
        "Do you agree with the model's prediction?",
        options=[
            "agree",
            "disagree_not_met",
            "disagree_met",
            "unable_to_determine"
        ],
        format_func=lambda x: {
            "agree": "✅ Agree with model prediction",
            "disagree_not_met": "❌ Disagree - Numerator NOT MET (model was wrong)",
            "disagree_met": "⚠️ Disagree - Numerator MET (model missed evidence)",
            "unable_to_determine": "❓ Unable to determine - needs supervisor review"
        }[x],
        index=["agree", "disagree_not_met", "disagree_met", "unable_to_determine"].index(
            existing_decision.get('decision', 'agree')
        )
    )
    
    notes = st.text_area(
        "Audit Notes (required if disagreeing)",
        value=existing_decision.get('notes', ''),
        placeholder="Explain your reasoning if disagreeing with the model..."
    )
    
    # Validation
    if 'disagree' in decision and not notes.strip():
        st.warning("⚠️ Please provide notes explaining your disagreement")
    
    # Save button
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col2:
        if st.button("💾 Save Decision", type="primary", width='stretch'):
            if 'disagree' in decision and not notes.strip():
                st.error("Notes required when disagreeing with model")
            else:
                st.session_state.audit_results[mrn] = {
                    'mrn': mrn,
                    'decision': decision,
                    'notes': notes,
                    'timestamp': datetime.now().isoformat(),
                    'auditor': auditor_name
                }
                st.success("Decision saved!")
                
                # Auto-advance to next patient
                if current_idx < len(patients) - 1:
                    st.session_state.current_patient_idx += 1
                    st.rerun()
    
    # Export section
    st.markdown("---")
    st.subheader("📤 Export Results")
    
    if len(st.session_state.audit_results) > 0:
        export_df = pd.DataFrame(list(st.session_state.audit_results.values()))
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Audit Results (CSV)",
            data=csv,
            file_name=f"crc_audit_{audit_month.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
    else:
        st.info("Complete at least one review to enable export")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("CRC Audit Dashboard v1.0 | LayoutLMv3 Triage Model | UDS 2025")
