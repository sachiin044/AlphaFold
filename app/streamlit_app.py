import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import json
import random
from datetime import datetime

# Mock function to replace PyTorch model (for testing UI)
def mock_predict_sequence(sequence):
    """Mock prediction function for testing UI without PyTorch"""
    # Simulate processing time
    time.sleep(1)
    
    # Mock families (replace with your actual families)
    families = ['Kinase', 'Immunoglobulin', 'Transcription Factor', 'Enzyme', 'Membrane Protein']
    
    # Simple mock logic based on sequence length and composition
    if len(sequence) > 200:
        predicted_family = 'Enzyme'
        confidence = 0.92
    elif 'K' in sequence and 'R' in sequence:
        predicted_family = 'Kinase' 
        confidence = 0.87
    elif 'C' in sequence:
        predicted_family = 'Immunoglobulin'
        confidence = 0.83
    else:
        predicted_family = random.choice(families)
        confidence = random.uniform(0.7, 0.95)
    
    return predicted_family, confidence

def add_3d_background():
    """Add beautiful animated background with particles"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(-45deg, #2d1b69, #1a1a2e, #0f4c75, #16213e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Add floating particles */
    .stApp::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(255,255,255,0.1) 2px, transparent 2px),
            radial-gradient(circle at 75% 75%, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 100px 100px, 50px 50px;
        animation: particleFloat 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    
    @keyframes particleFloat {
        0% { transform: translateY(0px) translateX(0px); }
        100% { transform: translateY(-100px) translateX(50px); }
    }
    
    /* Ensure content is above background */
    .main > div {
        position: relative;
        z-index: 1;
    }
    
    /* Remove overlay for better visibility */
    .stApp > div:first-child {
        background: transparent;
        padding: 20px;
        margin: 20px;
    }
    
    /* Font styling */
    h1 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 30px;
    }
    
    h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    .stMarkdown, .stText, .stCaption {
        font-family: 'Inter', sans-serif !important;
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.1) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 10px !important;
        backdrop-filter: blur(10px) !important;
    }
    </style>
    """, unsafe_allow_html=True)

def validate_protein_sequence(sequence):
    """Validate protein sequence contains only valid amino acids"""
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.upper().replace(' ', '').replace('\n', '')
    
    invalid_chars = set(sequence) - valid_amino_acids
    if invalid_chars:
        return False, f"Invalid characters found: {', '.join(invalid_chars)}"
    
    if len(sequence) < 3:
        return False, "Sequence too short (minimum 3 amino acids)"
    
    return True, "Valid sequence"

def get_amino_acid_composition(sequence):
    """Calculate amino acid composition of the sequence"""
    composition = {}
    sequence = sequence.upper()
    for aa in sequence:
        composition[aa] = composition.get(aa, 0) + 1
    
    # Convert to percentages
    total = len(sequence)
    for aa in composition:
        composition[aa] = (composition[aa] / total) * 100
    
    return composition

# Demo sequences for testing
DEMO_SEQUENCES = {
    'Example 1 - Short Peptide': 'MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLRWVQTLSEQVQEELLSSQVTQELRALMDETAQ',
    'Example 2 - Medium Protein': 'MGSSHHHHHHSSGLVPRGSHMQIKLVLSRAAKCAGKVTRNQKQLLLKRQQYSDWDLSAQRAALLQVGGWAWLGTTTTTTTMNSQGTV',
    'Example 3 - Long Sequence': 'MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELNHPNIVKLLDVIHTENKLYLVFEFLHQDLKKFMDASALTGIPLPLIKSYLFQLLQGLAFCHSHRVLHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL'
}

# Page configuration
st.set_page_config(
    page_title="Protein Family Classifier",
    page_icon="🧬",
    layout="wide"
)

# Add the beautiful background
add_3d_background()

# Mock device info (since we're not using torch)
DEVICE = "cpu"
with st.sidebar:
    st.markdown("### 🖥️ System Info")
    st.markdown(f"**Device:** `{DEVICE}` (Demo Mode)")
    st.markdown(f"**Mode:** `Mock Predictions`")
    
    st.markdown("### 📊 About")
    st.markdown("This protein family classifier uses deep learning to predict protein families based on amino acid sequences.")
    
    st.markdown("### 🚀 Features")
    st.markdown("""
    - Real-time predictions
    - Confidence scoring
    - Amino acid composition analysis
    - Beautiful visualizations
    - Export results as JSON
    """)
    
    st.markdown("### ⚠️ Demo Mode")
    st.info("Currently running in demo mode with mock predictions. Install PyTorch for real model inference.")

# Main header
st.markdown("# 🧬 Protein Family Classifier")
st.markdown("### Advanced AI-powered protein structure analysis")

# Create tabs for better organization
tab1, tab2 = st.tabs(["🔬 Prediction", "📊 Analysis"])

with tab1:
    # Input section
    st.markdown("## 🧪 Input Protein Sequence")
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "FASTA File Upload", "Example Sequences"],
        horizontal=True
    )
    
    sequence = ""
    
    if input_method == "Text Input":
        sequence = st.text_area(
            "Protein Sequence",
            height=150,
            placeholder="Enter protein sequence here (e.g., MKVLWA...)",
            help="Enter a protein sequence using standard amino acid codes"
        )
        
        if sequence:
            is_valid, message = validate_protein_sequence(sequence)
            if not is_valid:
                st.error(f"❌ {message}")
                sequence = ""
            else:
                st.success("✅ Valid protein sequence!")
                
    elif input_method == "FASTA File Upload":
        uploaded_file = st.file_uploader(
            "Choose a FASTA file", 
            type=['fasta', 'fa', 'txt'],
            help="Upload a FASTA file containing your protein sequence"
        )
        
        if uploaded_file is not None:
            file_content = uploaded_file.read().decode('utf-8')
            
            # FASTA parsing
            lines = file_content.strip().split('\n')
            sequence_lines = []
            for line in lines:
                if not line.startswith('>'):
                    sequence_lines.append(line.strip())
            
            sequence = ''.join(sequence_lines)
            
            if sequence:
                st.text_area(
                    "Uploaded Sequence", 
                    value=sequence[:200] + ("..." if len(sequence) > 200 else ""),
                    height=100, 
                    disabled=True
                )
                
                is_valid, message = validate_protein_sequence(sequence)
                if not is_valid:
                    st.error(f"❌ {message}")
                    sequence = ""
                else:
                    st.success("✅ Valid sequence uploaded!")
                    
    else:  # Example Sequences
        example_name = st.selectbox("Choose an example sequence:", list(DEMO_SEQUENCES.keys()))
        sequence = DEMO_SEQUENCES[example_name]
        st.text_area(
            "Example Sequence",
            value=sequence[:200] + ("..." if len(sequence) > 200 else ""),
            height=100,
            disabled=True
        )

    # Prediction section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Predict Protein Family", type="primary", use_container_width=True):
            if not sequence:
                st.error("⚠️ Please enter a protein sequence before predicting.")
            else:
                # Show progress
                with st.spinner("🧠 Analyzing protein sequence with AI model..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)  # Simulate processing time
                        progress_bar.progress(i + 1)
                    
                    try:
                        # Get prediction from mock model
                        predicted_family, confidence = mock_predict_sequence(sequence)
                        
                        progress_bar.empty()
                        
                        # Display results in an attractive format
                        st.markdown("## 🎯 Prediction Results")
                        
                        # Create metric cards
                        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                        
                        with result_col1:
                            st.metric(
                                label="🏷️ Predicted Family",
                                value=predicted_family
                            )
                        
                        with result_col2:
                            st.metric(
                                label="🎯 Confidence",
                                value=f"{confidence*100:.1f}%"
                            )
                        
                        with result_col3:
                            st.metric(
                                label="📏 Sequence Length",
                                value=f"{len(sequence)} AA"
                            )
                        
                        with result_col4:
                            unique_aa = len(set(sequence.upper()))
                            st.metric(
                                label="🧪 Unique AAs",
                                value=f"{unique_aa}/20"
                            )
                        
                        # Confidence visualization
                        fig_confidence = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = confidence * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Prediction Confidence"},
                            delta = {'reference': 80},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 80], 'color': "gray"},
                                    {'range': [80, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_confidence.update_layout(height=300, showlegend=False)
                        
                        st.plotly_chart(fig_confidence, use_container_width=True)
                        
                        # Export results
                        result_data = {
                            "predicted_family": predicted_family,
                            "confidence": float(confidence),
                            "sequence": sequence,
                            "sequence_length": len(sequence),
                            "unique_amino_acids": unique_aa,
                            "timestamp": datetime.now().isoformat(),
                            "device_used": DEVICE,
                            "mode": "demo"
                        }
                        
                        st.download_button(
                            label="📥 Download Results (JSON)",
                            data=json.dumps(result_data, indent=2),
                            file_name=f"protein_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                        
                        # Show success message
                        st.balloons()
                        st.success(f"🎉 Successfully classified as **{predicted_family}** with {confidence*100:.1f}% confidence!")
                        
                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"❌ Prediction failed: {str(e)}")
                        st.info("💡 Please check your sequence and try again.")

with tab2:
    st.markdown("## 📊 Sequence Analysis")
    
    if 'sequence' in locals() and sequence:
        # Amino acid composition analysis
        composition = get_amino_acid_composition(sequence)
        
        if composition:
            # Create DataFrame for plotting
            comp_df = pd.DataFrame(list(composition.items()), columns=['Amino Acid', 'Percentage'])
            comp_df = comp_df.sort_values('Percentage', ascending=False)
            
            # Create composition bar chart
            fig_comp = px.bar(
                comp_df, 
                x='Amino Acid', 
                y='Percentage',
                title="Amino Acid Composition (%)",
                color='Percentage',
                color_continuous_scale='viridis'
            )
            fig_comp.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Additional sequence statistics
            st.markdown("### 📈 Sequence Statistics")
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                hydrophobic_aa = sum(composition.get(aa, 0) for aa in ['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V'])
                st.metric("Hydrophobic AAs", f"{hydrophobic_aa:.1f}%")
                
            with stat_col2:
                charged_aa = sum(composition.get(aa, 0) for aa in ['R', 'H', 'K', 'D', 'E'])
                st.metric("Charged AAs", f"{charged_aa:.1f}%")
                
            with stat_col3:
                polar_aa = sum(composition.get(aa, 0) for aa in ['S', 'T', 'N', 'Q'])
                st.metric("Polar AAs", f"{polar_aa:.1f}%")
    else:
        st.info("🔍 Enter a protein sequence in the Prediction tab to see detailed analysis here.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.6);'>"
    
    "</div>",
    unsafe_allow_html=True
)
