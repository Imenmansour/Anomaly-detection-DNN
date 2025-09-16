import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Temporarily remove seaborn to avoid numpy compatibility issues
# import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tensorflow.keras.models import load_model
import joblib
import warnings
import sys
import os
import traceback
warnings.filterwarnings('ignore')

# LangGraph and LLM imports
try:
    from langchain_community.llms import Ollama
    from langgraph.graph import StateGraph, END
    from typing import TypedDict, List, Dict, Any
    import json
    LANGGRAPH_AVAILABLE = True
    print("✅ LangGraph and Ollama imports successful!")
except ImportError as e:
    print(f"⚠️ LangGraph/Ollama not available: {e}")
    LANGGRAPH_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="🎯 Real-Time Anomaly Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .decision-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-weight: bold;
        text-align: center;
        font-size: 1.2rem;
    }
    .autonomous {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .monitoring {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .oversight {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .stability-metric {
        display: inline-block;
        margin: 5px;
        padding: 10px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .stable {
        background-color: #d4edda;
        color: #155724;
    }
    .unstable {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Import SafeML modules
import sys
import os
sys.path.insert(0, r'C:\Users\G800613RTS\Desktop\Anomaly\SafeML\Implementation_in_Python')

try:
    from CVM_Distance import CVM_Dist as Cramer_Von_Mises_Dist
    from Anderson_Darling_Distance import Anderson_Darling_Dist
    from Kolmogorov_Smirnov_Distance import Kolmogorov_Smirnov_Dist
    from KuiperDistance import Kuiper_Dist
    from WassersteinDistance import Wasserstein_Dist
    from DTS_Distance import DTS_Dist
    print("✅ SafeML modules imported successfully!")
except ImportError as e:
    st.error(f"❌ Failed to import SafeML modules: {e}")
    st.stop()

# Load the original training data (X_train, y_train) that was used to create Class0.xlsx and Class1.xlsx
@st.cache_data
def load_original_training_data():
    """Load the original training data from the SafeML6.ipynb process"""
    try:
        # Load the dataset (same as in SafeML6.ipynb)
        df = pd.read_csv('training_data.csv')
        
        # Apply label encoding
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        df['Label'] = encoder.fit_transform(df['Label'])
        
        # Handle NaN and inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df = df.astype(int)
        
        # Get features and labels
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Select the same 10 features used in training
        new_columns = ['dst_port','flow_duration', 'fwd_pkt_len_max', 'fwd_pkt_len_mean',
                       'pkt_len_mean', 'pkt_len_std', 'fwd_iat_tot', 'syn_flag_cnt',
                       'pkt_size_avg', 'fwd_seg_size_avg']
        
        X_selected = X[new_columns].values
        y_selected = y.values
        
        # Split the data using the same parameters as in SafeML6.ipynb
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_selected, test_size=0.33, random_state=42, stratify=y_selected
        )
        
        return X_train, X_test, y_train, y_test, new_columns
        
    except Exception as e:
        st.error(f"❌ Error loading original training data: {e}")
        return None, None, None, None, None

def get_X_train_and_test_data_for_given_label(labels, label_index, pred_y, X_train, X_test, y_train, y_test, feature_columns):
    """
    Get training and test data for a specific label/class.
    This matches the function from SafeML6.ipynb
    """
    # Since X_train, X_test, y_train are numpy arrays, use boolean indexing
    train_mask = y_train == labels[label_index]
    test_mask = pred_y == labels[label_index]
    
    # Filter the data using boolean masks
    X_train_loc_for_label = X_train[train_mask]
    X_test_loc_for_label = X_test[test_mask]
    
    # Convert to pandas DataFrames with the correct feature names
    X_train_loc_for_label = pd.DataFrame(X_train_loc_for_label, columns=feature_columns)
    X_test_loc_for_label = pd.DataFrame(X_test_loc_for_label, columns=feature_columns)
    
    return X_train_loc_for_label, X_test_loc_for_label

def get_statistical_dist_measures_for_class_result(accuracy, X_train_L, X_test_L):
    """Calculate statistical distance measures for a specific class result including accuracy."""
    if len(X_train_L) == 0 or len(X_test_L) == 0:
        print(f"Warning: Empty dataset detected. Train size: {len(X_train_L)}, Test size: {len(X_test_L)}")
        return {'Accuracy': accuracy,
                'Anderson_Darling_dist': np.nan,
                'CVM_dist': np.nan,
                'DTS_dist': np.nan,
                'Kolmogorov_Smirnov_dist': np.nan,
                'Kuiper_dist': np.nan,
                'Wasserstein_dist': np.nan}
    
    num_of_features = len(X_train_L.columns)
    
    # Initialize arrays for distance measures
    CVM_distances = np.zeros(num_of_features)
    Anderson_Darling_distances = np.zeros(num_of_features)
    Kolmogorov_Smirnov_distances = np.zeros(num_of_features)
    Kuiper_distances = np.zeros(num_of_features)
    Wasserstein_distances = np.zeros(num_of_features)
    DTS_distances = np.zeros(num_of_features)

    for i in range(num_of_features):
        CVM_distances[i] = Cramer_Von_Mises_Dist(X_train_L.iloc[:, i], X_test_L.iloc[:, i])
        Anderson_Darling_distances[i] = Anderson_Darling_Dist(X_train_L.iloc[:, i], X_test_L.iloc[:, i])
        Kolmogorov_Smirnov_distances[i] = Kolmogorov_Smirnov_Dist(X_train_L.iloc[:, i], X_test_L.iloc[:, i])
        Kuiper_distances[i] = Kuiper_Dist(X_train_L.iloc[:, i], X_test_L.iloc[:, i])
        Wasserstein_distances[i] = Wasserstein_Dist(X_train_L.iloc[:, i], X_test_L.iloc[:, i])
        DTS_distances[i] = DTS_Dist(X_train_L.iloc[:, i], X_test_L.iloc[:, i])
        
    # Calculate means
    CVM_distance = np.mean(CVM_distances, dtype=np.float64)
    Anderson_Darling_distance = np.mean(Anderson_Darling_distances, dtype=np.float64)
    Kolmogorov_Smirnov_distance = np.mean(Kolmogorov_Smirnov_distances, dtype=np.float64)
    Kuiper_distance = np.mean(Kuiper_distances, dtype=np.float64)
    Wasserstein_distance = np.mean(Wasserstein_distances, dtype=np.float64)
    DTS_distance = np.mean(DTS_distances, dtype=np.float64)
    
    return {'Accuracy': accuracy,
            'Anderson_Darling_dist': Anderson_Darling_distance,
            'CVM_dist': CVM_distance,
            'DTS_dist': DTS_distance,
            'Kolmogorov_Smirnov_dist': Kolmogorov_Smirnov_distance,
            'Kuiper_dist': Kuiper_distance,
            'Wasserstein_dist': Wasserstein_distance}

def analyze_class_distances(train_distances, realtime_distances, distance_measures, class_name, class_accuracy):
    """Analyze distances for a specific class and calculate similarity."""
    print(f"\n📏 {class_name} Distance Analysis:")
    print(f"{'Measure':<25} {'Training':<12} {'Real-time':<12} {'Ratio':<8} {'Similarity %':<12}")
    print("-" * 80)
    
    similarities = []
    valid_measures = 0
    
    for measure in distance_measures:
        try:
            if measure in train_distances.index and measure in realtime_distances:
                train_val = train_distances[measure]
                real_val = realtime_distances[measure]
                
                if pd.isna(train_val) or train_val == 0:
                    continue
                
                ratio = real_val / train_val
                
                # Calculate similarity based on ratio
                if ratio <= 1.0:
                    similarity = 100 * (2 - ratio)
                    if similarity > 100:
                        similarity = 100
                else:
                    similarity = 100 / ratio
                
                similarities.append(similarity)
                valid_measures += 1
                
                print(f"{measure:<25} {train_val:<12.6f} {real_val:<12.6f} {ratio:<8.2f} {similarity:<12.2f}")
                
        except Exception as e:
            print(f"Error processing {measure}: {e}")
            continue
    
    avg_similarity = np.mean(similarities) if similarities else 0
    print(f"\nAverage similarity for {class_name}: {avg_similarity:.2f}%")
    
    return {
        'similarities': similarities,
        'average_similarity': avg_similarity,
        'valid_measures': valid_measures,
        'class_accuracy': class_accuracy
    }

def calculate_weighted_similarity(class0_analysis, class1_analysis):
    """Calculate weighted overall similarity based on both classes."""
    total_weighted_similarity = 0
    total_weight = 0
    
    for analysis in [class0_analysis, class1_analysis]:
        if analysis['valid_measures'] > 0:
            weight = analysis['valid_measures'] * analysis['class_accuracy']
            total_weighted_similarity += analysis['average_similarity'] * weight
            total_weight += weight
    
    if total_weight > 0:
        overall_similarity = total_weighted_similarity / total_weight
    else:
        overall_similarity = 50.0
    
    return overall_similarity

def estimate_final_accuracy(similarity_score, best_training_accuracy):
    """Estimate final accuracy based on similarity score."""
    if similarity_score >= 95.0:
        retention_factor = 0.98
        confidence = "EXCELLENT"
    elif similarity_score >= 90.0:
        retention_factor = 0.95
        confidence = "VERY_HIGH"
    elif similarity_score >= 85.0:
        retention_factor = 0.90
        confidence = "HIGH"
    elif similarity_score >= 80.0:
        retention_factor = 0.85
        confidence = "GOOD"
    elif similarity_score >= 75.0:
        retention_factor = 0.80
        confidence = "MODERATE"
    elif similarity_score >= 70.0:
        retention_factor = 0.75
        confidence = "FAIR"
    else:
        retention_factor = 0.65
        confidence = "LOW"
    
    estimated_accuracy = best_training_accuracy * retention_factor
    
    print(f"\nConfidence level: {confidence}")
    print(f"Retention factor: {retention_factor:.2f}")
    
    return estimated_accuracy

def estimate_accuracy_from_distance_ratios(training_data_path_class0, training_data_path_class1, 
                                         realtime_distances_class0, realtime_distances_class1):
    """Estimate accuracy based on distance ratios between training and real-time data."""
    try:
        print("📊 Loading training data and calculating accuracy estimation...")
        
        # Load training data
        df_class0 = pd.read_excel(training_data_path_class0)
        df_class1 = pd.read_excel(training_data_path_class1)
        
        # Get best accuracy from each class
        if 'Accuracy' in df_class0.columns:
            best_accuracy_class0 = df_class0['Accuracy'].max()
            best_idx_class0 = df_class0['Accuracy'].idxmax()
        else:
            best_accuracy_class0 = df_class0.iloc[0, 0]
            best_idx_class0 = 0
            
        if 'Accuracy' in df_class1.columns:
            best_accuracy_class1 = df_class1['Accuracy'].max()
            best_idx_class1 = df_class1['Accuracy'].idxmax()
        else:
            best_accuracy_class1 = df_class1.iloc[0, 0]
            best_idx_class1 = 0
        
        # Get best training distances
        train_distances_class0 = df_class0.iloc[best_idx_class0]
        train_distances_class1 = df_class1.iloc[best_idx_class1]
        
        print(f"Best Class 0 accuracy: {best_accuracy_class0:.4f} ({best_accuracy_class0*100:.2f}%)")
        print(f"Best Class 1 accuracy: {best_accuracy_class1:.4f} ({best_accuracy_class1*100:.2f}%)")
        
        # Distance measures
        distance_measures = [
            'Anderson_Darling_dist', 'CVM_dist', 'DTS_dist',
            'Kolmogorov_Smirnov_dist', 'Kuiper_dist', 'Wasserstein_dist'
        ]
        
        # Calculate ratios and similarities for each class
        class0_analysis = analyze_class_distances(
            train_distances_class0, realtime_distances_class0, 
            distance_measures, "Class 0", best_accuracy_class0
        )
        
        class1_analysis = analyze_class_distances(
            train_distances_class1, realtime_distances_class1, 
            distance_measures, "Class 1", best_accuracy_class1
        )
        
        # Calculate overall similarity score
        overall_similarity = calculate_weighted_similarity(class0_analysis, class1_analysis)
        
        # Estimate final accuracy
        overall_best_accuracy = max(best_accuracy_class0, best_accuracy_class1)
        estimated_accuracy = estimate_final_accuracy(overall_similarity, overall_best_accuracy)
        
        # Prepare detailed results
        detailed_analysis = {
            'class0_analysis': class0_analysis,
            'class1_analysis': class1_analysis,
            'overall_similarity': overall_similarity,
            'best_accuracy_class0': best_accuracy_class0,
            'best_accuracy_class1': best_accuracy_class1,
            'overall_best_accuracy': overall_best_accuracy,
            'estimated_accuracy': estimated_accuracy
        }
        
        print(f"\n🎯 Final Results:")
        print(f"Overall similarity score: {overall_similarity:.2f}%")
        print(f"Estimated accuracy: {estimated_accuracy:.4f} ({estimated_accuracy*100:.2f}%)")
        
        return estimated_accuracy, detailed_analysis
        
    except Exception as e:
        print(f"❌ Error in accuracy estimation: {e}")
        import traceback
        traceback.print_exc()
        return 0.5, {}

def safeml_confidence_assessment_both_classes(best_training_accuracy, estimated_accuracy, distance_analysis):
    """Assess confidence using SafeML methodology based on BOTH Class 0 and Class 1 statistical distances."""
    accuracy_difference = abs(best_training_accuracy - estimated_accuracy)
    overall_similarity = distance_analysis.get('overall_similarity', 0.0)
    overall_diff_percent = 100 - overall_similarity
    
    print(f"\n" + "="*80)
    print(f"SafeML Confidence Assessment (Both Classes):")
    print(f"   Best training accuracy: {best_training_accuracy:.4f} ({best_training_accuracy*100:.2f}%)")
    print(f"   Estimated accuracy: {estimated_accuracy:.4f} ({estimated_accuracy*100:.2f}%)")
    print(f"   Accuracy difference: {accuracy_difference:.4f} ({accuracy_difference*100:.2f}%)")
    print(f"   Overall similarity score: {overall_similarity:.2f}%")
    print(f"   Overall distance difference: {overall_diff_percent:.2f}%")
    
    # Multi-criteria decision making
    decision_factors = []
    
    # Factor 1: Overall similarity threshold
    if overall_similarity >= 90.0:
        decision_factors.append(("High Similarity", "AUTONOMOUS", "HIGH"))
    elif overall_similarity >= 75.0:
        decision_factors.append(("Moderate Similarity", "AUTONOMOUS", "MODERATE"))
    else:
        decision_factors.append(("Low Similarity", "HUMAN_INTERVENTION", "LOW"))
    
    # Factor 2: Accuracy drop threshold
    accuracy_drop_percent = (accuracy_difference / best_training_accuracy) * 100
    if accuracy_drop_percent <= 5.0:
        decision_factors.append(("Small Accuracy Drop", "AUTONOMOUS", "HIGH"))
    elif accuracy_drop_percent <= 15.0:
        decision_factors.append(("Moderate Accuracy Drop", "AUTONOMOUS", "MODERATE"))
    else:
        decision_factors.append(("Large Accuracy Drop", "HUMAN_INTERVENTION", "LOW"))
    
    # Factor 3: Class-specific analysis
    class0_analysis = distance_analysis.get('class0_analysis', {})
    class1_analysis = distance_analysis.get('class1_analysis', {})
    
    # Analyze Class 0
    if class0_analysis and class0_analysis.get('valid_measures', 0) > 0:
        avg_similarity_class0 = class0_analysis['average_similarity']
        if avg_similarity_class0 >= 90.0:
            decision_factors.append(("Class 0 Analysis", "AUTONOMOUS", "HIGH"))
        elif avg_similarity_class0 >= 75.0:
            decision_factors.append(("Class 0 Analysis", "AUTONOMOUS", "MODERATE"))
        else:
            decision_factors.append(("Class 0 Analysis", "HUMAN_INTERVENTION", "LOW"))
    
    # Analyze Class 1
    if class1_analysis and class1_analysis.get('valid_measures', 0) > 0:
        avg_similarity_class1 = class1_analysis['average_similarity']
        if avg_similarity_class1 >= 90.0:
            decision_factors.append(("Class 1 Analysis", "AUTONOMOUS", "HIGH"))
        elif avg_similarity_class1 >= 75.0:
            decision_factors.append(("Class 1 Analysis", "AUTONOMOUS", "MODERATE"))
        else:
            decision_factors.append(("Class 1 Analysis", "HUMAN_INTERVENTION", "LOW"))
    
    print(f"\nDecision Factors Analysis:")
    for factor_name, factor_decision, factor_confidence in decision_factors:
        print(f"   {factor_name:<25}: {factor_decision:<20} ({factor_confidence} confidence)")
    
    # Final decision based on majority vote and severity
    autonomous_votes = sum(1 for _, decision, _ in decision_factors if decision == "AUTONOMOUS")
    intervention_votes = sum(1 for _, decision, _ in decision_factors if decision == "HUMAN_INTERVENTION")
    
    # High confidence votes have more weight
    high_conf_autonomous = sum(1 for _, decision, conf in decision_factors 
                              if decision == "AUTONOMOUS" and conf == "HIGH")
    high_conf_intervention = sum(1 for _, decision, conf in decision_factors 
                                if decision == "HUMAN_INTERVENTION" and conf == "HIGH")
    
    # Conservative approach: if any high-confidence intervention vote, lean towards intervention
    if high_conf_intervention > 0:
        final_decision = "HUMAN_INTERVENTION"
        final_confidence = "LOW"
        primary_reason = "High-confidence factors indicate potential issues"
    elif high_conf_autonomous >= 2:
        final_decision = "AUTONOMOUS"
        final_confidence = "HIGH"
        primary_reason = "Multiple high-confidence factors support autonomous operation"
    elif autonomous_votes > intervention_votes:
        final_decision = "AUTONOMOUS"
        final_confidence = "MODERATE"
        primary_reason = "Majority of factors support autonomous operation"
    else:
        final_decision = "HUMAN_INTERVENTION"
        final_confidence = "LOW"
        primary_reason = "Majority of factors indicate potential issues"
    
    # Determine actions and messages
    if final_decision == "AUTONOMOUS":
        if final_confidence == "HIGH":
            message = "All indicators support autonomous operation - System highly reliable"
            action = "🟢 AUTONOMOUS OPERATION"
        else:
            message = "System can operate autonomously with increased monitoring"
            action = "🟡 ENHANCED MONITORING"
    else:
        message = "Multiple indicators suggest reliability issues - Human intervention required"
        action = "🔴 HUMAN OVERSIGHT REQUIRED"
    
    print(f"\n🎯 Final SafeML Decision:")
    print(f"   Decision: {final_decision}")
    print(f"   Confidence Level: {final_confidence}")
    print(f"   Primary Reason: {primary_reason}")
    print(f"   Message: {message}")
    print(f"   Recommended Action: {action}")
    
    return final_decision, final_confidence, accuracy_difference

def calculate_stability_analysis(realtime_distances_class0, realtime_distances_class1, 
                               training_data_path_class0, training_data_path_class1):
    """Calculate stability analysis based on actual distance ratios"""
    try:
        # Load training data
        df_class0 = pd.read_excel(training_data_path_class0)
        df_class1 = pd.read_excel(training_data_path_class1)
        
        # Get best training distances
        if 'Accuracy' in df_class0.columns:
            best_idx_class0 = df_class0['Accuracy'].idxmax()
        else:
            best_idx_class0 = 0
            
        if 'Accuracy' in df_class1.columns:
            best_idx_class1 = df_class1['Accuracy'].idxmax()
        else:
            best_idx_class1 = 0
        
        train_distances_class0 = df_class0.iloc[best_idx_class0]
        train_distances_class1 = df_class1.iloc[best_idx_class1]
        
        # Distance measures
        distance_measures = [
            'Anderson_Darling_dist', 'CVM_dist', 'DTS_dist',
            'Kolmogorov_Smirnov_dist', 'Kuiper_dist', 'Wasserstein_dist'
        ]
        
        # Calculate stability for Class 0
        class0_measure_details = []
        class0_stable_count = 0
        
        for measure in distance_measures:
            if measure in train_distances_class0.index and measure in realtime_distances_class0:
                train_val = train_distances_class0[measure]
                real_val = realtime_distances_class0[measure]
                
                if pd.isna(train_val) or train_val == 0:
                    ratio = float('inf')
                else:
                    ratio = real_val / train_val
                
                # FIXED: Consider stable if ratio is <= 1.0 (real-time distance is smaller or equal)
                # This means the distributions are similar or closer than during training
                is_stable = ratio <= 1.0
                status = '✅' if is_stable else '❌'
                
                if is_stable:
                    class0_stable_count += 1
                
                class0_measure_details.append({
                    'status': status,
                    'measure': measure.replace('_dist', ''),
                    'ratio': ratio
                })
        
        # Calculate stability for Class 1
        class1_measure_details = []
        class1_stable_count = 0
        
        for measure in distance_measures:
            if measure in train_distances_class1.index and measure in realtime_distances_class1:
                train_val = train_distances_class1[measure]
                real_val = realtime_distances_class1[measure]
                
                if pd.isna(train_val) or train_val == 0:
                    ratio = float('inf')
                else:
                    ratio = real_val / train_val
                
                # FIXED: Consider stable if ratio is <= 1.0 (real-time distance is smaller or equal)
                is_stable = ratio <= 1.0
                status = '✅' if is_stable else '❌'
                
                if is_stable:
                    class1_stable_count += 1
                
                class1_measure_details.append({
                    'status': status,
                    'measure': measure.replace('_dist', ''),
                    'ratio': ratio
                })
        
        return {
            'Class0': {
                'measure_details': class0_measure_details,
                'stable_count': class0_stable_count
            },
            'Class1': {
                'measure_details': class1_measure_details,
                'stable_count': class1_stable_count
            }
        }
        
    except Exception as e:
        print(f"❌ Error calculating stability analysis: {e}")
        return {
            'Class0': {'measure_details': [], 'stable_count': 0},
            'Class1': {'measure_details': [], 'stable_count': 0}
        }

# Cyber Attack Analysis Agent using LangGraph
class CyberAttackState(TypedDict):
    """State for the cyber attack analysis workflow"""
    original_data: pd.DataFrame
    chunk_reports: List[Dict[str, Any]]
    current_chunk: int
    total_chunks: int
    final_report: Dict[str, Any]
    analysis_complete: bool

class CyberAttackAnalysisAgent:
    """LangGraph-based agent for analyzing network traffic for cyber attacks"""
    
    def __init__(self):
        if not LANGGRAPH_AVAILABLE:
            self.available = False
            return
            
        try:
            # Initialize Ollama LLM
            self.llm = Ollama(
                model="qwen2.5:1.5b",
                temperature=0.7,
                base_url="http://localhost:11434"  # Default Ollama URL
            )
            self.available = True
            
            # Build the workflow graph
            self.workflow = self._build_workflow()
            
        except Exception as e:
            print(f"❌ Error initializing Ollama: {e}")
            self.available = False
    
    def _build_workflow(self):
        """Build the LangGraph workflow for cyber attack analysis"""
        # Create the graph
        workflow = StateGraph(CyberAttackState)
        
        # Add nodes
        workflow.add_node("chunk_analyzer", self._analyze_chunk)
        workflow.add_node("aggregator", self._aggregate_reports)
        
        # Add edges
        workflow.add_edge("chunk_analyzer", "aggregator")
        workflow.add_edge("aggregator", END)
        
        # Set entry point
        workflow.set_entry_point("chunk_analyzer")
        
        return workflow.compile()
    
    def _create_chunk_analysis_prompt(self, chunk_data: pd.DataFrame, chunk_num: int, total_chunks: int) -> str:
        """Create prompt for analyzing a chunk of network traffic data"""
        
        # Get basic statistics
        stats = {
            'total_flows': len(chunk_data),
            'unique_src_ips': chunk_data['src_ip'].nunique() if 'src_ip' in chunk_data.columns else 0,
            'unique_dst_ips': chunk_data['dst_ip'].nunique() if 'dst_ip' in chunk_data.columns else 0,
            'protocols': chunk_data['protocol'].value_counts().to_dict() if 'protocol' in chunk_data.columns else {},
            'avg_flow_duration': chunk_data['flow_duration'].mean() if 'flow_duration' in chunk_data.columns else 0,
            'max_flow_duration': chunk_data['flow_duration'].max() if 'flow_duration' in chunk_data.columns else 0,
            'avg_pkt_size': chunk_data['pkt_size_avg'].mean() if 'pkt_size_avg' in chunk_data.columns else 0,
            'syn_flags': chunk_data['syn_flag_cnt'].sum() if 'syn_flag_cnt' in chunk_data.columns else 0,
            'rst_flags': chunk_data['rst_flag_cnt'].sum() if 'rst_flag_cnt' in chunk_data.columns else 0,
        }
        
        # Get actual data rows for detailed analysis
        sample_data = chunk_data.head(20).to_string(index=False) if len(chunk_data) > 0 else "No data available"
        
        prompt = f"""
You are a cybersecurity expert analyzing network traffic data for potential cyber attacks, specifically DDoS attacks and other suspicious activities.

ANALYZING CHUNK {chunk_num}/{total_chunks}:

Network Traffic Statistics:
- Total flows in this chunk: {stats['total_flows']}
- Unique source IPs: {stats['unique_src_ips']}
- Unique destination IPs: {stats['unique_dst_ips']}
- Protocol distribution: {stats['protocols']}
- Average flow duration: {stats['avg_flow_duration']:.2f} seconds
- Maximum flow duration: {stats['max_flow_duration']:.2f} seconds
- Average packet size: {stats['avg_pkt_size']:.2f} bytes
- Total SYN flags: {stats['syn_flags']}
- Total RST flags: {stats['rst_flags']}

ACTUAL DATA SAMPLE (First 20 rows):
{sample_data}

Please analyze this network traffic data and provide a conversational response about any cyber security threats you detect. 

Talk naturally about:
- What threat level you see (LOW/MEDIUM/HIGH) and why
- Any attack patterns you detect (DDoS, port scanning, etc.)
- Suspicious IP addresses or behaviors
- Key indicators that concern you
- Your confidence in the analysis
- Recommendations for security teams

Respond in a natural, conversational way as if you're briefing a security team. Don't use JSON format - just talk through your findings like a cybersecurity expert would.
"""
        return prompt
    
    def _analyze_chunk(self, state: CyberAttackState) -> Dict[str, Any]:
        """Analyze chunks of network traffic data using conversational approach"""
        try:
            data = state["original_data"]
            
            # Use only the first 100 rows for analysis
            sample_size = min(20, len(data))
            data_sample = data.head(sample_size)
            
            # Process in 20-row chunks
            chunk_size = 20
            total_chunks = (len(data_sample) + chunk_size - 1) // chunk_size
            
            chunk_responses = []
            
            print(f"🔍 Analyzing {sample_size} rows in {total_chunks} chunks of {chunk_size} rows each...")
            
            # Process each chunk
            for chunk_idx in range(total_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(data_sample))
                chunk_data = data_sample.iloc[start_idx:end_idx]
                
                try:
                    # Create analysis prompt
                    prompt = self._create_chunk_analysis_prompt(chunk_data, chunk_idx + 1, total_chunks)
                    
                    # Get LLM response
                    response = self.llm.invoke(prompt)
                    
                    # Store the raw conversational response
                    chunk_response = {
                        "chunk_number": chunk_idx + 1,
                        "rows_analyzed": f"{start_idx + 1}-{end_idx}",
                        "response": response if isinstance(response, str) else str(response),
                        "data_size": len(chunk_data)
                    }
                    
                    chunk_responses.append(chunk_response)
                    print(f"✅ Completed chunk {chunk_idx + 1}/{total_chunks}")
                    
                except Exception as e:
                    # Handle individual chunk errors gracefully
                    error_response = {
                        "chunk_number": chunk_idx + 1,
                        "rows_analyzed": f"{start_idx + 1}-{end_idx}",
                        "response": f"Analysis error for this chunk: {str(e)}. Please review these {len(chunk_data)} rows manually for potential security threats.",
                        "data_size": len(chunk_data),
                        "error": True
                    }
                    chunk_responses.append(error_response)
                    print(f"⚠️ Error in chunk {chunk_idx + 1}: {e}")
            
            return {
                **state,
                "chunk_reports": chunk_responses,
                "total_chunks": total_chunks,
                "current_chunk": total_chunks,
                "sample_size": sample_size
            }
            
        except Exception as e:
            print(f"❌ Error in chunk analysis: {e}")
            return {
                **state,
                "chunk_reports": [{
                    "chunk_number": 1,
                    "response": f"Critical error in analysis system: {str(e)}. The network traffic analysis could not be completed. Please check the system and try again.",
                    "error": True
                }],
                "analysis_complete": True
            }
    
    def _aggregate_reports(self, state: CyberAttackState) -> Dict[str, Any]:
        """Aggregate all chunk reports into a final assessment"""
        try:
            chunk_reports = state.get("chunk_reports", [])
            
            if not chunk_reports:
                final_report = {
                    "overall_threat_level": "UNKNOWN",
                    "total_chunks_analyzed": 0,
                    "attack_summary": "No analysis performed - no chunk reports available",
                    "confidence_score": 0.0,
                    "recommendations": ["System check required"],
                    "conversational_response": "I haven't received any network traffic analysis chunks to review. This could indicate a system issue or that no data was processed. I recommend checking the data pipeline and ensuring network traffic is being captured properly."
                }
            else:
                # Extract patterns from conversational reports
                all_responses = []
                threat_indicators = []
                confidence_scores = []
                
                for report in chunk_reports:
                    response_text = report.get("conversational_response", "")
                    all_responses.append(response_text)
                    
                    # Extract simple metrics from conversational text
                    if any(word in response_text.lower() for word in ["high", "critical", "severe", "attack", "malicious"]):
                        threat_indicators.append("HIGH")
                        confidence_scores.append(0.8)
                    elif any(word in response_text.lower() for word in ["medium", "suspicious", "unusual", "anomaly"]):
                        threat_indicators.append("MEDIUM") 
                        confidence_scores.append(0.6)
                    else:
                        threat_indicators.append("LOW")
                        confidence_scores.append(0.4)
                
                # Determine overall threat level
                high_threats = threat_indicators.count("HIGH")
                medium_threats = threat_indicators.count("MEDIUM")
                
                if high_threats > len(chunk_reports) * 0.3:  # 30% high threats
                    overall_threat = "HIGH"
                elif medium_threats + high_threats > len(chunk_reports) * 0.5:  # 50% medium+ threats
                    overall_threat = "MEDIUM"
                else:
                    overall_threat = "LOW"
                
                # Create conversational aggregation prompt
                responses_summary = "\n\n".join([f"Chunk {i+1}: {resp[:200]}..." for i, resp in enumerate(all_responses)])
                
                aggregation_prompt = f"""
You are a senior cybersecurity analyst providing a final comprehensive assessment of network traffic analysis.

I've analyzed {len(chunk_reports)} chunks of network traffic data. Here's a summary of my findings from each chunk:

{responses_summary}

ANALYSIS METRICS:
- Total chunks analyzed: {len(chunk_reports)}
- High-risk indicators found: {high_threats}
- Medium-risk indicators found: {medium_threats}
- Low-risk indicators found: {threat_indicators.count("LOW")}
- Overall threat assessment: {overall_threat}

Please provide a comprehensive final security assessment in a conversational format. Include:
1. Overall threat level and confidence
2. Summary of key security findings
3. Most critical risks identified
4. Immediate actions recommended
5. Long-term security recommendations

Speak as a cybersecurity expert explaining the situation to a technical team.
"""
                
                try:
                    final_response = self.llm.invoke(aggregation_prompt)
                    
                    # Create structured report from conversational response
                    final_report = {
                        "overall_threat_level": overall_threat,
                        "total_chunks_analyzed": len(chunk_reports),
                        "attack_summary": f"Comprehensive analysis of {len(chunk_reports)} network traffic chunks completed",
                        "confidence_score": float(np.mean(confidence_scores)) if confidence_scores else 0.5,
                        "threat_distribution": {
                            "high": high_threats,
                            "medium": medium_threats, 
                            "low": threat_indicators.count("LOW")
                        },
                        "conversational_response": final_response,
                        "chunk_summaries": all_responses,
                        "recommendations": [
                            "Review detailed analysis findings",
                            "Monitor high-risk indicators",
                            "Implement recommended security measures"
                        ]
                    }
                        
                except Exception as e:
                    # Error in final aggregation - provide fallback
                    final_report = {
                        "overall_threat_level": overall_threat,
                        "total_chunks_analyzed": len(chunk_reports),
                        "attack_summary": f"Analyzed {len(chunk_reports)} chunks with aggregation processing error",
                        "confidence_score": float(np.mean(confidence_scores)) if confidence_scores else 0.5,
                        "conversational_response": f"I've completed analysis of {len(chunk_reports)} network traffic chunks. While I encountered a technical issue during final aggregation, I was able to process the individual chunks. Based on the chunk analysis, I detected {high_threats} high-risk patterns and {medium_threats} medium-risk patterns. The overall threat level appears to be {overall_threat}. I recommend reviewing the individual chunk analyses for detailed findings.",
                        "recommendations": ["Manual review of chunk reports", "System diagnostics", "Technical support consultation"],
                        "processing_error": str(e)
                    }
            
            return {
                **state,
                "final_report": final_report,
                "analysis_complete": True
            }
            
        except Exception as e:
            print(f"❌ Error in aggregation: {e}")
            return {
                **state,
                "final_report": {
                    "overall_threat_level": "HIGH",
                    "attack_summary": "Critical system error during analysis aggregation",
                    "confidence_score": 0.1,
                    "immediate_actions": ["System maintenance required"],
                    "error": str(e)
                },
                "analysis_complete": True
            }
    
    def analyze_network_traffic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main entry point for network traffic analysis"""
        if not self.available:
            return {
                "error": "Cyber attack analysis agent not available. Please install langchain-community and ensure Ollama is running.",
                "overall_threat_level": "UNKNOWN",
                "attack_summary": "Analysis system unavailable"
            }
        
        try:
            # Initialize state
            initial_state = CyberAttackState(
                original_data=data,
                chunk_reports=[],
                current_chunk=0,
                total_chunks=0,
                final_report={},
                analysis_complete=False
            )
            
            # Run the workflow
            result = self.workflow.invoke(initial_state)
            
            return result.get("final_report", {
                "overall_threat_level": "UNKNOWN",
                "attack_summary": "Analysis failed to complete",
                "confidence_score": 0.0
            })
            
        except Exception as e:
            print(f"❌ Error in network traffic analysis: {e}")
            return {
                "error": str(e),
                "overall_threat_level": "HIGH",
                "attack_summary": "Critical analysis system error",
                "confidence_score": 0.1,
                "immediate_actions": ["System diagnostics required"]
            }
    
    def answer_user_question(self, question: str, system_data: dict = None) -> str:
        """Answer user questions about cybersecurity using the local LLM"""
        if not self.available:
            return "❌ AI Assistant not available. Please ensure Ollama is running with qwen2.5:1.5b model."
        
        try:
            # Create context from system data if available
            context = ""
            if system_data:
                context = f"""
Current System Status:
- Total samples processed: {system_data.get('total_samples', 'N/A')}
- Anomalies detected: {system_data.get('anomalies_detected', 'N/A')}
- System decision: {system_data.get('system_decision', 'N/A')}
- Risk level: {system_data.get('risk_level', 'N/A')}
"""
            
            # Create cybersecurity expert prompt
            prompt = f"""You are an expert cybersecurity analyst and network security specialist. Answer the user's question with accurate, helpful information about cybersecurity, network security, DDoS attacks, anomaly detection, or related topics.

{context}

User Question: {question}

Please provide a clear, informative response as a cybersecurity expert. If the question is about the current system status, use the context provided above. Focus on practical, actionable advice."""

            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            return response
            
        except Exception as e:
            return f"⚠️ Error processing your question: {str(e)}. Please try again."


class RealTimeAnomalyDetector:
    def __init__(self, data_path="real.csv", model_path="best_dnn_model_final.h5", 
                 scaler_path="best_scaler_final.pkl", class0_path="Class0.xlsx", 
                 class1_path="Class1.xlsx"):
        self.data_path = data_path
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.class0_path = class0_path
        self.class1_path = class1_path
        self.model = None
        self.scaler = None
        self.training_data = None
        self.new_columns = ['dst_port','flow_duration', 'fwd_pkt_len_max', 'fwd_pkt_len_mean',
                           'pkt_len_mean', 'pkt_len_std', 'fwd_iat_tot', 'syn_flag_cnt',
                           'pkt_size_avg', 'fwd_seg_size_avg']
        
        # Load original training data
        self.X_train_orig, self.X_test_orig, self.y_train_orig, self.y_test_orig, self.feature_columns = load_original_training_data()
    
    def load_models(self):
        """Load trained model and scaler"""
        try:
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return False
    
    def load_training_distances(self):
        """Load training distance data"""
        try:
            if os.path.exists(self.class0_path) and os.path.exists(self.class1_path):
                
                return True
            else:
                st.error(f"❌ Training distance files not found")
                return False
        except Exception as e:
            st.error(f"❌ Error checking training files: {e}")
            return False
    
    def process_real_time_data(self, real_data):
        """Process real-time data and make predictions"""
        try:
            print(f"📥 Processing real-time data...")
            print(f"   Original shape: {real_data.shape}")

            # Check if required columns exist
            missing_cols = [col for col in self.new_columns if col not in real_data.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                return None, None, None

            # Select and preprocess features
            df_realtime_selected = real_data[self.new_columns].copy()
            df_realtime_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_realtime_selected.fillna(0, inplace=True)
            df_realtime_selected = df_realtime_selected.astype(int)

            # Scale and predict
            X_realtime_scaled = self.scaler.transform(df_realtime_selected)
            pred_prob = self.model.predict(X_realtime_scaled, verbose=0)
            pred_labels = (pred_prob > 0.5).astype(int).flatten()

            return df_realtime_selected, pred_prob, pred_labels
            
        except Exception as e:
            st.error(f"❌ Error processing real-time data: {e}")
            return None, None, None
    
    def calculate_distances(self, real_features_clean, predicted_classes):
        """Calculate statistical distances between training and real-time data using ACTUAL training data"""
        try:
            if self.X_train_orig is None or self.y_train_orig is None:
                st.error("❌ Original training data not loaded")
                return None
            
            # Set up labels and desired samples (matching SafeML6.ipynb)
            labels = sorted(np.unique(self.y_train_orig))  # [0, 1]
            desired_samples = {0: 13273, 1: 3832}
            
            # Convert real-time features to numpy array for consistency
            real_features_array = real_features_clean.values
            
            metrics_results = []
            
            # Loop over each label/class for ECDF statistical distance measures
            for current_label in range(len(labels)):
                print(f"\n📊 Processing Class {current_label}...")
                
                # Get training and test data for this specific label using the ACTUAL training data
                X_train_loc_for_label, X_test_loc_for_label = get_X_train_and_test_data_for_given_label(
                    labels, current_label, predicted_classes, 
                    self.X_train_orig, real_features_array, self.y_train_orig, predicted_classes,
                    self.feature_columns
                )
                
                print(f"   Training data shape for Class {current_label}: {X_train_loc_for_label.shape}")
                print(f"   Test data shape for Class {current_label}: {X_test_loc_for_label.shape}")
                print(f"   Training samples: {len(X_train_loc_for_label)} rows")
                print(f"   Test samples: {len(X_test_loc_for_label)} rows")
                
                # Force test data to desired size (matching SafeML6.ipynb)
                n_desired = desired_samples.get(current_label, len(X_test_loc_for_label))
                np.random.seed(42)
                
                if len(X_test_loc_for_label) > n_desired:
                    X_test_loc_for_label = X_test_loc_for_label.sample(n=n_desired, random_state=42)
                elif len(X_test_loc_for_label) < n_desired:
                    print(f"⚠️ Not enough samples for Class {current_label}: requested {n_desired}, available {len(X_test_loc_for_label)}. Oversampling with replacement.")
                    X_test_loc_for_label = X_test_loc_for_label.sample(n=n_desired, replace=True, random_state=42)
                
                print(f"👉 After sampling: Class {current_label} test set has {len(X_test_loc_for_label)} samples (target: {n_desired})")
                
                # Calculate statistical distance measures
                metrics = get_statistical_dist_measures_for_class_result(
                    accuracy=1.0, X_train_L=X_train_loc_for_label, X_test_L=X_test_loc_for_label
                )
                
                print(f"Class {current_label} metrics:", metrics)
                metrics_results.append(metrics)
            
            # Convert to required format
            realtime_distances_class0 = {
                'Anderson_Darling_dist': metrics_results[0]['Anderson_Darling_dist'],
                'CVM_dist': metrics_results[0]['CVM_dist'],
                'DTS_dist': metrics_results[0]['DTS_dist'],
                'Kolmogorov_Smirnov_dist': metrics_results[0]['Kolmogorov_Smirnov_dist'],
                'Kuiper_dist': metrics_results[0]['Kuiper_dist'],
                'Wasserstein_dist': metrics_results[0]['Wasserstein_dist']
            }
            
            realtime_distances_class1 = {
                'Anderson_Darling_dist': metrics_results[1]['Anderson_Darling_dist'],
                'CVM_dist': metrics_results[1]['CVM_dist'],
                'DTS_dist': metrics_results[1]['DTS_dist'],
                'Kolmogorov_Smirnov_dist': metrics_results[1]['Kolmogorov_Smirnov_dist'],
                'Kuiper_dist': metrics_results[1]['Kuiper_dist'],
                'Wasserstein_dist': metrics_results[1]['Wasserstein_dist']
            }
            
            return {
                'class0_distances': realtime_distances_class0,
                'class1_distances': realtime_distances_class1
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating distances: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def calculate_accuracy_estimation(self, runtime_distances, predicted_classes):
        """Calculate accuracy estimation based on runtime distances"""
        try:
            # Calculate class distribution
            total_samples = len(predicted_classes)
            class0_count = np.sum(predicted_classes == 0)
            class1_count = np.sum(predicted_classes == 1)
            
            # Calculate weights
            class0_weight = class0_count / total_samples
            class1_weight = class1_count / total_samples
            
            # Estimate accuracy from distance ratios
            estimated_accuracy, analysis = estimate_accuracy_from_distance_ratios(
                training_data_path_class0=self.class0_path,
                training_data_path_class1=self.class1_path,
                realtime_distances_class0=runtime_distances['class0_distances'],
                realtime_distances_class1=runtime_distances['class1_distances']
            )
            
            return {
                'total_samples': total_samples,
                'class0_count': class0_count,
                'class1_count': class1_count,
                'class0_weight': class0_weight,
                'class1_weight': class1_weight,
                'weighted_accuracy': estimated_accuracy,
                'analysis': analysis
            }
            
        except Exception as e:
            st.error(f"❌ Error calculating accuracy estimation: {e}")
            return {
                'total_samples': len(predicted_classes),
                'class0_count': np.sum(predicted_classes == 0),
                'class1_count': np.sum(predicted_classes == 1),
                'class0_weight': 0.5,
                'class1_weight': 0.5,
                'weighted_accuracy': 0.5,
                'analysis': {}
            }
    
    def make_decision(self, runtime_distances):
        """Make SafeML decision based on runtime distances"""
        try:
            # Estimate accuracy from distance ratios
            estimated_accuracy, analysis = estimate_accuracy_from_distance_ratios(
                training_data_path_class0=self.class0_path,
                training_data_path_class1=self.class1_path,
                realtime_distances_class0=runtime_distances['class0_distances'],
                realtime_distances_class1=runtime_distances['class1_distances']
            )
            
            best_training_accuracy = analysis.get('overall_best_accuracy', 0.9980)
            
            # Make SafeML decision
            decision, confidence_level, accuracy_difference = safeml_confidence_assessment_both_classes(
                best_training_accuracy=best_training_accuracy,
                estimated_accuracy=estimated_accuracy,
                distance_analysis=analysis
            )
            
            # Calculate stability analysis
            stability_analysis = calculate_stability_analysis(
                realtime_distances_class0=runtime_distances['class0_distances'],
                realtime_distances_class1=runtime_distances['class1_distances'],
                training_data_path_class0=self.class0_path,
                training_data_path_class1=self.class1_path
            )
            
            # Get stable measures count
            class0_stable_measures = stability_analysis['Class0']['stable_count']
            class1_stable_measures = stability_analysis['Class1']['stable_count']
            
            # Determine decision code and action
            if decision == "AUTONOMOUS":
                if confidence_level == "HIGH":
                    decision_code = 1
                    action = "🟢 AUTONOMOUS OPERATION"
                    risk_level = "LOW"
                else:
                    decision_code = 2
                    action = "🟡 ENHANCED MONITORING"
                    risk_level = "MEDIUM"
            else:
                decision_code = 3
                action = "🔴 HUMAN OVERSIGHT REQUIRED"
                risk_level = "HIGH"
            
            return {
                'decision': decision,
                'confidence_level': confidence_level,
                'accuracy_difference': accuracy_difference,
                'estimated_accuracy': estimated_accuracy,
                'best_training_accuracy': best_training_accuracy,
                'decision_code': decision_code,
                'action': action,
                'risk_level': risk_level,
                'class0_stable_measures': class0_stable_measures,
                'class1_stable_measures': class1_stable_measures,
                'stability_analysis': stability_analysis
            }
            
        except Exception as e:
            st.error(f"❌ Error making decision: {e}")
            return {
                'decision': 'HUMAN_INTERVENTION',
                'confidence_level': 'LOW',
                'accuracy_difference': 0.1,
                'estimated_accuracy': 0.8,
                'best_training_accuracy': 0.9980,
                'decision_code': 3,
                'action': '🔴 HUMAN OVERSIGHT REQUIRED',
                'risk_level': 'HIGH',
                'class0_stable_measures': 0,
                'class1_stable_measures': 0,
                'stability_analysis': {'Class0': {'measure_details': []}, 'Class1': {'measure_details': []}}
            }

def create_time_series_plots(prediction_df):
    """Create time series visualization plots using SAMPLED data"""
    try:
        # Time aggregation
        time_window = '30S'
        time_grouped = prediction_df.set_index('timestamp').groupby(pd.Grouper(freq=time_window)).agg({
            'prediction': ['count', 'sum', 'mean'],
            'prediction_probability': 'mean',
            'flow_byts_s': 'mean',
            'flow_pkts_s': 'mean',
            'pkt_size_avg': 'mean'
        }).reset_index()
        
        # Flatten column names
        time_grouped.columns = ['timestamp', 'total_flows', 'anomaly_count', 'anomaly_rate', 
                               'avg_probability', 'flow_byts_s', 'flow_pkts_s', 'pkt_size_avg']
        
        # Remove empty time windows
        time_grouped = time_grouped[time_grouped['total_flows'] > 0]
        
        return time_grouped
        
    except Exception as e:
        st.error(f"❌ Error creating time series plots: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Real-Time Network Anomaly Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Initialize detector and cyber attack agent
    detector = RealTimeAnomalyDetector()
    cyber_agent = CyberAttackAnalysisAgent() if LANGGRAPH_AVAILABLE else None
    
    # Check if original training data was loaded
    if detector.X_train_orig is None:
        st.error("❌ Failed to load original training data. Please ensure training_data.csv is available.")
        return
    
    # Load models and training data
    if not detector.load_models():
        st.error("Failed to load models. Please ensure model files are available.")
        return
    
    if not detector.load_training_distances():
        st.error("Failed to load training distances. Please ensure Excel files are available.")
        return
    
    # Load COMPLETE real-time data
    def load_complete_real_time_data():
        try:
            real_data = pd.read_csv("real.csv")
            return real_data
        except Exception as e:
            st.error(f"Could not load real-time data: {e}")
            return None
    
    # Load COMPLETE real-time data
    real_data = load_complete_real_time_data()
    
    if real_data is None:
        return
    
    # Process COMPLETE real-time data
    with st.spinner("Processing complete real-time data..."):
        # Use ALL data for prediction and distance calculation
        real_features_clean, predictions, predicted_classes = detector.process_real_time_data(real_data)
        
        if real_features_clean is None:
            st.error("Failed to process real-time data")
            return
        
        runtime_distances = detector.calculate_distances(real_features_clean, predicted_classes)
        
        if runtime_distances is None:
            st.error("Failed to calculate distances")
            return
        
        # Calculate accuracy estimation using COMPLETE data
        accuracy_results = detector.calculate_accuracy_estimation(runtime_distances, predicted_classes)
        
        # Make decision based on complete analysis
        decision_result = detector.make_decision(runtime_distances)
    
    # Initialize session state for chatbot (moved to main page)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'show_chatbot' not in st.session_state:
        st.session_state.show_chatbot = False
    
    # Display metrics based on COMPLETE data
    st.sidebar.title("📊 System Metrics (Complete Data)")
    
    # Display key metrics from COMPLETE data
    total_samples = accuracy_results['total_samples']
    class0_count = accuracy_results['class0_count']
    class1_count = accuracy_results['class1_count']
    anomaly_rate = (class1_count / total_samples) * 100
    
    st.sidebar.metric("Total Samples", f"{total_samples:,}")
    st.sidebar.metric("Normal Traffic", f"{class0_count:,} ({(class0_count/total_samples)*100:.1f}%)")
    st.sidebar.metric("Anomaly Detected", f"{class1_count:,} ({anomaly_rate:.1f}%)")
    
    # Decision display in sidebar
    st.sidebar.markdown("---")
    st.sidebar.title("🎯 System Decision")
    
    decision_class = "autonomous" if decision_result['decision_code'] == 1 else ("monitoring" if decision_result['decision_code'] == 2 else "oversight")
    decision_html = f"""
    <div class="decision-box {decision_class}">
        {decision_result['action']}
    </div>
    """
    st.sidebar.markdown(decision_html, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"**Confidence Level:** {decision_result['confidence_level']}")
    st.sidebar.markdown(f"**Risk Level:** {decision_result['risk_level']}")
    st.sidebar.markdown(f"**Decision Code:** {decision_result['decision_code']}")
    
    # Stability metrics
    st.sidebar.markdown("---")
    st.sidebar.title("🔍 Stability Analysis")
    
    stability_html = f"""
    <div class="stability-metric {'stable' if decision_result['class0_stable_measures'] >= 4 else 'unstable'}">
        Class 0: {decision_result['class0_stable_measures']}/6 Stable
    </div>
    <div class="stability-metric {'stable' if decision_result['class1_stable_measures'] >= 4 else 'unstable'}">
        Class 1: {decision_result['class1_stable_measures']}/6 Stable
    </div>
    """
    st.sidebar.markdown(stability_html, unsafe_allow_html=True)
    
    # Main dashboard - Use SAMPLED data for visualization only
    col1, col2 = st.columns(2)
    
    # FIXED: Initialize sample_size and sample_fraction at the beginning
    sample_fraction = 0.0008  # 0.08% as decimal
    sample_size = max(50, int(total_samples * sample_fraction))  # Minimum 50 samples
    
    # Prepare SAMPLED data for visualization
    if 'timestamp' in real_data.columns:
        # Take only the FIRST sample_size lines for visualization
        sampled_indices = range(min(sample_size, len(real_data)))
        
        prediction_df = pd.DataFrame({
            'timestamp': pd.to_datetime(real_data['timestamp'].iloc[sampled_indices]),
            'prediction': predicted_classes[sampled_indices],
            'prediction_probability': predictions.flatten()[sampled_indices]
        })
        
        # Add traffic features for visualization
        traffic_features = ['flow_byts_s', 'flow_pkts_s', 'pkt_size_avg']
        for feature in traffic_features:
            if feature in real_data.columns:
                prediction_df[feature] = real_data[feature].iloc[sampled_indices]
            else:
                # Create dummy data if feature not available
                prediction_df[feature] = np.random.normal(1000, 200, len(sampled_indices))
        
        # Create time series data for visualization
        time_grouped = create_time_series_plots(prediction_df)
        
       
        
        # Plot 1: Flow Bytes Continuous Traffic Monitoring
        with col1:
            st.subheader("📊 Flow Bytes - Continuous Traffic Monitoring")
            
            fig1 = go.Figure()
            
            # Add normal traffic line
            fig1.add_trace(go.Scatter(
                x=time_grouped['timestamp'],
                y=time_grouped['flow_byts_s'],
                mode='lines',
                name='Normal Traffic',
                line=dict(color='blue', width=2)
            ))
            
            # Add anomaly markers
            anomaly_periods = time_grouped[time_grouped['anomaly_count'] > 0]
            if len(anomaly_periods) > 0:
                fig1.add_trace(go.Scatter(
                    x=anomaly_periods['timestamp'],
                    y=anomaly_periods['flow_byts_s'],
                    mode='markers',
                    name='Anomaly Detection',
                    marker=dict(color='red', size=8)
                ))
            
            fig1.update_layout(
                title=f"Flow Bytes per Second Over Time (Sample: {sample_size:,})",
                xaxis_title="Time",
                yaxis_title="Flow Bytes/s",
                showlegend=True
            )
            
            st.plotly_chart(fig1, use_container_width=True)
        
        # Plot 2: Anomaly Detection Rate Over Time
        with col2:
            st.subheader("📈 Anomaly Detection Rate Over Time")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=time_grouped['timestamp'],
                y=time_grouped['anomaly_rate'] * 100,
                mode='lines',
                name='Anomaly Rate (%)',
                line=dict(color='orange', width=2),
                fill='tozeroy'
            ))
            
            # Add threshold lines
            fig2.add_hline(y=50, line_dash="dash", line_color="red", 
                          annotation_text="High Risk (50%)")
            fig2.add_hline(y=25, line_dash="dash", line_color="yellow", 
                          annotation_text="Medium Risk (25%)")
            
            fig2.update_layout(
                title=f"Anomaly Detection Rate (%) (Sample: {sample_size:,})",
                xaxis_title="Time",
                yaxis_title="Anomaly Rate (%)",
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Second row of plots
        col3, col4 = st.columns(2)
        
        # Plot 3: Flow Packets - Supporting Traffic Metric
        with col3:
            st.subheader("📊 Flow Packets - Supporting Traffic Metric")
            
            fig3 = go.Figure()
            
            fig3.add_trace(go.Scatter(
                x=time_grouped['timestamp'],
                y=time_grouped['flow_pkts_s'],
                mode='lines',
                name='Normal Traffic',
                line=dict(color='green', width=2)
            ))
            
            # Add anomaly markers
            if len(anomaly_periods) > 0:
                fig3.add_trace(go.Scatter(
                    x=anomaly_periods['timestamp'],
                    y=anomaly_periods['flow_pkts_s'],
                    mode='markers',
                    name='Anomaly Detection',
                    marker=dict(color='red', size=8)
                ))
            
            fig3.update_layout(
                title=f"Flow Packets per Second (Sample: {sample_size:,})",
                xaxis_title="Time",
                yaxis_title="Flow Packets/s",
                showlegend=True
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        # Plot 4: Traffic Volume vs Anomaly Detection
        with col4:
            st.subheader("📊 Traffic Volume vs Anomaly Detection")
            
            fig4 = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Total flows
            fig4.add_trace(
                go.Scatter(
                    x=time_grouped['timestamp'],
                    y=time_grouped['total_flows'],
                    mode='lines',
                    name='Total Flows',
                    line=dict(color='blue', width=2)
                ),
                secondary_y=False
            )
            
            # Anomaly count
            fig4.add_trace(
                go.Scatter(
                    x=time_grouped['timestamp'],
                    y=time_grouped['anomaly_count'],
                    mode='lines',
                    name='Anomaly Count',
                    line=dict(color='red', width=2)
                ),
                secondary_y=True
            )
            
            fig4.update_xaxes(title_text="Time")
            fig4.update_yaxes(title_text="Total Flows", secondary_y=False)
            fig4.update_yaxes(title_text="Anomaly Count", secondary_y=True)
            
            fig4.update_layout(title=f"Traffic Volume vs Anomaly Detection (Sample: {sample_size:,})")
            
            st.plotly_chart(fig4, use_container_width=True)
    else:
        # Handle case where timestamp is not available
        st.warning("⚠️ No timestamp column found in real-time data. Skipping time series visualization.")
    
    # Detailed Analysis Section (based on COMPLETE data)
    st.markdown("---")
    st.title("📋 Detailed Analysis (Complete Data)")
    
    # Accuracy and Performance Metrics
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.subheader("🎯 Accuracy Metrics")
        weighted_accuracy = accuracy_results['weighted_accuracy']
        st.metric("Estimated Accuracy", f"{weighted_accuracy:.4f}")
        st.metric("Baseline Accuracy", "0.9980")
        accuracy_drop = 0.9980 - weighted_accuracy
        st.metric("Accuracy Drop", f"{accuracy_drop:.4f}")
    
    with col6:
        st.subheader("📊 Distance Measures")
        st.markdown("**Class 0 (Normal) Stability:**")
        for detail in decision_result['stability_analysis']['Class0']['measure_details']:
            st.markdown(f"{detail['status']} {detail['measure']}: {detail['ratio']:.2f}x")
    
    with col7:
        st.subheader("🔍 Class 1 (Anomaly) Stability")
        st.markdown("**Class 1 (Anomaly) Stability:**")
        for detail in decision_result['stability_analysis']['Class1']['measure_details']:
            st.markdown(f"{detail['status']} {detail['measure']}: {detail['ratio']:.2f}x")
    
    # Summary Statistics (based on COMPLETE data)
    st.markdown("---")
    st.title("📈 Summary Statistics (Complete Data)")
    
    summary_data = {
        'Metric': ['Total Samples', 'Normal Traffic', 'Anomaly Detected', 'Anomaly Rate', 
                   'Class 0 Stable Measures', 'Class 1 Stable Measures', 'System Decision',
                   'Estimated Accuracy', 'Visualization Sample Size'],
        'Value': [f"{total_samples:,}", f"{class0_count:,}", f"{class1_count:,}", 
                 f"{anomaly_rate:.1f}%", f"{decision_result['class0_stable_measures']}/6",
                 f"{decision_result['class1_stable_measures']}/6", decision_result['action'],
                 f"{weighted_accuracy:.4f}", f"{sample_size:,} ({sample_fraction*100:.3f}%)"]
    }
    # Auto-refresh option
    if st.checkbox("Auto-refresh (every 30 seconds)"):
        import time
        time.sleep(30)
        st.rerun()
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)
    
    # Display complete data statistics
    st.markdown("---")
    st.title("📊 Complete Data Analysis")
    
    col8, col9 = st.columns(2)
    
    with col8:
        st.subheader("🎯 Class Distribution (Complete Data)")
        st.markdown(f"**Total Samples:** {total_samples:,}")
        st.markdown(f"**Class 0 Weight:** {accuracy_results['class0_weight']:.1%}")
        st.markdown(f"**Class 1 Weight:** {accuracy_results['class1_weight']:.1%}")
        st.markdown(f"**Weighted Accuracy:** {weighted_accuracy:.4f}")
    
    with col9:
        st.subheader("📈 Visualization vs Analysis")
        st.markdown(f"**Analysis Data:** {total_samples:,} samples (100%)")
        st.markdown(f"**Visualization Data:** {sample_size:,} samples ({sample_fraction*100:.3f}%)")
        st.markdown(f"**Decision Based On:** Complete data analysis with ACTUAL training data")
        st.markdown(f"**Charts Show:** Sample for performance")
    
    # Floating Chatbot Icon and Popup (Bottom Right)
    st.markdown("""
    <style>
    .chatbot-float {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        cursor: pointer;
        font-size: 24px;
        transition: all 0.3s ease;
    }
    .chatbot-float:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    }
    .chatbot-popup {
        position: fixed;
        bottom: 90px;
        right: 20px;
        z-index: 999;
        background: white;
        border-radius: 15px;
        width: 400px;
        max-height: 600px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid #e0e0e0;
        overflow: hidden;
    }
    .chatbot-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        font-weight: bold;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .chatbot-body {
        padding: 15px;
        max-height: 500px;
        overflow-y: auto;
    }
    .close-btn {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .close-btn:hover {
        background: rgba(255,255,255,0.2);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Floating chatbot button
    chatbot_placeholder = st.empty()
    
    # Create columns to position the button
    col_spacer, col_button = st.columns([10, 1])
    
    with col_button:
        # Use custom HTML for floating button
        if st.button("🤖", key="floating_chatbot", help="Open AI Security Assistant"):
            st.session_state.show_chatbot = not st.session_state.show_chatbot
    
    # Chatbot popup window
    if cyber_agent and cyber_agent.available and st.session_state.show_chatbot:
        # Create a container for the popup
        with st.container():
            st.markdown('<div class="chatbot-popup">', unsafe_allow_html=True)
            
            # Header
            st.markdown("### 🤖 AI Security Assistant")
            st.markdown("---")
            
            # Chat Interface
            st.subheader("💬 Ask AI Assistant")
            
            # Display chat messages
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask about cybersecurity, DDoS attacks, or current system status..."):
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Generate and display assistant response
                with st.spinner("🤖 Thinking..."):
                    # Prepare system context
                    system_context = {
                        'total_samples': f"{accuracy_results['total_samples']:,}",
                        'anomalies_detected': f"{accuracy_results['class1_count']:,} ({(accuracy_results['class1_count']/accuracy_results['total_samples']*100):.1f}%)",
                        'system_decision': decision_result['action'],
                        'risk_level': decision_result['risk_level']
                    }
                    
                    # Get AI response
                    ai_response = cyber_agent.answer_user_question(prompt, system_context)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                    
                    # Keep only last 10 exchanges (20 messages)
                    if len(st.session_state.chat_history) > 20:
                        st.session_state.chat_history = st.session_state.chat_history[-20:]
                
                # Rerun to show updated conversation
                st.rerun()
            
            # Quick Analysis Button
            
            if st.button("🔍 Analyze for Cyber Attacks", help="Click to analyze real-time data for DDoS and other cyber attacks", key="analyze_popup"):
                with st.expander("🚨 Cyber Attack Analysis Report", expanded=True):
                    with st.spinner("Analyzing network traffic for cyber attacks..."):
                        # Run cyber attack analysis
                        cyber_report = cyber_agent.analyze_network_traffic(real_data)
                        
                        # Display threat level with color coding
                        threat_level = cyber_report.get("overall_threat_level", "UNKNOWN")
                        if threat_level == "HIGH":
                            st.error(f"🚨 **THREAT LEVEL: {threat_level}**")
                        elif threat_level == "MEDIUM":
                            st.warning(f"⚠️ **THREAT LEVEL: {threat_level}**")
                        elif threat_level == "LOW":
                            st.success(f"✅ **THREAT LEVEL: {threat_level}**")
                        else:
                            st.info(f"❓ **THREAT LEVEL: {threat_level}**")
                        
                        # Display main findings
                        st.subheader("📋 Analysis Summary")
                        
                        # Check if we have conversational response (new format)
                        if "conversational_response" in cyber_report:
                            st.write("### 🤖 AI Cybersecurity Analysis")
                            st.write(cyber_report["conversational_response"])
                            
                            # Show chunk count and threat distribution if available
                            if "total_chunks_analyzed" in cyber_report:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Chunks Analyzed", cyber_report["total_chunks_analyzed"])
                                with col2:
                                    st.metric("Confidence Score", f"{cyber_report.get('confidence_score', 0):.1%}")
                            
                            # Show threat distribution if available
                            if "threat_distribution" in cyber_report:
                                st.subheader("📊 Threat Distribution")
                                threat_dist = cyber_report["threat_distribution"]
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🔴 High Risk", threat_dist.get("high", 0))
                                with col2:
                                    st.metric("🟡 Medium Risk", threat_dist.get("medium", 0))
                                with col3:
                                    st.metric("🟢 Low Risk", threat_dist.get("low", 0))
                        
                        else:
                            # Legacy format
                            st.write(cyber_report.get("attack_summary", "No summary available"))
                        
                        # Download report button
                        report_json = json.dumps(cyber_report, indent=2)
                        st.download_button(
                            label="📥 Download Report",
                            data=report_json,
                            file_name=f"cyber_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif not cyber_agent or not cyber_agent.available:
        # Show setup message when chatbot is opened but agent is not available
        if st.session_state.show_chatbot:
            with st.container():
                st.warning("🤖 **AI Cyber Attack Analyst Setup Required**")
                st.info("""
                **Setup Instructions:**
                1. Install: `pip install langchain-community`
                2. Install Ollama: https://ollama.ai/
                3. Run: `ollama pull qwen2.5:1.5b`
                4. Start: `ollama serve`
                """)
                if st.button("❌ Close Setup", key="close_setup"):
                    st.session_state.show_chatbot = False
                    st.rerun()
    
    

if __name__ == "__main__":
    main()