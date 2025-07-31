import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from tensorflow.keras.models import load_model
import joblib
import warnings
warnings.filterwarnings('ignore')

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
            st.success(f"✅ Models loaded successfully from {self.model_path} and {self.scaler_path}")
            return True
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            return False
    
    def load_training_distances(self):
        """Load training distance data"""
        try:
            if os.path.exists(self.class0_path) and os.path.exists(self.class1_path):
                st.success(f"✅ Training distance files found")
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
    
    # Initialize detector
    detector = RealTimeAnomalyDetector()
    
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
        
        # Display sampling info
        st.info(f"📊 Visualization shows {sample_size:,} samples ({sample_fraction*100:.3f}% of {total_samples:,} total samples). Analysis and decisions are based on complete data using ACTUAL training data.")
        
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
    
    # Auto-refresh option
    if st.checkbox("Auto-refresh (every 30 seconds)"):
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()