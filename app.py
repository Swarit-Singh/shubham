import streamlit as st
import numpy as np
from PIL import Image
import os
import io
import pandas as pd
import datetime

from watermark_pipeline import watermark_pipeline
from ml_cnn_xgboost import (
    extract_cnn_features,
    predict_threshold_xgboost,
    load_cnn_model,
    load_xgboost_model,
)
from watermark_operations import extract_pred_error, extract_hist_shift
from visualization_utils import plot_histograms_styled, plot_difference_styled, plot_comparative_bar
from utils import (
    get_psnr,
    get_embedding_capacity_pee,
    get_embedding_capacity_hs
)

from skimage.metrics import structural_similarity as ssim

def get_ssim_fixed(img1, img2):
    img1 = np.asarray(img1, dtype=np.uint8)
    img2 = np.asarray(img2, dtype=np.uint8)
    return ssim(img1, img2, data_range=255)

get_ssim = get_ssim_fixed

def add_gaussian_noise(image, sigma=10):
    row, col = image.shape
    gauss = np.random.normal(0, sigma, (row, col))
    noisy_image = np.clip(image + gauss, 0, 255)
    return noisy_image.astype(np.uint8)

def add_salt_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, image.shape[axis], int(num_salt)) for axis in range(image.ndim)]
    noisy[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, image.shape[axis], int(num_pepper)) for axis in range(image.ndim)]
    noisy[tuple(coords)] = 0
    return noisy

def jpeg_compress(image, quality=70):
    pil_img = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def median_filter_attack(image, kernel_size=3):
    from scipy.signal import medfilt2d
    return medfilt2d(image, kernel_size=kernel_size).astype(np.uint8)

def image_to_bytes(pil_image, img_format="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=img_format)
    return buf.getvalue()

st.set_page_config(page_title="Reversible Watermarking Toolkit", layout="wide", page_icon="🛡️")
st.title("🖼️ Comprehensive Reversible Watermarking Toolkit")

if 'processed' not in st.session_state: st.session_state.processed = False
if 'input_image' not in st.session_state: st.session_state.input_image = None
if 'generated_payload' not in st.session_state: st.session_state.generated_payload = ""
if 'cnn_model_global' not in st.session_state: st.session_state.cnn_model_global = None
if 'xgb_model_global' not in st.session_state: st.session_state.xgb_model_global = None

col1, col2 = st.columns([1, 2]) 

with col1:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload Grayscale Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded_file is not None:
        try:
            input_pil_image = Image.open(uploaded_file).convert("L")
        except Exception as e:
            st.error(f"Could not open image: {e}")
            st.stop()
        input_image_arr = np.array(input_pil_image)
        st.session_state.input_image = input_image_arr
        st.image(input_pil_image, caption="Uploaded Original Image", use_container_width=True)

        method_options = {
            "prediction_error": "Prediction Error Expansion (PEE)",
            "histogram_shift": "Histogram Shifting (HS)",
            "ml_cnn_xgboost": "ML-Assisted (CNN+XGBoost for PEE T)"
        }
        method = st.selectbox("Select Watermarking Method", list(method_options.keys()),
                              format_func=lambda x: method_options[x], key="watermarking_method")

        st.subheader("Payload Configuration")
        payload_source = st.radio("Payload source:", ("Manual Input", "Generate from Patient Info"), key="payload_src", horizontal=True)
        payload_bits_str_manual = "01010101010101010101010101010101"
        if payload_source == "Manual Input":
            payload_bits_str = st.text_area("Binary payload:", value=payload_bits_str_manual, height=100, key="manual_payload")
        else:
            with st.form(key="patient_info_form"):
                patient_name = st.text_input("Name", "J. Doe")
                patient_id = st.text_input("ID", "P123")
                patient_age_form = st.number_input("Age", 1, 120, 30, key="pat_age_form")
                patient_gender_form = st.selectbox("Gender", ["Male", "Female", "Other"], key="pat_gender_form")
                patient_diag_form = st.text_input("Diagnosis", "Normal", key="pat_diag_form")
                submitted_info = st.form_submit_button("Generate Payload")
                if submitted_info:
                    info_str = f"Name:{patient_name};ID:{patient_id};Age:{patient_age_form};Gender:{patient_gender_form};Diag:{patient_diag_form}" 
                    st.session_state.generated_payload = ''.join(format(ord(c), '08b') for c in info_str)
            payload_bits_str = st.session_state.generated_payload
            st.text_area("Generated Payload:", value=payload_bits_str, height=100, disabled=True, key="gen_payload_disp")

        payload_len = len(payload_bits_str)
        st.write(f"Payload Length: {payload_len} bits")

        params_for_pipeline = {}
        capacity = 0
        can_proceed = True

        if method == "prediction_error":
            T_pee = st.slider("PEE Threshold (T)", 1, 10, 1, key="pee_T_slider")
            params_for_pipeline['T'] = T_pee
            capacity = get_embedding_capacity_pee(input_image_arr, T_pee)
            st.info(f"PEE using T={T_pee}. Est. Capacity: {capacity} bits")

        elif method == "histogram_shift":
            capacity = get_embedding_capacity_hs(input_image_arr)
            st.info(f"HS Est. Capacity: {capacity} bits")

        elif method == "ml_cnn_xgboost":
            st.markdown("Predicts optimal PEE `T` via CNN+XGBoost.")
            if st.session_state.cnn_model_global is None:
                st.session_state.cnn_model_global = load_cnn_model()
            if st.session_state.xgb_model_global is None:
                if os.path.exists("xgb_threshold_predictor.pkl"):
                    st.session_state.xgb_model_global = load_xgboost_model()
                else:
                    st.warning("`xgb_threshold_predictor.pkl` not found. Train model below or ensure file exists."); can_proceed = False
            if st.session_state.cnn_model_global and st.session_state.xgb_model_global and can_proceed:
                features_xgb = extract_cnn_features(input_image_arr, st.session_state.cnn_model_global)
                predicted_T_xgb_float = predict_threshold_xgboost(features_xgb, st.session_state.xgb_model_global)
                predicted_T_xgb = max(1, min(10, int(round(predicted_T_xgb_float))))
                st.success(f"ML Predicted Optimal T = {predicted_T_xgb}")
                params_for_pipeline['T'] = predicted_T_xgb
                capacity = get_embedding_capacity_pee(input_image_arr, predicted_T_xgb)
                st.info(f"Est. Capacity (CNN+XGB PEE, T={predicted_T_xgb}): {capacity} bits")

            with st.expander("Retrain CNN+XGBoost Model (Optional)"):
                data_folder = st.text_input("Training Data Folder", value="ml_training_data/", key="train_folder_input")
                if st.button("Start Retraining XGBoost", key="retrain_xgb_btn"):
                    if os.path.isdir(data_folder):
                        with st.spinner("Retraining XGBoost... This may take a while."):
                            from ml_cnn_xgboost import train_xgboost_model
                            train_images_list, train_thresholds_list, valid_found = [], [], False
                            for fname in os.listdir(data_folder):
                                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                                    img_p, lbl_p = os.path.join(data_folder, fname), os.path.join(data_folder, fname.rsplit('.',1)[0] + ".txt")
                                    if os.path.exists(lbl_p):
                                        train_images_list.append(np.array(Image.open(img_p).convert("L")))
                                        with open(lbl_p, "r") as f: train_thresholds_list.append(float(f.read().strip()))
                                        valid_found = True
                            if not valid_found:
                                st.error("No valid image-label pairs found for retraining.")
                            else:
                                cnn_train_model = load_cnn_model(force_reload=True)
                                feats_train = [extract_cnn_features(img, cnn_train_model) for img in train_images_list]
                                train_xgboost_model(feats_train, train_thresholds_list)
                                st.session_state.xgb_model_global = load_xgboost_model() 
                                st.success("XGBoost model retrained & reloaded!")
                    else:
                        st.error(f"Folder not found: '{data_folder}' for retraining.")

        payload_bits_str_final = payload_bits_str
        if capacity > 0 and payload_len > capacity:
            st.warning(f"Payload ({payload_len}b) > capacity ({capacity}b). Truncated.")
            payload_bits_str_final = payload_bits_str[:capacity]
        elif capacity == 0 and can_proceed:
            st.warning("Capacity 0. No data embedded."); payload_bits_str_final = ""
        elif not can_proceed:
            st.error("Cannot proceed due to errors."); payload_bits_str_final = ""

        st.markdown("---")
        btn_disabled = not can_proceed or (not payload_bits_str_final and payload_len > 0)
        if st.button("Process Image (Embed & Extract)", disabled=btn_disabled, key="process_btn"):
            if not payload_bits_str_final and payload_len > 0: st.error("Payload empty due to capacity.")
            elif not can_proceed: st.error("Cannot process due to config errors.")
            else:
                with st.spinner("Processing..."):
                    wm_img_arr, rec_img_arr, ext_bits_str, op_prms = watermark_pipeline(
                        input_image_arr, payload_bits_str_final, method, **params_for_pipeline)
                    st.session_state.update({
                        'processed': True, 'watermarked_image_display': wm_img_arr,
                        'recovered_image_display': rec_img_arr, 'original_payload_display': payload_bits_str_final,
                        'extracted_payload_display': ext_bits_str, 'op_params_display': op_prms,
                        'method_used_display': method,
                        'current_filename': uploaded_file.name
                    })
                    st.success("Processing Complete!")
    else: st.info("Please upload an image to begin.")

with col2:
    st.header("📊 Results & Analysis")
    if st.session_state.get('processed', False):
        input_img_disp = st.session_state.get('input_image')
        wm_img_disp = st.session_state.get('watermarked_image_display')
        rec_img_disp = st.session_state.get('recovered_image_display')
        orig_payload_disp = st.session_state.get('original_payload_display')
        ext_payload_disp = st.session_state.get('extracted_payload_display')
        op_params_fordisp = st.session_state.get('op_params_display')
        method_used_fordisp = st.session_state.get('method_used_display')
        current_filename = st.session_state.get('current_filename', "Unknown")
        current_run_T = op_params_fordisp.get('T') if op_params_fordisp else None

        st.subheader("🖼️ Image Comparison")
        if input_img_disp is not None and wm_img_disp is not None and rec_img_disp is not None:
            img_col1, img_col2, img_col3 = st.columns(3)
            with img_col1: st.image(input_img_disp, caption="Original", use_container_width=True)
            with img_col2: st.image(wm_img_disp, caption=f"Watermarked ({method_used_fordisp}, T={current_run_T})", use_container_width=True)
            with img_col3: st.image(rec_img_disp, caption="Recovered", use_container_width=True)
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1: st.download_button("Download WM", image_to_bytes(Image.fromarray(wm_img_disp.astype(np.uint8))), f"wm.png", "image/png")
            with dl_col2: st.download_button("Download REC", image_to_bytes(Image.fromarray(rec_img_disp.astype(np.uint8))), f"rec.png", "image/png")

        st.subheader("🔍 Payload & Parameters")
        if method_used_fordisp: st.write(f"**Method:** {method_used_fordisp}")
        if op_params_fordisp: st.write(f"**Params:** `{op_params_fordisp}`")
        if orig_payload_disp is not None and ext_payload_disp is not None:
            pay_col1, pay_col2 = st.columns(2)
            pay_col1.text_area("Original Payload (<=256b):", orig_payload_disp[:256], height=80)
            pay_col2.text_area("Extracted Payload (<=256b):", ext_payload_disp[:256], height=80)

        st.subheader("📈 Performance Metrics")
        if input_img_disp is not None and wm_img_disp is not None and rec_img_disp is not None:
            psnr_wm = get_psnr(input_img_disp, wm_img_disp)
            ssim_wm = get_ssim(input_img_disp, wm_img_disp)
            psnr_rec = get_psnr(input_img_disp, rec_img_disp)
            ssim_rec = get_ssim(input_img_disp, rec_img_disp)
            met_cols1=st.columns(4)
            met_cols1[0].metric("Payload Emb", f"{len(orig_payload_disp or '')}b")
            met_cols1[1].metric("Payload Ext", f"{len(ext_payload_disp or '')}b")
            acc = 0.0
            if orig_payload_disp and ext_payload_disp:
                m_len = min(len(orig_payload_disp), len(ext_payload_disp))
                if m_len > 0: acc = (sum(1 for i in range(m_len) if orig_payload_disp[i] == ext_payload_disp[i]) / len(orig_payload_disp)) * 100 if len(orig_payload_disp) > 0 else 0.0
            met_cols1[2].metric("Accuracy", f"{acc:.2f}%")
            met_cols1[3].metric("Changed Pixels", f"{np.sum(input_img_disp != wm_img_disp)}")
            met_cols2=st.columns(2)
            met_cols2[0].metric("PSNR (Orig vs WM)", f"{psnr_wm:.2f}dB")
            met_cols2[1].metric("SSIM (Orig vs WM)", f"{ssim_wm:.4f}")
            if np.array_equal(input_img_disp, rec_img_disp): st.success("🎉 Image Perfectly Recovered!")
            else: 
                st.warning("Image not perfectly recovered."); met_cols3=st.columns(2)
                met_cols3[0].metric("PSNR (Orig vs Rec)", f"{psnr_rec:.2f}dB")
                met_cols3[1].metric("SSIM (Orig vs Rec)", f"{ssim_rec:.4f}")

        st.subheader("📊 Visualizations")
        if input_img_disp is not None and wm_img_disp is not None:
            st.pyplot(plot_histograms_styled(input_img_disp, wm_img_disp, "Original", "Watermarked"))
            st.pyplot(plot_difference_styled(input_img_disp, wm_img_disp))

        # --- Attack Simulation and Method Comparison (runs automatically) ---
        import matplotlib.pyplot as plt
        st.subheader("🛡️ Attack Simulation and Method Comparison")
        if input_img_disp is not None and orig_payload_disp is not None and len(orig_payload_disp) > 0:
            payload_len = len(orig_payload_disp)
            methods_to_run = {
                "PEE(T=1)": {"method": "prediction_error", "params": {"T": 1}},
                "HS": {"method": "histogram_shift", "params": {}},
            }
            if st.session_state.cnn_model_global and st.session_state.xgb_model_global:
                feat_ml = extract_cnn_features(input_img_disp, st.session_state.cnn_model_global)
                pred_T_ml = max(1, min(10, int(round(predict_threshold_xgboost(feat_ml, st.session_state.xgb_model_global)))))
                methods_to_run[f"ML(T={pred_T_ml})"] = {"method": "ml_cnn_xgboost", "params": {"T": pred_T_ml}}

            attacks_to_run = {
                "Gaussian (σ=10)": lambda img: add_gaussian_noise(img, sigma=10),
                "Salt & Pepper (2%)": lambda img: add_salt_pepper_noise(img, amount=0.02),
                "JPEG (Q=70)": lambda img: jpeg_compress(img, quality=70),
                "Median Filter (3x3)": lambda img: median_filter_attack(img, kernel_size=3)
            }

            attack_acc_results = {atk: [] for atk in attacks_to_run}
            method_names = list(methods_to_run.keys())
            log_rows = []

            for atk_name, atk_func in attacks_to_run.items():
                st.markdown(f"**{atk_name}**")
                cols = st.columns(3)
                for i, method_label in enumerate(method_names):
                    wm_img, _, _, op_params = watermark_pipeline(
                        input_img_disp, orig_payload_disp, methods_to_run[method_label]["method"], **methods_to_run[method_label]["params"]
                    )
                    attacked_img = atk_func(wm_img)
                    psnr_val = get_psnr(wm_img, attacked_img)
                    if methods_to_run[method_label]["method"] in ("prediction_error", "ml_cnn_xgboost"):
                        T = methods_to_run[method_label]["params"].get("T", 1)
                        ext_bits, _ = extract_pred_error(attacked_img, payload_len, T)
                    elif methods_to_run[method_label]["method"] == "histogram_shift":
                        p = op_params.get('p')
                        z = op_params.get('z')
                        d = op_params.get('direction')
                        ext_bits, _ = extract_hist_shift(attacked_img, payload_len, p, z, d)
                    else:
                        ext_bits = ""
                    acc = 0.0
                    if ext_bits:
                        min_len = min(len(orig_payload_disp), len(ext_bits))
                        if min_len > 0:
                            acc = sum(1 for j in range(min_len) if orig_payload_disp[j] == ext_bits[j]) / len(orig_payload_disp) * 100
                    attack_acc_results[atk_name].append(acc)
                    log_rows.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "filename": current_filename,
                        "method": method_label,
                        "attack": atk_name,
                        "psnr": psnr_val,
                        "extraction_accuracy": acc
                    })
                    with cols[i % 3]:
                        st.image(attacked_img, caption=f"{method_label}\nPSNR: {psnr_val:.2f} dB", use_container_width=True)
                        st.write(f"Extraction Accuracy: {acc:.2f}%")

                # Extraction accuracy bar chart for this attack (smaller)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(method_names, attack_acc_results[atk_name], color=['#4CAF50', '#2196F3', '#FF9800'])
                ax.set_ylabel('Extraction Accuracy (%)')
                ax.set_title(f'Extraction Accuracy After {atk_name}')
                st.pyplot(fig)

            # Logging: append new results to CSV
            df_log_new = pd.DataFrame(log_rows)
            csv_path = "attack_results_log.csv"
            if os.path.exists(csv_path):
                df_log_old = pd.read_csv(csv_path)
                df_log_all = pd.concat([df_log_old, df_log_new], ignore_index=True)
            else:
                df_log_all = df_log_new
            df_log_all.to_csv(csv_path, index=False)

            # Show summary table
            avg_acc_data = {"Method": method_names}
            for atk_name in attacks_to_run:
                avg_acc_data[atk_name] = attack_acc_results[atk_name]
            df_acc = pd.DataFrame(avg_acc_data)
            df_acc["Average"] = df_acc[[atk for atk in attacks_to_run]].mean(axis=1)
            st.subheader("Average Extraction Accuracy Percentage After Attacks (Current Image Only)")
            st.table(df_acc)
            st.info("A log of all attack results is saved in 'attack_results_log.csv' for later review.")

        st.subheader("🆚 Method Comparison Chart")
        if input_img_disp is not None:
            with st.spinner("Calculating comparison..."):
                comp_data = {"PSNR": {}, "Capacity": {}}
                pay_comp = orig_payload_disp if (orig_payload_disp and len(orig_payload_disp)>0) else "01"
                wm_p1,_,_,_ = watermark_pipeline(input_img_disp,pay_comp,"prediction_error",T=1)
                comp_data["PSNR"]["PEE(T=1)"] = get_psnr(input_img_disp,wm_p1)
                comp_data["Capacity"]["PEE(T=1)"] = get_embedding_capacity_pee(input_img_disp,1)
                wm_h,_,_,_ = watermark_pipeline(input_img_disp,pay_comp,"histogram_shift")
                comp_data["PSNR"]["HS"] = get_psnr(input_img_disp,wm_h)
                comp_data["Capacity"]["HS"] = get_embedding_capacity_hs(input_img_disp)
                if st.session_state.cnn_model_global and st.session_state.xgb_model_global:
                    feat_c = extract_cnn_features(input_img_disp, st.session_state.cnn_model_global)
                    pred_T_c = max(1,min(10,int(round(predict_threshold_xgboost(feat_c,st.session_state.xgb_model_global)))))
                    wm_c,_,_,_ = watermark_pipeline(input_img_disp,pay_comp,"ml_cnn_xgboost",T=pred_T_c)
                    comp_data["PSNR"][f"CNN+XGB(T={pred_T_c})"] = get_psnr(input_img_disp,wm_c)
                    comp_data["Capacity"][f"CNN+XGB(T={pred_T_c})"] = get_embedding_capacity_pee(input_img_disp,pred_T_c)
                if comp_data["PSNR"]: st.pyplot(plot_comparative_bar(comp_data["PSNR"], "PSNR Comparison", "PSNR (dB)"))
                if comp_data["Capacity"]: st.pyplot(plot_comparative_bar(comp_data["Capacity"], "Capacity Comparison", "Capacity (bits)"))
        else: st.warning("Original image not available for comparison.")

    elif uploaded_file: st.info("Configure parameters and click 'Process Image'.")
    else: st.info("Welcome! Upload an image on the left to begin.")
