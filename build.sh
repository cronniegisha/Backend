#!/bin/bash

# Install Python dependencies
pip install gdown

# Create model folder
mkdir -p model

# Download model files
gdown --id 1II2Aj2Y5nQMM910LOsP1EodUurkX5udQ -O model/best_skill_gap_model.pkl
gdown --id 1Kpc29MAVhk4wzatwcq_MBt2nZ-NNOeS7 -O model/education_encoder.pkl
gdown --id 1f-mKKAS5ZZPK_lYMiRPAlxiWeszPbWWe -O model/features_names.pkl
gdown --id 1bJLbWrfUI_L1nRZkTzwgBbxQbx1xU0SI -O model/interests_encoder.pkl
gdown --id 1gW07mJPmK4xBreTsPk27EaJhwcbtwAlf -O model/label_encoder_skill_name.pkl
gdown --id 1l_kK9oFjb_W0rYnNt8W5Ur802Wqw44PX -O model/label_encoder_skill_type.pkl
gdown --id 1978dTzRdE_0CfR2aR4SlLCat7PtfND1j -O model/target_encoder.pkl
gdown --id 1LdTFVLIIDTDPwf4WSYHZNwe7OJg_8WuO -O model/skill_encoder.pkl
gdown --id 1W_gvhSP89D7dVN7Tn8MFyMJHTwQvKRpE -O model/skill_gap_predictor_model.pkl
gdown --id 1xBfAkzcJ_DwPXAuH8jzyR46OuDHKxAsF -O model/skill_assessment_model.pkl
gdown --id 1LGuOcb2Ecq5NSNgNO2vLsh8pTWS_3MgW -O model/rf_model.pkl
gdown --id 17YV0kR2FCCgcp2vuBfELMYg9hTPSMfa9 -O model/scaler.pkl

# (Optional) Collect static files (Django)
python manage.py collectstatic --noinput
