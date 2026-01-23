# ==================== 0. 导入库 ====================
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import warnings

warnings.filterwarnings("ignore")

# ==================== 1. 基础配置 ====================
# 加载训练好的随机森林模型（确保 RF.pkl 与脚本同目录）
model = joblib.load("xgb_model.pkl")

# 加载测试数据（用于 LIME 解释，确保 X_test.csv 与脚本同目录）
X_test = pd.read_csv("traindata.csv")

# 特征名称（需与你训练模型时的特征顺序一致）
feature_names = [
    'time_5_sts',
    'body_mass',
    'SBP',
    'Height',
    'CESD10',
    'Fallen_down_history',
    'unDomain_2KG',
    'Waist_Circumference',
    'DBP',
    'pef_mean',
    'Pulse',
    'Age',
    'felt_depressed',
    'self_rated_health1',
    'daily_activity_ability',
    'PP'
]

# ==================== 2. Streamlit 页面配置 ====================
st.set_page_config(page_title="Fall Risk Prediction", layout="wide")
st.title("Fall risk prediction of Chinese post-menopausal women")
st.markdown("Please fill the following blank to predict")

# ==================== 3. 特征输入组件（按编码规则设计） ====================
time_5_sts = st.number_input(
    "time_5_sts",
    min_value=0.0,
    step=0.1
)

body_mass = st.number_input(
    "body_mass",
    min_value=0.0,
    step=0.1
)

SBP = st.number_input(
    "SBP",
    min_value=0.0,
    step=0.1
)

Height = st.number_input(
    "Height",
    min_value=0.0,
    step=0.1
)

CESD10 = st.number_input(
    "CESD10",
    min_value=0,
    step=1
)

Fallen_down_history = st.selectbox(
    "Have you fallen down before?",
    options=[0, 1],
    format_func=lambda x: "yes" if x == 0 else "no"
)

unDomain_2KG = st.number_input(
    "unDomain_2KG",
    min_value=0.0,
    step=0.1
)

Waist_Circumference = st.number_input(
    "Waist_Circumference",
    min_value=0.0,
    step=0.1
)

DBP = st.number_input(
    "DBP",
    min_value=0.0,
    step=0.1
)

pef_mean = st.number_input(
    "pef_mean",
    min_value=0.0,
    step=0.1
)

Pulse = st.number_input(
    "Pulse",
    min_value=0.0,
    step=0.1
)

Age = st.number_input(
    "Age",
    min_value=0.0,
    step=0.1
)

felt_depressed = st.selectbox(
    "Felt depressed in the past week",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Rarely or none of the time (<1 day)",
        1: "Some or a little of the time (1–2 days)",
        2: "Occasionally or a moderate amount of the time (3–4 days)",
        3: "Most or all of the time (5–7 days)"
    }[x]
)

self_rated_health1 = st.selectbox(
    "How would you rate your health?",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: {
        1: "Very good",
        2: "Good",
        3: "Fair",
        4: "Poor",
        5: "Very poor"
    }[x]
)

daily_activity_ability = st.selectbox(
    "daily_activity_ability",
    options=[0, 1],
    format_func=lambda x: "yes" if x == 0 else "no"
)

PP = st.number_input(
    "PP",
    min_value=0.0,
    step=0.1
)
# ==================== 4. 数据处理与预测 ====================
feature_values = [
    time_5_sts, body_mass, SBP, Height, CESD10,
    Fallen_down_history, unDomain_2KG, Waist_Circumference,
    DBP, pef_mean, Pulse, Age,
    felt_depressed, self_rated_health1, daily_activity_ability, PP
]


features = np.array([feature_values])

if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]          # 0: 低风险, 1: 高风险
    predicted_proba = model.predict_proba(features)[0]    # 概率

    # ==================== 5. 结果展示 ====================
    st.subheader("📊 Prediction Results")
    risk_label = "高风险" if predicted_class == 1 else "低风险"

    st.write(f"**风险等级：{predicted_class}（{risk_label}）**")
    st.write(
        f"**风险概率：** "
        f"低风险 {predicted_proba[0]:.2%} ｜ 高风险 {predicted_proba[1]:.2%}"
    )

    # 个性化建议
    st.subheader("💡 健康建议")
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (
            f"模型预测您的 XX 风险为 **高风险**（概率 {probability:.1f}%）。"
            "建议尽快前往医疗机构进行全面评估，重点关注营养摄入、"
            "睡眠质量与心理健康等方面，并根据自身情况增加适度体育锻炼，"
            "改善生活环境。"
        )
    else:
        advice = (
            f"模型预测您的 XX 风险为 **低风险**（概率 {probability:.1f}%）。"
            "请继续保持良好的生活方式，合理饮食、规律作息，并定期进行健康检查。"
        )

    st.success(advice)

    # ==================== 6. LIME 解释 ====================
    st.subheader("🔍 LIME 特征贡献解释")

    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=feature_names,
        class_names=["低XX风险", "高XX风险"],
        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba,
        num_features=16
    )

    lime_html = lime_exp.as_html(show_table=True)
    st.components.v1.html(lime_html, height=600, scrolling=True)


