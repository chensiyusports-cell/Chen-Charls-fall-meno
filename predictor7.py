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
Age = st.number_input(
    "Age (years)",
    min_value=0.0,
    step=0.1
)

Height = st.number_input(
    "Height (cm)",
    min_value=0.0,
    step=0.1
)

body_mass = st.number_input(
    "Body Weight (kg)",
    min_value=0.0,
    step=0.1
)

Waist_Circumference = st.number_input(
    "Waist Circumference (cm)",
    min_value=0.0,
    step=0.1
)

Pulse = st.number_input(
    "Resting Heart Rate (bpm)",
    min_value=0.0,
    step=0.1
)

SBP = st.number_input(
    "SBP (mmHg)",
    min_value=0.0,
    step=0.1
)

DBP = st.number_input(
    "DBP (mmHg)",
    min_value=0.0,
    step=0.1
)

PP = st.number_input(
    "Pulse Pressure (mmHg)",
    min_value=0.0,
    step=0.1
)

time_5_sts = st.number_input(
    "5-times Sit-to-Stand Test Time (s)",
    min_value=0.0,
    step=0.1
)

pef_mean = st.number_input(
    "Peak Expiratory Flow (L/min)",
    min_value=0.0,
    step=0.1
)

unDomain_2KG = st.number_input(
    "Maximum Non-dominant Arm Biceps Curl Repetitions with 2 kg Load",
    min_value=0.0,
    step=0.1
)


# ===== CES-D 10: 自动计算（0–30）=====
st.markdown("### CES-D 10 (Past Week)")

CESD_OPTIONS = [0, 1, 2, 3]
CESD_LABELS = {
    0: "Rarely or none of the time (<1 day)",
    1: "Some or a little of the time (1–2 days)",
    2: "Occasionally or a moderate amount of the time (3–4 days)",
    3: "Most or all of the time (5–7 days)"
}

def cesd_item(question: str, key: str) -> int:
    return st.selectbox(
        question,
        options=CESD_OPTIONS,
        format_func=lambda x: CESD_LABELS[x],
        key=key
    )

# 8 个负向题（直接加分）
felt_depressed1 = cesd_item("How often did you feel depressed during the past week?", "felt_depressed1")
everything_was_an_effort = cesd_item("How often did you feel that everything was an effort during the past week?", "everything_was_an_effort")
felt_fearful = cesd_item("How often did you feel fearful during the past week?", "felt_fearful")
poor_sleep = cesd_item("How often did you have restless sleep during the past week?", "poor_sleep")
felt_lonely = cesd_item("How often did you feel lonely during the past week?", "felt_lonely")
felt_could_not_go_on_with_my_life = cesd_item("How often did you feel you could not go on with your life during the past week?", "felt_could_not_go_on_with_my_life")
bothered_by_trivial_things = cesd_item("How often were you bothered by things that don't usually bother you during the past week?", "bothered_by_trivial_things")
hard_to_concentrate = cesd_item("How often did you have trouble keeping your mind on what you were doing during the past week?", "hard_to_concentrate")

# 2 个正向题（需要反向计分：3-x）
felt_happy = cesd_item("How often did you feel happy during the past week?", "felt_happy")
hopeful_about_future = cesd_item("How often did you feel hopeful about the future during the past week?", "hopeful_about_future")

CESD10 = (
    felt_depressed1
    + everything_was_an_effort
    + felt_fearful
    + poor_sleep
    + felt_lonely
    + felt_could_not_go_on_with_my_life
    + bothered_by_trivial_things
    + hard_to_concentrate
    + (3 - felt_happy)
    + (3 - hopeful_about_future)
)

st.number_input(
    "CES-D 10 Total Score (0–30)",
    min_value=0,
    max_value=30,
    step=1,
    value=int(CESD10),
    disabled=True
)

Fallen_down_history = st.selectbox(
    "Have you fallen down before?",
    options=[0, 1],
    format_func=lambda x: "yes" if x == 1 else "no"
)

felt_depressed = st.selectbox(
    "How often did you feel depressed during the past week?",
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
     "**Independent in daily activities**  \n"
    "Can you complete **ALL** of the following tasks independently?\n\n"
    "- Bathing or showering\n"
    "- Eating\n"
    "- Getting into or out of bed\n"
    "- Using the toilet\n"
    "- Controlling urination and defecation\n"
    "- Doing household chores\n"
    "- Preparing hot meals\n"
    "- Shopping for groceries",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
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
    X_test1 = X_test[feature_names]
    lime_explainer = LimeTabularExplainer(
        training_data=X_test1.values,
        feature_names=feature_names,
        class_names=["低XX风险", "高XX风险"],
        mode="classification"
    )

    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba,
        num_features=13
    )

    lime_html = lime_exp.as_html(show_table=True)
    st.components.v1.html(lime_html, height=600, scrolling=True)








