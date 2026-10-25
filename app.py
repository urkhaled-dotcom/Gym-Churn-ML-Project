"""Arabic demo dashboard for the gym churn model. Run with: streamlit run app.py"""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MODEL_FILE = ROOT / "models" / "gym_churn_model.joblib"
DATA_FILE = ROOT / "data" / "gym_churn_us.csv"

st.set_page_config(page_title="تحليل مخاطر انسحاب المشتركين", page_icon="🏋️", layout="wide")
st.markdown("""
<style>
.main .block-container {direction: rtl; text-align: right; max-width: 1400px;}
[data-testid="stSidebar"] {direction: rtl; text-align: right;}
[data-testid="stMetricLabel"] {direction: rtl; text-align: right;}
</style>
<div dir="rtl">
<h1>🏋️ نظام تحليل مخاطر انسحاب المشتركين</h1>
<p>لوحة استكشافية لنتائج نموذج تعلم آلة تجريبي، مع مقارنة النماذج وتجربة تنبؤ توضيحية.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_FILE.exists() or not (REPORTS / "final_results.json").exists():
    st.error("لم يتم العثور على ملفات النموذج والتقرير. شغّل أولًا: python -m src.train --data data/gym_churn_us.csv --out .")
    st.stop()

with open(REPORTS / "final_results.json", encoding="utf-8") as f:
    results = json.load(f)
bundle = joblib.load(MODEL_FILE)
data = pd.read_csv(DATA_FILE)
cv = pd.read_csv(REPORTS / "model_comparison_cv.csv")
importance = pd.read_csv(REPORTS / "feature_importance_permutation.csv")

m = results["final_test_metrics_selected_threshold"]
threshold = results["threshold_selection"]["selected_threshold"]
best_name = results["best_model"]
feature_set = "مهندسة" if results["best_feature_set"] == "engineered" else "أصلية"

k1, k2, k3, k4 = st.columns(4)
k1.metric("عدد السجلات", f"{len(data):,}")
k2.metric("نسبة الانسحاب في البيانات", f"{data['Churn'].mean():.1%}")
k3.metric("أفضل PR-AUC في CV", f"{results['cv_best_mean_pr_auc']:.1%}")
k4.metric("PR-AUC على الاختبار", f"{m['pr_auc_average_precision']:.1%}")

st.warning("هذا نموذج تعليمي على مجموعة بيانات منشورة؛ لا يثبت وحده الأداء في نادٍ حقيقي. يجب التحقق من توقيت الخصائص، ومنحنى المعايرة، والنتائج على بيانات محلية قبل استخدامه تشغيليًا.")

tab1, tab2, tab3, tab4 = st.tabs(["ملخص المشروع", "مقارنة النماذج", "نتيجة الاختبار", "جرّب تنبؤًا"])

with tab1:
    st.subheader("ما المشكلة التي يحاول النموذج المساعدة فيها؟")
    st.write("تحليل خصائص العضوية وأنماط الحضور لتقدير احتمال تطابق سجل المشترك مع فئة الانسحاب المسجلة في البيانات. يمكن أن تساعد هذه الإشارة الإدارة على توجيه التواصل، لكنها ليست حكمًا مؤكدًا على أي شخص.")
    left, right = st.columns(2)
    with left:
        st.markdown("**توزيع الفئة المستهدفة**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        counts = data.Churn.value_counts().reindex([0, 1])
        ax.bar(["Retained (0)", "Churned (1)"], counts.values, color=["#357E9A", "#D77A61"])
        ax.set_ylabel("Members")
        ax.set_title("Target distribution")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("**أكثر الخصائص حساسية لأداء النموذج**")
        top = importance.head(8).sort_values("importance_mean")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(top.feature, top.importance_mean, color="#3D9A91")
        ax.set_xlabel("Decrease in average precision")
        ax.set_title("Permutation importance")
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.markdown("**النموذج المختار**")
    st.write(f"{best_name} مع مجموعة الخصائص {feature_set}. جرى اختياره اعتمادًا على متوسط PR-AUC في التحقق المتقاطع، وليس على الاختبار النهائي.")

with tab2:
    st.subheader("مقارنة النماذج والخصائص")
    view = cv[["model", "feature_set", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].copy()
    view["feature_set"] = view.feature_set.map({"original":"أصلية", "engineered":"مهندسة"})
    view = view.rename(columns={"model":"النموذج", "feature_set":"مجموعة الخصائص", "accuracy":"Accuracy", "balanced_accuracy":"Balanced Accuracy", "precision":"Precision", "recall":"Recall", "f1":"F1", "roc_auc":"ROC-AUC", "pr_auc":"PR-AUC"})
    st.dataframe(view.style.format({c:"{:.3f}" for c in view.columns if c not in ["النموذج", "مجموعة الخصائص"]}), use_container_width=True, hide_index=True)
    c = cv.sort_values("pr_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(c)*0.32)))
    labels = c.model + " — " + c.feature_set.map({"original":"Original", "engineered":"Engineered"})
    colors = ["#357E9A" if x == "original" else "#8D79B8" for x in c.feature_set]
    ax.barh(labels, c.pr_auc, color=colors)
    ax.set_xlim(0, 1); ax.set_xlabel("Mean CV PR-AUC"); ax.set_title("Model comparison")
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with tab3:
    st.subheader("التقييم النهائي")
    st.caption(f"عتبة القرار المختارة من تنبؤات التدريب خارج الطية فقط: {threshold:.2f}")
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
    cols[1].metric("Precision", f"{m['precision']:.1%}")
    cols[2].metric("Recall", f"{m['recall']:.1%}")
    cols[3].metric("F1", f"{m['f1']:.1%}")
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0,1], ["Actual 0", "Actual 1"])
    ax.set_title("Confusion matrix at selected threshold")
    for (i,j), v in np.ndenumerate(cm): ax.text(j,i,str(v),ha="center",va="center",fontsize=14)
    fig.colorbar(im, ax=ax, fraction=.05); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.write(f"TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}.")
    st.write("خفض العتبة قد يزيد اكتشاف الانسحاب لكنه يخلق إنذارات كاذبة أكثر. يجب أن يعكس الاختيار كلفة كل نوع من الأخطاء للنادي.")

with tab4:
    st.subheader("تجربة تنبؤ توضيحية")
    st.write("أدخل خصائص مشترك افتراضي. لا تدخل اسمًا أو رقم هاتف أو أي بيانات تعريفية؛ هذه البيانات لا يحتاجها النموذج.")
    features = bundle["feature_columns"]
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender (ترميز المصدر)", [0, 1], index=0)
            near = st.selectbox("Near_Location — قريب من النادي؟", [1, 0], index=0)
            partner = st.selectbox("Partner — شركة شريكة؟", [0, 1], index=0)
            promo = st.selectbox("Promo_friends — إحالة صديق؟", [0, 1], index=0)
            phone = st.selectbox("Phone — رقم مسجل؟", [1, 0], index=0)
        with c2:
            contract = st.selectbox("Contract_period — مدة العقد بالأشهر", [1, 6, 12], index=1)
            group = st.selectbox("Group_visits — حصص جماعية؟", [0, 1], index=1)
            age = st.slider("Age — العمر", 18, 41, 29)
            charges = st.number_input("Avg_additional_charges_total", min_value=0.0, max_value=600.0, value=120.0, step=5.0)
            months_left = st.slider("Month_to_end_contract", 1, int(contract), min(5, int(contract)))
        with c3:
            lifetime = st.slider("Lifetime — مدة العلاقة بالأشهر", 0, 31, 4)
            freq_total = st.slider("Avg_class_frequency_total", 0.0, 6.2, 1.8, 0.1)
            freq_current = st.slider("Avg_class_frequency_current_month", 0.0, 6.2, 1.2, 0.1)
        go = st.form_submit_button("احسب تقدير النموذج")
    if go:
        row = pd.DataFrame([{
            "gender": gender, "Near_Location": near, "Partner": partner, "Promo_friends": promo,
            "Phone": phone, "Contract_period": contract, "Group_visits": group, "Age": age,
            "Avg_additional_charges_total": charges, "Month_to_end_contract": months_left,
            "Lifetime": lifetime, "Avg_class_frequency_total": freq_total,
            "Avg_class_frequency_current_month": freq_current,
        }])[features]
        probability = float(bundle["pipeline"].predict_proba(row)[0,1])
        pred = int(probability >= threshold)
        if pred == 1:
            st.error(f"درجة النموذج: {probability:.1%} — تقع فوق عتبة القرار {threshold:.2f}؛ يطابق هذا التصنيف فئة الانسحاب في البيانات.")
        else:
            st.success(f"درجة النموذج: {probability:.1%} — تقع تحت عتبة القرار {threshold:.2f}؛ يطابق هذا التصنيف فئة عدم الانسحاب في البيانات.")
        st.caption("هذه النتيجة توضيحية وغير مؤكدة، وقد لا تعكس أداءً واقعيًا خارج مجموعة البيانات.")

st.divider()
st.caption("المشروع تعليمي بحثي. لا تستخدم مخرجاته وحدها لاتخاذ قرارات مؤثرة على الأعضاء. راجع README والتقرير النهائي لفهم حدود البيانات.")
