import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image


# Load model
@st.cache_resource
def load_model():
    return joblib.load('model/event_budget_predictor.pkl')


# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #121212;
            color: #f5f5f5;
        }
        .stButton > button {
            background-color: green;
            color: white;
            font-size: 18px;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            transition: background-color 0.3s ease, transform 0.2s ease;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            color: white;
            font-weight: bold;
            transform: scale(1.05);
            box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.15);
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
        .stExpander {
            background-color: #2c3e50;
            color: white;
        }
        .stExpanderHeader {
            color: white;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            color: #bdc3c7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown("<h1 style='text-align: center; color: #f1c40f;'>🎉 CUET Event Budget Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #bdc3c7; margin-bottom: 40px;'>AI-Powered Event Cost Estimation</h4>", unsafe_allow_html=True)
# Sidebar for Info/Stats
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        This app predicts total event budget using machine learning.
        
        **Key Features:** 
        - Trained on 900+ events held in CUET
        - 95% prediction accuracy
        - Real-time calculations
        """
    )


# Main Content
def main():
    # Feature Input Section
    with st.container():
        st.subheader("📝 Event Details")
        col1, col2 = st.columns(2)

        with col1:
            event_type = st.selectbox(
                "Event Type",
                ["Cultural Fest", "Tech Fest", "Sports Tournament", "Seminar", "Religious Event"],
                index=1
            )

            event_scale = st.selectbox(
                "Event Scale",
                ["Intra-University", "Inter-University", "National", "International"],
                index=1
            )

            participants_range = st.selectbox(
                "Expected Participants",
                ["<500", "500-1000", ">1000"],
                index=1
            )

            duration = st.slider(
                "Event Duration (days)",
                min_value=1,
                max_value=7,
                value=3
            )

        with col2:
            equipment = st.number_input(
                "Equipment Costs (Tk)",
                min_value=0,
                max_value=500000,
                value=15000,
                step=1000
            )

            staff = st.number_input(
                "Staff Costs (Tk)",
                min_value=0,
                max_value=1000000,
                value=35000,
                step=1000
            )

            marketing = st.number_input(
                "Marketing Costs (Tk)",
                min_value=0,
                max_value=200000,
                value=8000,
                step=500
            )

            sponsorship = st.number_input(
                "Sponsorship Amount (Tk)",
                min_value=0,
                value=5000
            )

            last_budget = st.number_input(
                "Last Event Budget (Tk)",
                min_value=0,
                value=25000
            )

    # Prediction Section
    st.divider()
    with st.container():
        if st.button("### Predict Total Budget", use_container_width=True):
            try:
                model = load_model()
                input_data = pd.DataFrame({
                    'Event_Type': [event_type],
                    'Event_Scale': [event_scale],
                    'Expected_Participants_Range': [participants_range],
                    'Event_Duration': [duration],
                    'Equipment_Costs': [equipment],
                    'Staff_Costs': [staff],
                    'Marketing_Costs': [marketing],
                    'Sponsorship_Amount': [sponsorship],
                    'Last_Event_Budget': [last_budget]
                })

                prediction = np.expm1(model.predict(input_data))[0]

                st.success(f"""
                **📊 Prediction Results**  
                Estimated Total Cost: Tk {prediction:,.2f}  
                Cost Breakdown:  
                - Base Costs: Tk {(equipment + staff + marketing):,.2f}  
                - Net Sponsorship: Tk {sponsorship:,.2f}
                """)

                # Visual Progress Bar
                budget_ratio = (prediction / last_budget) if last_budget > 0 else 1
                st.progress(min(budget_ratio, 1))
                st.caption(f"Compared to last budget (Tk {last_budget:,.2f})")

            except Exception as e:
                st.error(f"Error in prediction: {str(e)}")

    # Footer
    st.markdown("---")
    st.caption("<div class='footer'>🔍 Predictions are estimates - Actual costs may vary</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
