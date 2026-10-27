
import streamlit as st

from src.model_loader import load_dataset, load_model
from src.prediction import predict_car_price


def _apply_styles():
    """Apply a consistent, professional automotive UI theme."""
    st.markdown(
        """
        <style>
        .stApp {
            background: #f4f7fb;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .hero {
            background: linear-gradient(120deg, #101c36, #243f69);
            padding: 2.2rem 2.4rem;
            border-radius: 18px;
            margin-bottom: 1.5rem;
            color: white;
        }

        .hero h1 {
            color: white;
            font-size: 2.25rem;
            font-weight: 750;
            margin: 0 0 0.55rem 0;
            letter-spacing: -0.7px;
        }

        .hero p {
            color: #dce6f5;
            font-size: 1.02rem;
            line-height: 1.6;
            margin: 0;
        }

        .hero-tag {
            display: inline-block;
            background: rgba(255,255,255,0.12);
            color: #e8efff;
            padding: 0.35rem 0.7rem;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-bottom: 1rem;
            letter-spacing: 0.5px;
        }

        .section-heading {
            color: #172744;
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .section-description {
            color: #68768b;
            font-size: 0.9rem;
            margin-bottom: 1.1rem;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            padding: 1rem 1.25rem;
            box-shadow: 0 3px 12px rgba(24, 39, 75, 0.035);
        }

        label, [data-testid="stWidgetLabel"] p {
            color: #27364e !important;
            font-weight: 600 !important;
            font-size: 0.91rem !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 9px !important;
            border-color: #d5deea !important;
            background: #ffffff !important;
            min-height: 43px;
        }

        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="input"] > div:hover {
            border-color: #7894bd !important;
        }

        input, textarea {
            color: #172744 !important;
        }

        [data-testid="stNumberInput"] button {
            color: #314969 !important;
        }

        .stButton > button {
            border-radius: 10px;
            min-height: 3.1rem;
            font-size: 1rem;
            font-weight: 700;
            border: none;
            transition: all 0.15s ease-in-out;
        }

        .stButton > button[kind="primary"] {
            background: #2458a6;
            color: white;
        }

        .stButton > button[kind="primary"]:hover {
            background: #194783;
            box-shadow: 0 4px 12px rgba(36, 88, 166, 0.2);
        }

        [data-testid="stMetric"] {
            background: #edf4ff;
            border: 1px solid #d7e5fb;
            border-radius: 13px;
            padding: 1.2rem 1.4rem;
        }

        [data-testid="stMetricLabel"] p {
            color: #435775 !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: #173d79 !important;
            font-weight: 750 !important;
        }

        .footer-note {
            color: #718096;
            text-align: center;
            font-size: 0.82rem;
            margin-top: 2rem;
            line-height: 1.6;
        }

        div[data-testid="stExpander"] {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
        }

        @media (max-width: 700px) {
            .block-container {
                padding: 1rem 0.8rem 2rem 0.8rem;
            }

            .hero {
                padding: 1.5rem;
            }

            .hero h1 {
                font-size: 1.7rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header():
    """Display the application hero section."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-tag">SMART VEHICLE VALUATION</div>
            <h1>Used Car Price Estimator</h1>
            <p>
                Discover an estimated resale value for your car.
                Enter its details below to get a data-driven price estimate.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_options(data, column):
    """Return sorted, unique, non-empty options."""
    return sorted(data[column].dropna().unique().tolist())


def _select_option(label, options, placeholder, disabled=False):
    """Render a dropdown with no automatic selection."""
    return st.selectbox(
        label,
        options=options,
        index=None,
        placeholder=placeholder,
        disabled=disabled,
    )


def _render_basic_details(data):
    """Collect the basic vehicle details."""
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">01 · Basic information</div>'
            '<div class="section-description">'
            'Identify the car and its usage.'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            make = _select_option(
                "Brand / Manufacturer",
                _get_options(data, "make"),
                "Choose a brand",
            )

        with col2:
            available_models = (
                sorted(
                    data.loc[data["make"] == make, "model"]
                    .dropna()
                    .unique()
                    .tolist()
                )
                if make
                else []
            )

            model_name = _select_option(
                "Car Model",
                available_models,
                "Choose a brand first" if not make else "Choose a model",
                disabled=not bool(make),
            )

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input(
                "Manufacturing Year",
                min_value=1988,
                max_value=2022,
                value=None,
                placeholder="Enter year",
                step=1,
            )

        with col2:
            kilometer = st.number_input(
                "Kilometers Driven",
                min_value=0,
                max_value=2_000_000,
                value=None,
                placeholder="Enter total kilometers",
                step=1,
            )

    return {
        "make": make,
        "model": model_name,
        "year": year,
        "kilometer": kilometer,
    }


def _render_vehicle_details(data):
    """Collect categorical vehicle details."""
    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">02 · Vehicle details</div>'
            '<div class="section-description">'
            'Provide the vehicle configuration and seller information.'
            '</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            fuel_type = _select_option(
                "Fuel Type",
                _get_options(data, "fuel_type"),
                "Choose fuel type",
            )
            transmission = _select_option(
                "Transmission",
                _get_options(data, "transmission"),
                "Choose transmission",
            )
            location = _select_option(
                "Vehicle Location",
                _get_options(data, "location"),
                "Choose location",
            )
            color = _select_option(
                "Exterior Color",
                _get_options(data, "color"),
                "Choose color",
            )

        with col2:
            owner = _select_option(
                "Ownership History",
                _get_options(data, "owner"),
                "Choose ownership history",
            )
            seller_type = _select_option(
                "Seller Type",
                _get_options(data, "seller_type"),
                "Choose seller type",
            )
            drivetrain = _select_option(
                "Drivetrain",
                _get_options(data, "drivetrain"),
                "Choose drivetrain",
            )

    return {
        "fuel_type": fuel_type,
        "transmission": transmission,
        "location": location,
        "color": color,
        "owner": owner,
        "seller_type": seller_type,
        "drivetrain": drivetrain,
    }


def _optional_number(label, minimum, placeholder, step=1):
    """Render an optional numeric field."""
    return st.number_input(
        label,
        min_value=minimum,
        value=None,
        placeholder=placeholder,
        step=step,
    )


def _render_technical_details():
    """Collect optional technical specifications."""
    with st.expander(
        "03 · Technical specifications (optional)",
        expanded=False,
    ):
        st.caption(
            "Add specifications if available. Unknown values can remain blank."
        )

        col1, col2 = st.columns(2)

        with col1:
            engine = _optional_number(
                "Engine Capacity (cc)", 0, "e.g. 1498"
            )
            max_power = _optional_number(
                "Maximum Power (bhp)", 0.0, "e.g. 116", 1.0
            )
            max_torque = _optional_number(
                "Maximum Torque (Nm)", 0.0, "e.g. 200", 1.0
            )
            length = _optional_number(
                "Length (mm)", 0, "e.g. 4370"
            )

        with col2:
            width = _optional_number(
                "Width (mm)", 0, "e.g. 1770"
            )
            height = _optional_number(
                "Height (mm)", 0, "e.g. 1545"
            )
            seating_capacity = _optional_number(
                "Seating Capacity", 1, "e.g. 5"
            )
            fuel_tank_capacity = _optional_number(
                "Fuel Tank Capacity (litres)",
                0.0,
                "e.g. 50",
                1.0,
            )

    return {
        "engine": engine,
        "max_power": max_power,
        "max_torque": max_torque,
        "length": length,
        "width": width,
        "height": height,
        "seating_capacity": seating_capacity,
        "fuel_tank_capacity": fuel_tank_capacity,
    }


def _validate_details(car_details):
    """Return labels of missing required inputs."""
    required_fields = {
        "make": "Brand / Manufacturer",
        "model": "Car Model",
        "year": "Manufacturing Year",
        "kilometer": "Kilometers Driven",
        "fuel_type": "Fuel Type",
        "transmission": "Transmission",
        "location": "Vehicle Location",
        "color": "Exterior Color",
        "owner": "Ownership History",
        "seller_type": "Seller Type",
        "drivetrain": "Drivetrain",
    }

    return [
        label
        for field, label in required_fields.items()
        if car_details[field] is None
    ]


def _render_prediction(model, car_details):
    """Validate inputs and show the predicted resale price."""
    st.markdown("")

    if not st.button(
        "Estimate Resale Price",
        type="primary",
        use_container_width=True,
    ):
        return

    missing = _validate_details(car_details)

    if missing:
        st.warning(
            "Please complete these fields: " + ", ".join(missing)
        )
        return

    try:
        predicted_price = predict_car_price(model, car_details)

        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">Your price estimate</div>',
                unsafe_allow_html=True,
            )
            st.metric(
                "Estimated Resale Value",
                f"₹{predicted_price:,.0f}",
            )
            st.caption(
                "This is a model-based estimate, not a guaranteed sale price. "
                "Actual value depends on vehicle condition, service history, "
                "local demand, and prevailing market conditions."
            )

    except (ValueError, TypeError, KeyError) as error:
        st.error(f"Unable to generate a prediction: {error}")


def run_app():
    """Build and run the application."""
    st.set_page_config(
        page_title="Used Car Price Estimator",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _apply_styles()
    _render_header()

    try:
        model = load_model()
        data = load_dataset()
    except (FileNotFoundError, OSError, ValueError) as error:
        st.error(f"The application could not load its required files: {error}")
        return

    basic_details = _render_basic_details(data)
    vehicle_details = _render_vehicle_details(data)
    technical_details = _render_technical_details()

    car_details = {
        **basic_details,
        **vehicle_details,
        **technical_details,
    }

    _render_prediction(model, car_details)

    st.markdown(
        """
        <div class="footer-note">
            Used Car Price Estimator · Powered by machine learning<br>
            Predictions are indicative and should be verified against current
            market listings.
        </div>
        """,
        unsafe_allow_html=True,
    )
