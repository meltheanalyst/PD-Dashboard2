import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import hashlib
from datetime import datetime, timedelta
import numpy as np
import sys
print("Python executable:", sys.executable)  # Debug
print("Python path:", sys.path)  # Debug
try:
    import sklearn
    print("scikit-learn version:", sklearn.__version__)  # Debug
except ModuleNotFoundError:
    print("scikit-learn not found in this environment")  # Debug
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io
import base64

# Load environment variables
load_dotenv()
PASSWORD = os.getenv("DASHBOARD_PASSWORD", "sales2025")

# Password protection
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "failed_attempts" not in st.session_state:
        st.session_state["failed_attempts"] = 0
    if "account_locked" not in st.session_state:
        st.session_state["account_locked"] = False

    if st.session_state["account_locked"]:
        st.error("Account locked due to too many failed attempts. Contact admin.")
        st.stop()

    if not st.session_state["password_correct"]:
        st.title("AI-Solutions Sales Dashboard - Login")
        st.markdown("Enter the password to access the dashboard. Contact admin for assistance.")
        password = st.text_input("Enter Password", type="password")
        if st.button("Login"):
            if password == PASSWORD:
                st.session_state["password_correct"] = True
                st.session_state["failed_attempts"] = 0
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.session_state["failed_attempts"] += 1
                remaining = 3 - st.session_state["failed_attempts"]
                if remaining > 0:
                    st.warning(f"Incorrect password. {remaining} attempts left.")
                else:
                    st.session_state["account_locked"] = True
                    st.error("Account locked. Please contact admin.")
        st.stop()

check_password()

# Data validation and anonymization
def validate_and_clean_data(df):
    initial_rows = len(df)
    required_columns = ["Timestamp", "Request_Type", "Product_Type", "Country", "User_Agent", "Estimated_Revenue", "Response_Code"]
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"CSV file missing required columns: {', '.join(missing_cols)}. Please check the CSV file.")
        st.stop()
    
    # Clean data
    df = df.dropna(subset=["Timestamp", "Request_Type", "Product_Type", "Country"]).copy()
    df = df.drop_duplicates()
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M", errors="coerce")
    except Exception as e:
        st.error(f"Timestamp format error: {str(e)}. Expected format: DD/MM/YYYY HH:MM. Please check the CSV file.")
        st.stop()
    df = df.dropna(subset=["Timestamp"])
    
    # Parse Device_Type before anonymizing User_Agent
    df["Device_Type"] = df["User_Agent"].apply(parse_device)
    
    # Anonymize User_Agent
    df["User_Agent"] = df["User_Agent"].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    
    # Validate Estimated_Revenue
    df["Estimated_Revenue"] = pd.to_numeric(df["Estimated_Revenue"], errors="coerce").fillna(0)
    
    cleaned_rows = len(df)
    if cleaned_rows == 0:
        st.warning("No valid data remains after cleaning. Using fallback data.")
        # Create fallback data
        df = pd.DataFrame({
            "Timestamp": [datetime.now() - timedelta(days=i) for i in range(5)],
            "Request_Type": ["Inquiry", "Job Request", "Demo Request", "Promo Event", "Inquiry"],
            "Product_Type": ["Product A", "Product B", "Product A", "Product C", "Product B"],
            "Country": ["USA", "UK", "Germany", "France", "Canada"],
            "User_Agent": [hashlib.sha256(f"Agent_{i}".encode()).hexdigest() for i in range(5)],
            "Estimated_Revenue": [1000, 2000, 1500, 3000, 1200],
            "Response_Code": [200] * 5,
            "Device_Type": ["Mobile", "Desktop", "Mobile", "Desktop", "Other"]
        })
        df.to_csv("web_server_logs (3).csv", index=False)
        cleaned_rows = len(df)
    
    st.info(f"Data validation: {initial_rows - cleaned_rows} invalid rows removed.")
    return df

# Parse Device Type
def parse_device(user_agent):
    if "iPhone" in user_agent or "Android" in user_agent:
        return "Mobile"
    elif "Windows" in user_agent or "Macintosh" in user_agent:
        return "Desktop"
    else:
        return "Other"

# Simulate automated data collection
def simulate_data_collection(df):
    # Ensure Product_Type and Country have valid values
    product_types = df["Product_Type"].unique()
    countries = df["Country"].unique()
    
    if len(product_types) == 0:
        product_types = ["Product A", "Product B", "Product C"]
        st.warning("No valid Product_Type values found. Using default products.")
    if len(countries) == 0:
        countries = ["USA", "UK", "Germany"]
        st.warning("No valid Country values found. Using default countries.")
    
    new_data = pd.DataFrame({
        "Timestamp": [datetime.now() - timedelta(minutes=np.random.randint(0, 1440)) for _ in range(10)],
        "Request_Type": np.random.choice(["Inquiry", "Job Request", "Demo Request", "Promo Event"], size=10),
        "Product_Type": np.random.choice(product_types, size=10),
        "Country": np.random.choice(countries, size=10),
        "User_Agent": [hashlib.sha256(f"Agent_{i}".encode()).hexdigest() for i in range(10)],
        "Estimated_Revenue": np.random.randint(1000, 10000, size=10),
        "Response_Code": [200] * 10,
        "Device_Type": np.random.choice(["Mobile", "Desktop", "Other"], size=10, p=[0.5, 0.4, 0.1])
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv("web_server_logs (3).csv", index=False)
    return df

# Load data
@st.cache_data
def load_data():
    try:
        if not os.path.exists("web_server_log (3).csv"):
            st.error("CSV file 'web_server_logs (3).csv' not found. Creating a default dataset.")
            df = pd.DataFrame({
                "Timestamp": [datetime.now() - timedelta(days=i) for i in range(5)],
                "Request_Type": ["Inquiry", "Job Request", "Demo Request", "Promo Event", "Inquiry"],
                "Product_Type": ["Product A", "Product B", "Product A", "Product C", "Product B"],
                "Country": ["USA", "UK", "Germany", "France", "Canada"],
                "User_Agent": [f"Agent_{i}" for i in range(5)],
                "Estimated_Revenue": [1000, 2000, 1500, 3000, 1200],
                "Response_Code": [200] * 5,
                "Device_Type": ["Mobile", "Desktop", "Mobile", "Desktop", "Other"]
            })
            df.to_csv("web_server_logs (3).csv", index=False)
        df = pd.read_csv("web_server_logs (3).csv")
        df = validate_and_clean_data(df)
        df = simulate_data_collection(df)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}. Please check the CSV file.")
        st.stop()

df = load_data()

# AI Model: Customer behavior clustering
def cluster_customers(df):
    features = df[["Estimated_Revenue", "Timestamp"]].copy()
    features["Days_Since"] = (datetime.now() - features["Timestamp"]).dt.days
    features = features[["Estimated_Revenue", "Days_Since"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(scaled_features)
    df["Cluster"] = df["Cluster"].map({0: "High Value", 1: "Recent Low Value", 2: "Older Low Value"})
    return df

df = cluster_customers(df)

# Sidebar filters and instructions
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("""
### Instructions
1. Use the filters below to segment data by year, month, time period, product, or country.
2. Navigate tabs to view Sales Overview, Regional Analysis, or Performance Trends.
3. Download reports in CSV or PDF format from the 'Sales Overview' tab.
4. Contact admin for password reset or data issues.
""")
year_filter = st.sidebar.selectbox("Year", ["All", "2025"], index=1)
month_filter = st.sidebar.selectbox("Month", ["All"] + list(pd.date_range(start="2025-01-01", end="2025-12-31", freq="M").strftime("%B")), index=0)
time_filter = st.sidebar.selectbox("Time Period", ["All", "2025-01"])
product_filter = st.sidebar.multiselect("Product Type", df["Product_Type"].unique(), default=df["Product_Type"].unique())
country_filter = st.sidebar.multiselect("Country", df["Country"].unique(), default=df["Country"].unique())
cluster_filter = st.sidebar.multiselect("Customer Cluster", df["Cluster"].unique(), default=df["Cluster"].unique())

# Apply filters
filtered_df = df[
    (df["Timestamp"].dt.year == int(year_filter) if year_filter != "All" else True) &
    (df["Timestamp"].dt.month_name() == month_filter if month_filter != "All" else True) &
    ((df["Timestamp"].dt.strftime("%Y-%m") == time_filter) if time_filter != "All" else True) &
    (df["Product_Type"].isin(product_filter)) &
    (df["Country"].isin(country_filter)) &
    (df["Cluster"].isin(cluster_filter))
].copy()

# Metrics
total_inquiries = len(filtered_df[filtered_df["Request_Type"] == "Inquiry"])
total_jobs = len(filtered_df[filtered_df["Request_Type"] == "Job Request"])
total_demos = len(filtered_df[filtered_df["Request_Type"] == "Demo Request"])
total_promo = len(filtered_df[filtered_df["Request_Type"] == "Promo Event"])
estimated_revenue = filtered_df[filtered_df["Request_Type"] == "Job Request"]["Estimated_Revenue"].sum()
gross_profit = estimated_revenue * 0.40
net_profit = estimated_revenue * 0.10
conversion_rate = (total_jobs / total_inquiries) * 100 if total_inquiries > 0 else 0
demo_conversion_rate = (total_demos / total_inquiries) * 100 if total_inquiries > 0 else 0
avg_revenue = estimated_revenue / total_jobs if total_jobs > 0 else 0
inquiry_std = filtered_df[filtered_df["Request_Type"] == "Inquiry"].groupby(filtered_df["Timestamp"].dt.date)["Request_Type"].count().std()

# Product-specific metrics
product_metrics = filtered_df.groupby("Product_Type").agg(
    Inquiries=("Request_Type", lambda x: (x == "Inquiry").sum()),
    Job_Requests=("Request_Type", lambda x: (x == "Job Request").sum()),
    Demo_Requests=("Request_Type", lambda x: (x == "Demo Request").sum()),
    Promo_Events=("Request_Type", lambda x: (x == "Promo Event").sum()),
    Revenue=("Estimated_Revenue", "sum")
).reset_index()
product_metrics["Gross_Profit"] = product_metrics["Revenue"] * 0.40
product_metrics["Net_Profit"] = product_metrics["Revenue"] * 0.10
product_metrics["Mean_Revenue"] = product_metrics["Revenue"] / product_metrics["Job_Requests"] if product_metrics["Job_Requests"].sum() > 0 else 0
top_product = product_metrics.loc[product_metrics["Revenue"].idxmax(), "Product_Type"] if total_jobs > 0 else "N/A"

# AI Trend Analysis
def predict_trends(df):
    if df.empty or "Timestamp" not in df.columns or not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        st.warning("No valid data for trend prediction. Displaying default trend.")
        return pd.DataFrame({
            "Date": pd.date_range(datetime.now(), periods=30),
            "Predicted_Revenue": [1000 + i * 50 for i in range(30)]
        })
    
    # Group by date while preserving datetime type
    trend_data = df.groupby(df["Timestamp"].dt.floor("D"))["Estimated_Revenue"].sum().reset_index()
    trend_data["Timestamp"] = pd.to_datetime(trend_data["Timestamp"])
    trend_data["Days"] = (trend_data["Timestamp"] - trend_data["Timestamp"].min()).dt.days
    
    if trend_data.empty or len(trend_data) < 2:
        st.warning("Insufficient data for trend prediction. Displaying default trend.")
        return pd.DataFrame({
            "Date": pd.date_range(datetime.now(), periods=30),
            "Predicted_Revenue": [1000 + i * 50 for i in range(30)]
        })
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X = trend_data[["Days"]]
    y = trend_data["Estimated_Revenue"]
    model.fit(X, y)
    future_days = np.array([[trend_data["Days"].max() + i] for i in range(1, 31)])
    predictions = model.predict(future_days)
    return pd.DataFrame({
        "Date": pd.date_range(trend_data["Timestamp"].max() + timedelta(days=1), periods=30),
        "Predicted_Revenue": predictions
    })

trend_predictions = predict_trends(filtered_df)

# Previous period mock
prev_inquiries = total_inquiries * 0.95
prev_jobs = total_jobs * 0.90
prev_profit = net_profit * 0.92
profit_goal = 50000
profit_achievement = (net_profit / profit_goal) * 100 if profit_goal > 0 else 0
prev_achievement = (prev_profit / profit_goal) * 100 if profit_goal > 0 else 0

# Report generation
def generate_pdf_report(metrics, product_metrics, trend_predictions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("AI-Solutions Sales Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    
    # Key Metrics
    data = [
        ["Metric", "Value"],
        ["Total Inquiries", f"{metrics['total_inquiries']:,}"],
        ["Transactions", f"{metrics['total_jobs']:,}"],
        ["Net Profit", f"${metrics['net_profit']:,.0f}"],
        ["Gross Profit", f"${metrics['gross_profit']:,.0f}"],
        ["Conversion Rate", f"{metrics['conversion_rate']:.2f}%"],
        ["Top Product", metrics["top_product"]]
    ]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    # Product Metrics
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Product Metrics", styles["Heading2"]))
    product_data = [product_metrics.columns.tolist()] + product_metrics.values.tolist()
    table = Table(product_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Tabs
tab1, tab2, tab3 = st.tabs(["Sales Overview", "Regional Analysis", "Performance"])

with tab1:
    st.header("Sales Overview")
    st.markdown("View key sales metrics and download reports below.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Inquiries", f"{total_inquiries:,}", delta=f"{int(total_inquiries - prev_inquiries):,} CY Growth")
    col2.metric("Transactions", f"{total_jobs:,}", delta=f"{int(total_jobs - prev_jobs):,} CY Growth")
    col3.metric("Net Profit", f"${net_profit:,.0f}", delta=f"${int(net_profit - prev_profit):,} CY Growth")

    col4, col5, col6 = st.columns(3)
    col4.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    col5.metric("Avg. Revenue per Transaction", f"${avg_revenue:,.2f}")
    col6.metric("Top Product", top_product)

    col7, col8, col9 = st.columns(3)
    col7.metric("Gross Profit", f"${gross_profit:,.0f}")
    col8.metric("Demo Conversion Rate", f"{demo_conversion_rate:.2f}%")
    col9.metric("Promo Events", f"{total_promo:,}")

    with st.expander("Profit Goal Achievement"):
        st.metric("Achievement", f"{profit_achievement:.1f}%", delta=f"{profit_achievement - prev_achievement:.1f}%")
        st.progress(min(profit_achievement / 100, 1.0))

    st.subheader("Product-Specific Metrics")
    st.table(product_metrics[["Product_Type", "Inquiries", "Job_Requests", "Demo_Requests", "Promo_Events", "Revenue", "Gross_Profit", "Net_Profit", "Mean_Revenue"]])

    st.subheader("Generate Report")
    report_frequency = st.selectbox("Report Frequency", ["Daily", "Weekly", "Monthly"])
    report_format = st.selectbox("Report Format", ["CSV", "PDF"])
    if st.button("Download Report"):
        metrics = {
            "total_inquiries": total_inquiries,
            "total_jobs": total_jobs,
            "net_profit": net_profit,
            "gross_profit": gross_profit,
            "conversion_rate": conversion_rate,
            "top_product": top_product
        }
        if report_format == "CSV":
            csv = product_metrics.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"sales_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            pdf_buffer = generate_pdf_report(metrics, product_metrics, trend_predictions)
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name=f"sales_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

with tab2:
    st.header("Regional Analysis")
    st.markdown("Explore inquiries, revenue, and customer behavior by region and product.")
    
    country_counts = filtered_df[filtered_df["Response_Code"] == 200]\
        .groupby("Country")["Request_Type"].count().reset_index(name="Inquiries")
    country_counts = country_counts.sort_values("Inquiries", ascending=False)
    st.plotly_chart(px.choropleth(country_counts, locations="Country", locationmode="country names",
                                  color="Inquiries", color_continuous_scale="Viridis",
                                  title="Inquiries by Country"))

    revenue_by_country = filtered_df.groupby("Country")["Estimated_Revenue"].sum().reset_index()
    revenue_by_country = revenue_by_country.sort_values("Estimated_Revenue", ascending=False)
    st.plotly_chart(px.choropleth(revenue_by_country, locations="Country", locationmode="country names",
                                  color="Estimated_Revenue", color_continuous_scale="Blues",
                                  title="Revenue by Country"))

    st.subheader("Top 5 Countries by Revenue")
    st.table(revenue_by_country.head(5))

    col1, col2 = st.columns(2)
    with col1:
        prod_counts = filtered_df.groupby("Product_Type")["Request_Type"].count().reset_index(name="Count")
        prod_counts = prod_counts.sort_values("Count", ascending=False)
        chart_type = st.selectbox("Chart Type", ["Bar", "Pie"], key="prod_chart")
        if chart_type == "Bar":
            st.plotly_chart(px.bar(prod_counts, x="Product_Type", y="Count", title="Requests by Product"))
        else:
            st.plotly_chart(px.pie(prod_counts, names="Product_Type", values="Count", title="Requests by Product"))
    with col2:
        req_counts = filtered_df.groupby("Request_Type")["Request_Type"].count().reset_index(name="Count")
        req_counts = req_counts.sort_values("Count", ascending=False)
        chart_type = st.selectbox("Chart Type", ["Pie", "Bar"], key="req_chart")
        if chart_type == "Pie":
            st.plotly_chart(px.pie(req_counts, names="Request_Type", values="Count", title="Request Type Distribution"))
        else:
            st.plotly_chart(px.bar(req_counts, x="Request_Type", y="Count", title="Request Type Distribution"))

    st.subheader("Customer Behavior Clusters")
    cluster_counts = filtered_df.groupby(["Cluster", "Country"])["Request_Type"].count().reset_index(name="Count")
    st.plotly_chart(px.bar(cluster_counts, x="Country", y="Count", color="Cluster", title="Customer Clusters by Country"))

    st.subheader("Device Usage Summary")
    device_counts = filtered_df["Device_Type"].value_counts().reset_index()
    device_counts.columns = ["Device", "Count"]
    st.plotly_chart(px.pie(device_counts, names="Device", values="Count", title="Device Type Distribution"))

with tab3:
    st.header("Performance Trends")
    st.markdown("Analyze product performance and predicted sales trends.")
    
    profit_goal_line = pd.DataFrame({"Product_Type": product_metrics["Product_Type"], "Profit_Goal": [5000000] * len(product_metrics)})
    fig_bar_line = px.bar(product_metrics, x="Product_Type", y="Net_Profit", title="Product Profit vs. Profit Goal")
    fig_bar_line.add_scatter(x=profit_goal_line["Product_Type"], y=profit_goal_line["Profit_Goal"], mode="lines",
                             name="Profit Goal ($5,000,000)", line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_bar_line)

    st.subheader("Profit vs. Revenue by Product")
    profit_revenue_data = product_metrics.melt(id_vars=["Product_Type"], value_vars=["Revenue", "Net_Profit"],
                                              var_name="Metric", value_name="Amount")
    chart_type = st.selectbox("Chart Type", ["Bar", "Line"], key="profit_chart")
    if chart_type == "Bar":
        st.plotly_chart(px.bar(profit_revenue_data, x="Product_Type", y="Amount", color="Metric",
                               title="Profit vs. Revenue by Product", barmode="group"))
    else:
        st.plotly_chart(px.line(profit_revenue_data, x="Product_Type", y="Amount", color="Metric",
                                title="Profit vs. Revenue by Product"))

    st.subheader("Sales Trend Prediction (Next 30 Days)")
    st.plotly_chart(px.line(trend_predictions, x="Date", y="Predicted_Revenue", title="Predicted Revenue Trend"))

    st.subheader("Overall Profit Performance")
    st.metric("Overall Profit Performance", f"{profit_achievement:.1f}%", delta=f"{profit_achievement - prev_achievement:.1f}% CY Growth")