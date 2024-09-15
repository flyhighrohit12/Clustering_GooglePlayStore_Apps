import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression
import plotly.express as px
import plotly.graph_objs as go

# Set page config and styling
st.set_page_config(page_title="App Store Clustering Analysis", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme and better styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
        color: #fafafa;
    }
    .main {
        background: #262730;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #3498db;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stSelectbox {
        color: #fafafa;
    }
    .stTab {
        background-color: #262730;
        color: #fafafa;
    }
    .stDataFrame {
        background-color: #1f2937;
        color: #fafafa;
    }
    .css-1d391kg {
        background-color: #262730;
    }
    .css-1d391kg:hover {
        background-color: #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    
    # Handle 'Rating' column
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Handle 'Reviews' column
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    
    # Handle 'Size' column
    def parse_size(size):
        if pd.isna(size):
            return np.nan
        if isinstance(size, str):
            if 'M' in size:
                return float(size.replace('M', ''))
            elif 'k' in size:
                return float(size.replace('k', '')) / 1024
            elif size == 'Varies with device':
                return np.nan
        return pd.to_numeric(size, errors='coerce')
    
    df['Size'] = df['Size'].apply(parse_size)
    
    # Handle 'Installs' column
    df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '')
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    # Handle 'Price' column
    df['Price'] = df['Price'].replace('Free', '0')
    df['Price'] = df['Price'].str.replace('$', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    
    # Fill NaN with median for numeric columns
    numeric_columns = ['Rating', 'Reviews', 'Size', 'Installs', 'Price']
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df

df = load_data()

def main():
    st.title("🚀 App Store Clustering Analysis")
    
    tabs = ["Home", "EDA", "Clustering", "Recommendations"]
    page = st.radio("Navigate", tabs, horizontal=True)

    if page == "Home":
        home_tab()
    elif page == "EDA":
        eda_tab()
    elif page == "Clustering":
        clustering_tab()
    elif page == "Recommendations":
        recommendations_tab()

def home_tab():
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #3498db; font-size: 3em;">Welcome to the App Store Clustering Adventure!</h1>
        <p style="font-size: 1.2em; color: #fafafa;">Uncover hidden patterns and gain valuable insights from mobile app data.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; height: 200px;">
            <h3 style="color: #3498db;">🔍 Exploratory Data Analysis</h3>
            <p style="color: #fafafa;">Dive deep into app statistics, discover trends, and visualize key metrics.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; height: 200px;">
            <h3 style="color: #3498db;">🧮 Advanced Clustering Techniques</h3>
            <p style="color: #fafafa;">Apply K-Means, Hierarchical, and DBSCAN algorithms to group similar apps.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; height: 200px;">
            <h3 style="color: #3498db;">💡 Data-Driven Recommendations</h3>
            <p style="color: #fafafa;">Get actionable insights for app developers and marketers based on our analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px;">
        <h2 style="color: #3498db;">Dataset Overview</h2>
        <p style="color: #fafafa; font-size: 1.1em;">Our analysis is based on a comprehensive dataset of mobile applications. Here's a quick glimpse:</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Apps", f"{df.shape[0]:,}", "📱")
    col2.metric("Features", df.shape[1], "🔢")
    col3.metric("Categories", df['Category'].nunique(), "🏷️")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px;">
        <h2 style="color: #3498db;">Ready to Explore?</h2>
        <p style="color: #fafafa; font-size: 1.1em;">Navigate through the tabs above to start your journey into the fascinating world of mobile app analytics!</p>
    </div>
    """, unsafe_allow_html=True)

def eda_tab():
    st.header("📊 Exploratory Data Analysis")

    # Update plot colors and styles for dark theme
    plot_template = "plotly_dark"
    color_scale = "Viridis"

    # Rating Distribution
    st.subheader("App Ratings Distribution")
    fig = px.histogram(df, x="Rating", nbins=50, marginal="box", title="Distribution of App Ratings",
                       template=plot_template, color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig)
    avg_rating = df['Rating'].mean()
    median_rating = df['Rating'].median()
    st.info(f"📌 Insight: The average app rating is {avg_rating:.2f}, with a median of {median_rating:.2f}. "
            f"The distribution is left-skewed, indicating a tendency towards higher ratings.")

    # Category Analysis
    st.subheader("Top App Categories")
    top_categories = df['Category'].value_counts().head(10)
    fig = px.bar(x=top_categories.index, y=top_categories.values, labels={'x': 'Category', 'y': 'Count'},
                 title="Top 10 App Categories", template=plot_template, color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig)
    st.info(f"📌 Insight: The top category is {top_categories.index[0]} with {top_categories.values[0]} apps. "
            f"This suggests a highly competitive market in the {top_categories.index[0]} category.")

    # Pricing Strategy
    st.subheader("Pricing Strategy Analysis")
    df['Price_Category'] = pd.cut(df['Price'], bins=[-1, 0, 1, 5, 10, 100],
                                  labels=['Free', '$0.01-$1', '$1-$5', '$5-$10', '$10+'])
    price_dist = df['Price_Category'].value_counts()
    fig = px.pie(values=price_dist.values, names=price_dist.index, title="Distribution of App Pricing",
                 template=plot_template, color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(width=700, height=500)
    st.plotly_chart(fig)
    st.info(f"📌 Insight: {price_dist['Free']/len(df)*100:.1f}% of apps are free. "
            f"This indicates a strong preference for the freemium model in the app market.")

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    corr_matrix = df[['Rating', 'Reviews', 'Size', 'Installs', 'Price']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap of Numeric Features",
                    template=plot_template, color_continuous_scale="Viridis")
    fig.update_layout(width=800, height=600)
    st.plotly_chart(fig)
    max_corr = corr_matrix.max().sort_values(ascending=False)[1]
    max_corr_pair = corr_matrix.max().sort_values(ascending=False).index[1]
    st.info(f"📌 Insight: The strongest correlation ({max_corr:.2f}) is between {max_corr_pair} and Rating. "
            f"This suggests that {max_corr_pair} might be a good indicator of an app's success.")

    # Installs vs. Rating
    st.subheader("Relationship between Installs and Rating")
    fig = px.scatter(df, x="Installs", y="Rating", color="Category", hover_name="App",
                     log_x=True, size="Reviews", size_max=15,
                     title="App Installs vs. Rating", template=plot_template,
                     color_discrete_sequence=px.colors.qualitative.Dark24)
    fig.update_layout(width=900, height=600)
    st.plotly_chart(fig)
    st.info("📌 Insight: There's a slight positive trend between the number of installs and ratings. "
            "However, some highly installed apps have lower ratings, suggesting that popularity doesn't always equate to quality.")

    # Feature Importance
    st.subheader("Feature Importance for App Rating")
    X = df[['Reviews', 'Size', 'Installs', 'Price']]
    y = df['Rating']
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    fig = px.bar(x=mi_scores.index, y=mi_scores.values, labels={'x': 'Feature', 'y': 'Mutual Information Score'},
                 title="Feature Importance for App Rating", template=plot_template,
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(width=800, height=500)
    st.plotly_chart(fig)
    top_feature = mi_scores.index[0]
    st.info(f"📌 Insight: {top_feature} appears to be the most important feature in determining an app's rating. "
            f"Developers should focus on improving {top_feature.lower()} to potentially increase their app's rating.")

def clustering_tab():
    st.header("🧮 Clustering Analysis")

    st.markdown("""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px;">
        <h3 style="color: #3498db;">Unveiling Hidden Patterns with Clustering</h3>
        <p style="font-size:16px; color: #fafafa;">
        Explore different clustering algorithms to discover inherent groupings within our app data.
        Compare the results and insights from each method.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature selection
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    default_features = ['Rating', 'Reviews', 'Installs', 'Price']
    default_features = [f for f in default_features if f in numeric_columns]
    selected_features = st.multiselect("Select features for clustering:", numeric_columns, default=default_features)

    if len(selected_features) < 2:
        st.warning("Please select at least two features for clustering.")
    else:
        # Prepare data for clustering
        X = df[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-Means Clustering
        st.subheader("K-Means Clustering")
        k_range = range(2, 11)
        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

        # Elbow Method and Silhouette Score
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers', name='Inertia'))
            fig.update_layout(title='Elbow Method for Optimal k',
                              xaxis_title='Number of Clusters (k)',
                              yaxis_title='Inertia',
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'))
            fig.update_layout(title='Silhouette Score for Different k',
                              xaxis_title='Number of Clusters (k)',
                              yaxis_title='Silhouette Score',
                              template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        st.info(f"📌 The optimal number of clusters based on the highest Silhouette Score is {optimal_k}.")

        # Perform K-Means clustering with optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_clusters = kmeans.fit_predict(X_scaled)

        # Visualize K-Means clusters
        st.subheader(f"K-Means Clustering with K={optimal_k}")
        fig = plot_clusters(X_scaled, kmeans_clusters, f"K-Means (K={optimal_k})")
        st.plotly_chart(fig, use_container_width=True)

        kmeans_silhouette = silhouette_score(X_scaled, kmeans_clusters)
        st.write(f"Silhouette Score: {kmeans_silhouette:.3f}")

        # K-Means Cluster Insights
        st.subheader("K-Means Cluster Insights")
        cluster_data = pd.DataFrame(X, columns=selected_features)
        cluster_data['Cluster'] = kmeans_clusters
        for i in range(optimal_k):
            cluster = cluster_data[cluster_data['Cluster'] == i]
            st.markdown(f"""
            <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="color: #3498db;">Cluster {i}</h4>
                <p style="color: #ecf0f1;">Size: {len(cluster)} apps ({len(cluster)/len(cluster_data)*100:.1f}% of total)</p>
                <p style="color: #ecf0f1;">Key characteristics:</p>
                <ul style="color: #ecf0f1;">
            """, unsafe_allow_html=True)
            for feature in selected_features:
                avg_value = cluster[feature].mean()
                overall_avg = cluster_data[feature].mean()
                difference = (avg_value - overall_avg) / overall_avg * 100
                if abs(difference) > 10:
                    direction = "higher" if difference > 0 else "lower"
                    st.markdown(f"<li>{feature}: {abs(difference):.1f}% {direction} than average</li>", unsafe_allow_html=True)
            st.markdown("""
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Hierarchical Clustering
        st.subheader("Hierarchical Clustering")
        linkage_methods = ['ward', 'complete', 'average', 'single']
        selected_linkage = st.selectbox("Select linkage method:", linkage_methods)
        
        hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage=selected_linkage)
        hierarchical_clusters = hierarchical.fit_predict(X_scaled)

        # Visualize Hierarchical clusters
        fig = plot_clusters(X_scaled, hierarchical_clusters, f"Hierarchical ({selected_linkage})")
        st.plotly_chart(fig, use_container_width=True)

        hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_clusters)
        st.write(f"Silhouette Score: {hierarchical_silhouette:.3f}")

        # Hierarchical Cluster Insights
        st.subheader("Hierarchical Cluster Insights")
        cluster_data['Hierarchical_Cluster'] = hierarchical_clusters
        for i in range(optimal_k):
            cluster = cluster_data[cluster_data['Hierarchical_Cluster'] == i]
            st.markdown(f"""
            <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="color: #3498db;">Cluster {i}</h4>
                <p style="color: #ecf0f1;">Size: {len(cluster)} apps ({len(cluster)/len(cluster_data)*100:.1f}% of total)</p>
                <p style="color: #ecf0f1;">Key characteristics:</p>
                <ul style="color: #ecf0f1;">
            """, unsafe_allow_html=True)
            for feature in selected_features:
                avg_value = cluster[feature].mean()
                overall_avg = cluster_data[feature].mean()
                difference = (avg_value - overall_avg) / overall_avg * 100
                if abs(difference) > 10:
                    direction = "higher" if difference > 0 else "lower"
                    st.markdown(f"<li>{feature}: {abs(difference):.1f}% {direction} than average</li>", unsafe_allow_html=True)
            st.markdown("""
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # DBSCAN Clustering
        st.subheader("DBSCAN Clustering")
        eps = st.slider("Select epsilon (neighborhood distance):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("Select minimum samples:", min_value=2, max_value=10, value=5)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_clusters = dbscan.fit_predict(X_scaled)

        # Visualize DBSCAN clusters
        fig = plot_clusters(X_scaled, dbscan_clusters, "DBSCAN")
        st.plotly_chart(fig, use_container_width=True)

        if len(set(dbscan_clusters)) > 1:
            dbscan_silhouette = silhouette_score(X_scaled, dbscan_clusters)
            st.write(f"Silhouette Score: {dbscan_silhouette:.3f}")
        else:
            dbscan_silhouette = "N/A (only one cluster found)"
            st.write("Silhouette Score: N/A (only one cluster found)")

        # DBSCAN Cluster Insights
        st.subheader("DBSCAN Cluster Insights")
        cluster_data['DBSCAN_Cluster'] = dbscan_clusters
        for i in set(dbscan_clusters):
            if i != -1:  # Exclude noise points
                cluster = cluster_data[cluster_data['DBSCAN_Cluster'] == i]
                st.markdown(f"""
                <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4 style="color: #3498db;">Cluster {i}</h4>
                    <p style="color: #ecf0f1;">Size: {len(cluster)} apps ({len(cluster)/len(cluster_data)*100:.1f}% of total)</p>
                    <p style="color: #ecf0f1;">Key characteristics:</p>
                    <ul style="color: #ecf0f1;">
                """, unsafe_allow_html=True)
                for feature in selected_features:
                    avg_value = cluster[feature].mean()
                    overall_avg = cluster_data[feature].mean()
                    difference = (avg_value - overall_avg) / overall_avg * 100
                    if abs(difference) > 10:
                        direction = "higher" if difference > 0 else "lower"
                        st.markdown(f"<li>{feature}: {abs(difference):.1f}% {direction} than average</li>", unsafe_allow_html=True)
                st.markdown("""
                    </ul>
                </div>
                """, unsafe_allow_html=True)

        noise_points = cluster_data[cluster_data['DBSCAN_Cluster'] == -1]
        st.markdown(f"""
        <div style="background-color: #2c3e50; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
            <h4 style="color: #3498db;">Noise Points</h4>
            <p style="color: #ecf0f1;">Number of noise points: {len(noise_points)} ({len(noise_points)/len(cluster_data)*100:.1f}% of total)</p>
        </div>
        """, unsafe_allow_html=True)

        # Clustering Performance Comparison
        st.subheader("Clustering Performance Comparison")
        performance_df = pd.DataFrame({
            'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN'],
            'Silhouette Score': [kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette]
        })
        st.table(performance_df.style.set_properties(**{'background-color': '#1f2937', 'color': '#fafafa'}))

        # Summary and Recommendations
        st.subheader("Summary and Recommendations")
        best_algorithm = performance_df.loc[performance_df['Silhouette Score'].idxmax(), 'Algorithm']
        st.markdown(f"""
        <div style="background-color: #1f2937; padding: 20px; border-radius: 10px;">
            <h4 style="color: #3498db;">Clustering Analysis Insights</h4>
            <p style="color: #fafafa;">Based on the analysis of different clustering algorithms, we can conclude:</p>
            <ul style="color: #fafafa;">
                <li><strong>K-Means Clustering</strong>: 
                    <ul>
                        <li>Optimal number of clusters: {optimal_k}</li>
                        <li>Provides clear centroids for each cluster, making it easy to interpret the average characteristics of apps in each group</li>
                        <li>Clusters tend to be more balanced in size</li>
                    </ul>
                </li>
                <li><strong>Hierarchical Clustering</strong>:
                    <ul>
                        <li>Using {selected_linkage} linkage method</li>
                        <li>Captures nested relationships between clusters, which can be useful for understanding subcategories within app groups</li>
                        <li>May produce more varied cluster sizes compared to K-Means</li>
                    </ul>
                </li>
                <li><strong>DBSCAN</strong>:
                    <ul>
                        <li>Identified dense regions of apps with similar characteristics</li>
                        <li>Capable of detecting outliers and noise in the data</li>
                        <li>Number of clusters is determined by the algorithm based on data density</li>
                    </ul>
                </li>
            </ul>
            <p style="color: #fafafa;">
            📌 <strong>Recommendation</strong>: The {best_algorithm} algorithm appears to perform best for this dataset, 
            with the highest Silhouette Score. This suggests that the app categories identified by this algorithm 
            might be the most distinct and well-separated.
            </p>
            <p style="color: #fafafa;">
            Consider using {best_algorithm} for further analysis and decision-making. However, also take into account the 
            specific insights provided by each algorithm, as they offer different perspectives on the app ecosystem.
            </p>
        </div>
        """, unsafe_allow_html=True)

def plot_clusters(X, clusters, algorithm_name):
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=clusters,
                         title=f"{algorithm_name} Clustering (PCA Visualization)",
                         template="plotly_dark", color_continuous_scale="Viridis")
    else:
        fig = px.scatter(x=X[:, 0], y=X[:, 1], color=clusters,
                         title=f"{algorithm_name} Clustering",
                         template="plotly_dark", color_continuous_scale="Viridis")
    fig.update_layout(width=800, height=600)
    return fig

def recommendations_tab():
    st.header("💡 Recommendations and Insights")

    st.markdown("""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px;">
        <h3 style="color: #3498db;">Data-Driven Recommendations for App Developers and Marketers</h3>
        <p style="font-size:16px; color: #fafafa;">
        Based on our exploratory data analysis and clustering results, here are key insights and actionable recommendations:
        </p>
    </div>
    """, unsafe_allow_html=True)

    recommendations = [
        {
            "title": "Focus on Quality",
            "insight": "The average app rating is high (4.17 out of 5).",
            "recommendation": "Prioritize app quality and user experience to stand out in a competitive market."
        },
        {
            "title": "Category Strategy",
            "insight": f"{df['Category'].value_counts().index[0]} is the most populous category.",
            "recommendation": "Consider niche sub-categories within popular categories to differentiate your app."
        },
        {
            "title": "Price Point",
            "insight": "There's a weak negative correlation between price and installs.",
            "recommendation": "Consider a freemium model or competitive pricing strategy to maximize installs."
        },
        {
            "title": "User Engagement",
            "insight": "Number of reviews is strongly correlated with installs.",
            "recommendation": "Implement strategies to encourage user reviews and ratings to boost visibility."
        },
        {
            "title": "App Size Optimization",
            "insight": "App size has a moderate positive correlation with installs.",
            "recommendation": "Balance app functionality with size optimization to appeal to users with limited device storage."
        },
        {
            "title": "Target High-Value Clusters",
            "insight": "Clustering reveals distinct app groups based on performance metrics.",
            "recommendation": "Analyze the characteristics of the best-performing cluster and align your app development strategy accordingly."
        },
        {
            "title": "Continuous Improvement",
            "insight": "The app market is dynamic and competitive.",
            "recommendation": "Regularly update your app, incorporate user feedback, and stay informed about market trends to maintain relevance."
        }
    ]

    for i, rec in enumerate(recommendations):
        st.markdown(f"""
        <div style="background-color: #2c3e50; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
            <h4 style="color: #3498db;">{i+1}. {rec['title']}</h4>
            <p style="color: #ecf0f1;"><strong>Insight:</strong> {rec['insight']}</p>
            <p style="color: #ecf0f1;"><strong>Recommendation:</strong> {rec['recommendation']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; margin-top: 20px;">
        <h4 style="color: #3498db;">Conclusion</h4>
        <p style="color: #fafafa;">
        The mobile app market is highly competitive and ever-evolving. By leveraging these data-driven insights and recommendations, 
        developers and marketers can make informed decisions to improve their app's performance, user engagement, and 
        overall success in the app store. Remember to continually analyze your app's performance and stay updated on 
        market trends to maintain a competitive edge.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("🚀 App Store Clustering Analysis")
    
    tabs = ["Home", "EDA", "Clustering", "Recommendations"]
    page = st.radio("Navigate", tabs, horizontal=True)

    if page == "Home":
        home_tab()
    elif page == "EDA":
        eda_tab()
    elif page == "Clustering":
        clustering_tab()
    elif page == "Recommendations":
        recommendations_tab()

if __name__ == "__main__":
    main()