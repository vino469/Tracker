import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.title("📊 Social Network Ads – KNN Classifier")

# 1. Load dataset
uploaded_file = st.file_uploader(
    "Upload Social_Network_Ads.csv", type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # 2. Feature matrix & target variable
    features = df[["Age", "EstimatedSalary"]]
    target = df["Purchased"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=0
    )

    # 4. Feature scaling
    std_scaler = StandardScaler()
    X_train_scaled = std_scaler.fit_transform(X_train)
    X_test_scaled = std_scaler.transform(X_test)

    # 5. KNN model
    k_neighbors = st.slider(
        "Select number of neighbors (k)", 1, 30, 10
    )

    knn_model = KNeighborsClassifier(
        n_neighbors=k_neighbors
    )
    knn_model.fit(X_train_scaled, y_train)

    # 6. Model accuracy
    accuracy = knn_model.score(
        X_test_scaled, y_test
    )
    st.success(f"Test Accuracy: {accuracy:.3f}")

    # 7. Prediction section
    st.subheader("🔍 Predict for a New User")

    user_age = st.number_input(
        "Age", min_value=18, max_value=70, value=45
    )
    user_salary = st.number_input(
        "Estimated Salary",
        min_value=15000,
        max_value=200000,
        value=42000
    )

    if st.button("Predict"):
        new_user = std_scaler.transform(
            [[user_age, user_salary]]
        )
        prediction = knn_model.predict(new_user)[0]

        result = (
            "Purchased (1)"
            if prediction == 1
            else "Not Purchased (0)"
        )
        st.success(f"Prediction Result: {result}")

    # 8. Decision boundary visualization
    st.subheader("📈 KNN Decision Boundary")

    show_boundary = st.checkbox(
        "Show Decision Boundary", value=True
    )

    if show_boundary:
        age_min, age_max = (
            features["Age"].min() - 1,
            features["Age"].max() + 1
        )
        sal_min, sal_max = (
            features["EstimatedSalary"].min() - 5000,
            features["EstimatedSalary"].max() + 5000
        )

        age_step = 0.25
        salary_step = 2000

        age_grid, salary_grid = np.meshgrid(
            np.arange(age_min, age_max, age_step),
            np.arange(sal_min, sal_max, salary_step)
        )

        grid_points = np.c_[
            age_grid.ravel(),
            salary_grid.ravel()
        ]

        grid_points_scaled = std_scaler.transform(
            grid_points
        )

        grid_predictions = knn_model.predict(
            grid_points_scaled
        ).reshape(age_grid.shape)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.contourf(
            age_grid,
            salary_grid,
            grid_predictions,
            alpha=0.3,
            cmap="RdYlGn"
        )

        ax.scatter(
            features["Age"][target == 0],
            features["EstimatedSalary"][target == 0],
            c="red",
            s=15,
            label="Not Purchased"
        )

        ax.scatter(
            features["Age"][target == 1],
            features["EstimatedSalary"][target == 1],
            c="green",
            s=15,
            label="Purchased"
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Estimated Salary")
        ax.set_title("KNN Classification – Social Network Ads")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

else:
    st.info("⬆️ Please upload the CSV file to start.")
