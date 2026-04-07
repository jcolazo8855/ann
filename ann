import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss

st.set_page_config(page_title="BAT 3305 - Colazo - ANN Demo", layout="wide")


def generate_data(dataset_name: str, n_samples: int, noise: float, random_state: int):
    if dataset_name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_name == "Circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.45, random_state=random_state)
    else:
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            class_sep=1.2,
            flip_y=noise,
            random_state=random_state,
        )
    return X, y


def build_and_train_model(
    X_train,
    y_train,
    hidden_1,
    hidden_2,
    hidden_3,
    activation,
    alpha,
    learning_rate_init,
    max_iter,
    random_state,
):
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_1, hidden_2, hidden_3),
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
    )
    model.fit(X_train, y_train)
    return model


def plot_decision_boundary(model, X, y, scaler):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid)
    probs = model.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(xx, yy, probs, levels=20, alpha=0.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=30)
    ax.set_title("Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(contour, ax=ax, label="Predicted probability")
    return fig


def plot_loss_curve(model):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(model.loss_curve_)
    ax.set_title("Training Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    return fig


st.title("BAT 3305 - Colazo - ANN Demo")
st.caption("Interactive demonstration of a basic 3-hidden-layer artificial neural network for binary classification.")

with st.sidebar.form("ann_controls"):
    st.subheader("Data Settings")
    dataset_name = st.selectbox("Dataset", ["Moons", "Circles", "Linearly Separable"])
    n_samples = st.slider("Number of observations", 100, 1500, 400, 50)
    noise = st.slider("Noise", 0.0, 0.40, 0.15, 0.01)
    test_size = st.slider("Test proportion", 0.10, 0.50, 0.25, 0.05)

    st.subheader("Network Architecture")
    hidden_1 = st.slider("Hidden layer 1 neurons", 2, 100, 16, 1)
    hidden_2 = st.slider("Hidden layer 2 neurons", 2, 100, 12, 1)
    hidden_3 = st.slider("Hidden layer 3 neurons", 2, 100, 8, 1)
    activation = st.selectbox("Activation function", ["relu", "tanh", "logistic"])

    st.subheader("Training Parameters")
    alpha = st.slider("Regularization (alpha)", 0.0001, 0.1000, 0.0010, 0.0001, format="%.4f")
    learning_rate_init = st.slider("Learning rate", 0.0005, 0.0500, 0.0050, 0.0005, format="%.4f")
    max_iter = st.slider("Maximum iterations", 100, 2000, 500, 50)
    random_state = st.number_input("Random seed", 1, 9999, 3305)

    run_model = st.form_submit_button("Run ANN Demo")

if "ann_params" not in st.session_state:
    st.session_state.ann_params = {
        "dataset_name": "Moons",
        "n_samples": 400,
        "noise": 0.15,
        "test_size": 0.25,
        "hidden_1": 16,
        "hidden_2": 12,
        "hidden_3": 8,
        "activation": "relu",
        "alpha": 0.0010,
        "learning_rate_init": 0.0050,
        "max_iter": 500,
        "random_state": 3305,
    }

if run_model:
    st.session_state.ann_params = {
        "dataset_name": dataset_name,
        "n_samples": n_samples,
        "noise": noise,
        "test_size": test_size,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "hidden_3": hidden_3,
        "activation": activation,
        "alpha": alpha,
        "learning_rate_init": learning_rate_init,
        "max_iter": max_iter,
        "random_state": int(random_state),
    }

params = st.session_state.ann_params

X, y = generate_data(
    params["dataset_name"], params["n_samples"], params["noise"], params["random_state"]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"], stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = build_and_train_model(
    X_train_scaled,
    y_train,
    params["hidden_1"],
    params["hidden_2"],
    params["hidden_3"],
    params["activation"],
    params["alpha"],
    params["learning_rate_init"],
    params["max_iter"],
    params["random_state"],
)

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)
train_prob = model.predict_proba(X_train_scaled)
test_prob = model.predict_proba(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
train_loss = log_loss(y_train, train_prob)
test_loss = log_loss(y_test, test_prob)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Train Accuracy", f"{train_acc:.3f}")
m2.metric("Test Accuracy", f"{test_acc:.3f}")
m3.metric("Train Log Loss", f"{train_loss:.3f}")
m4.metric("Test Log Loss", f"{test_loss:.3f}")

st.subheader("Network Summary")
summary_df = pd.DataFrame(
    {
        "Parameter": [
            "Dataset",
            "Architecture",
            "Activation",
            "Regularization alpha",
            "Learning rate",
            "Iterations used",
        ],
        "Value": [
            params["dataset_name"],
            f"({params['hidden_1']}, {params['hidden_2']}, {params['hidden_3']})",
            params["activation"],
            params["alpha"],
            params["learning_rate_init"],
            model.n_iter_,
        ],
    }
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.pyplot(plot_decision_boundary(model, X, y, scaler), clear_figure=True)
with right:
    st.pyplot(plot_loss_curve(model), clear_figure=True)

st.subheader("Data Preview")
preview_df = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
preview_df["Class"] = y
st.dataframe(preview_df.head(15), use_container_width=True)

st.subheader("Teaching Prompts")
st.markdown(
    """
1. How does changing the **activation function** alter the decision boundary?
2. What happens when you increase the number of neurons in each hidden layer?
3. How does stronger **regularization (alpha)** affect fit and generalization?
4. When do you begin to see signs of overfitting?
5. Which datasets are easiest or hardest for this network to classify?
"""
)
