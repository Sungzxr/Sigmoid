# sigmoid_app.py
# Streamlit app 2 of 3: Sigmoid (logistic) activation function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sigmoid Activation Function", layout="wide")
st.title("Sigmoid Activation Function")
st.write(
    "Sigmoid squashes any real number into the range (0, 1), so its output can be "
    "read as a probability. It is common in the output layer of binary classifiers."
)
st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


# --- Sidebar controls ---
with st.sidebar:
    st.header("Neuron settings")
    x = st.slider("Input x", -5.0, 5.0, 1.5, 0.1)
    w = st.slider("Weight w", -3.0, 3.0, 1.0, 0.1)
    b = st.slider("Bias b", -3.0, 3.0, 0.0, 0.1)
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05)

    st.header("Plot settings")
    z_min, z_max = st.slider("z range", -10.0, 10.0, (-6.0, 6.0), 0.5)
    show_grad = st.checkbox("Show derivative", value=True)

# --- One neuron: z = w*x + b, output = sigmoid(z) ---
z = w * x + b
out = float(sigmoid(z))
grad = float(sigmoid_grad(z))
label = int(out >= threshold)

st.subheader("1. One neuron calculation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("z = w·x + b", f"{z:.3f}")
c2.metric("σ(z)", f"{out:.4f}")
c3.metric("Derivative at z", f"{grad:.4f}")
c4.metric("Class at threshold", label)
st.write(f"z = ({w:.1f} × {x:.1f}) + {b:.1f} = **{z:.3f}**  →  σ(z) = **{out:.4f}**")

# --- Plot ---
st.subheader("2. Activation curve")
zs = np.linspace(z_min, z_max, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(zs, sigmoid(zs), label="σ(z)", linewidth=2)
if show_grad:
    ax.plot(zs, sigmoid_grad(zs), "--", label="Derivative", linewidth=1.5)
ax.axhline(threshold, color="orange", linestyle=":", label=f"Threshold = {threshold}")
ax.scatter([z], [out], color="red", zorder=5, label="Current neuron")
ax.axvline(0, color="grey", linewidth=0.8)
ax.set_xlabel("z (weighted sum)")
ax.set_ylabel("Output")
ax.set_title("Sigmoid activation function")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Value table ---
st.subheader("3. Sample values")
sample = np.array([-6, -4, -2, 0, 2, 4, 6], dtype=float)
st.dataframe(
    pd.DataFrame(
        {"z": sample, "σ(z)": sigmoid(sample).round(4), "Derivative": sigmoid_grad(sample).round(4)}
    ),
    hide_index=True,
)

# --- Notes ---
st.subheader("4. Key properties")
st.markdown(
    """
- **Output range:** 0 to 1
- **Advantages:** smooth and differentiable, output reads as a probability, ideal for binary output layers.
- **Disadvantages:** saturates for large |z| so the gradient becomes almost zero (vanishing gradient); output is not zero-centred; exponential is costlier to compute.
- **Maximum gradient:** 0.25, reached at z = 0.
- **Typical use:** output layer for binary classification; also gates in LSTM cells.
"""
)
