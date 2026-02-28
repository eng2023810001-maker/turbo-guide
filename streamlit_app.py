import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # (موجود زي كودك، حتى لو ما نستخدمه كثير هنا)
import os
import numpy as np
import time
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title="AI Drilling Command Center",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# THEME / UI
# -------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&display=swap');
        html, body, [class*="css"] { font-family: 'Orbitron', sans-serif; }
        .stApp {
            background-color: #060d1f;
            background-image: linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
            background-size: 40px 40px;
            color: #dce9ff;
        }
        header, #MainMenu, footer {visibility: hidden;}
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .dc-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 18px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.35);
        }
        .dc-title { font-size: 26px; font-weight: 700; margin: 0; }
        .dc-subtitle { font-size: 12px; color: rgba(220,233,255,0.65); margin-top: 6px; }
        .stButton > button {
            background: linear-gradient(135deg, #1e3a8a, #0ea5e9);
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1rem;
            transition: 0.2s ease-in-out;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            filter: brightness(1.1);
        }
        .stTextInput input, .stChatInput input {
            border-radius: 10px !important;
            background: rgba(255,255,255,0.05) !important;
            color: #dce9ff !important;
        }
        .stChatMessage {
            background-color: rgba(255,255,255,0.02);
            border-left: 4px solid #1e3a8a;
            border-radius: 10px;
            padding: 10px 15px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def init_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "active_section" not in st.session_state:
        st.session_state.active_section = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_wells" not in st.session_state:
        st.session_state.selected_wells = []
    if "assistant" not in st.session_state:
        st.session_state.assistant = None

    # Added for Benchmark Plot button behavior
    if "bp_show_plot" not in st.session_state:
        st.session_state.bp_show_plot = False

# -------------------------
# DATA LOADING
# -------------------------
@st.cache_data(show_spinner=False)
def load_kpi(path="kpi.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

# -------------------------
# PLOT STYLING HELPERS
# -------------------------
THEME_BG = "#0b1220"
THEME_TEXT = "#dce9ff"
THEME_GRID = "white"
TEAL_MAIN = "#11cbd7"
TEAL_EDGE = "#22d3ee"
GREEN_BEST = "#22c55e"
RED_WORST = "#ef4444"

def style_axes_dark(fig, ax, title=None, xlabel=None, ylabel=None):
    fig.patch.set_facecolor(THEME_BG)
    ax.set_facecolor(THEME_BG)

    ax.tick_params(colors=THEME_TEXT)
    ax.xaxis.label.set_color(THEME_TEXT)
    ax.yaxis.label.set_color(THEME_TEXT)

    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.12))

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25, color=THEME_GRID)

    if title:
        ax.set_title(title, color=THEME_TEXT, fontsize=14, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

def gradient_bar(ax, x_labels, y_values, highlight=False):
    """
    Draw bars with gradient fill via imshow overlay.
    BEST/WORST tags removed (text labels).
    """
    x_pos = np.arange(len(x_labels))
    edge = TEAL_EDGE

    y_arr = np.array(y_values, dtype=float)
    y_arr = np.where(np.isfinite(y_arr), y_arr, np.nan)

    bars = ax.bar(x_pos, y_arr, color="none", edgecolor=edge, linewidth=1.6, zorder=2)

    cmap = LinearSegmentedColormap.from_list("teal_grad", ["#0ea5e9", TEAL_MAIN])

    finite_mask = np.isfinite(y_arr)
    if highlight and np.any(finite_mask):
        finite_vals = y_arr[finite_mask]
        finite_idx = np.where(finite_mask)[0]
        best_i = int(finite_idx[np.argmax(finite_vals)])
        worst_i = int(finite_idx[np.argmin(finite_vals)])
    else:
        best_i = worst_i = None

    for i, bar in enumerate(bars):
        x0 = bar.get_x()
        w = bar.get_width()
        h = bar.get_height()

        if np.isfinite(h) and h != 0:
            grad = np.linspace(0, 1, 128).reshape(128, 1)
            ax.imshow(
                grad,
                extent=[x0, x0 + w, 0, h],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                alpha=0.85,
                zorder=1
            )

        if highlight and best_i is not None and i == best_i:
            bar.set_edgecolor(GREEN_BEST)
            bar.set_linewidth(2.2)
        elif highlight and worst_i is not None and i == worst_i:
            bar.set_edgecolor(RED_WORST)
            bar.set_linewidth(2.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, color=THEME_TEXT)

def bar_with_highlight(ax, x_labels, y_values, highlight=True):
    """
    BEST/WORST tags removed (text labels).
    """
    x_pos = np.arange(len(x_labels))

    y_arr = np.array(y_values, dtype=float)
    y_arr = np.where(np.isfinite(y_arr), y_arr, np.nan)

    finite_mask = np.isfinite(y_arr)
    best_i = worst_i = None

    if highlight and np.any(finite_mask):
        finite_vals = y_arr[finite_mask]
        finite_idx = np.where(finite_mask)[0]
        best_i = int(finite_idx[np.argmax(finite_vals)])
        worst_i = int(finite_idx[np.argmin(finite_vals)])

        colors = []
        for i in range(len(y_arr)):
            if i == best_i:
                colors.append(GREEN_BEST)
            elif i == worst_i:
                colors.append(RED_WORST)
            else:
                colors.append(TEAL_MAIN)
    else:
        colors = [TEAL_MAIN] * len(y_arr)

    ax.bar(x_pos, y_arr, color=colors, edgecolor="white", linewidth=0.8, alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, color=THEME_TEXT)

def glow_line(ax, x_vals, y_vals, is_categorical=False):
    """
    Glow effect: draw multiple wider transparent lines behind main line.
    """
    if is_categorical:
        x_pos = np.arange(len(x_vals))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in x_vals], color=THEME_TEXT)
        x_plot = x_pos
    else:
        x_plot = x_vals

    for lw, alpha in [(12, 0.06), (8, 0.10), (5, 0.18)]:
        ax.plot(x_plot, y_vals, color=TEAL_EDGE, linewidth=lw, alpha=alpha, zorder=2)

    ax.plot(x_plot, y_vals, color=TEAL_MAIN, linewidth=3, marker="o", zorder=3)

def scatter_with_glow(ax, x_vals, y_vals, is_categorical=False):
    if is_categorical:
        x_pos = np.arange(len(x_vals))
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(v) for v in x_vals], color=THEME_TEXT)
        x_plot = x_pos
    else:
        x_plot = x_vals

    ax.scatter(x_plot, y_vals, s=260, color=TEAL_EDGE, alpha=0.12, edgecolors="none", zorder=2)
    ax.scatter(x_plot, y_vals, s=120, color=TEAL_MAIN, alpha=0.9, edgecolors=TEAL_EDGE, linewidths=1.2, zorder=3)

# -------------------------
# APP SECTIONS
# -------------------------
def render_login():
    _, col, _ = st.columns([1.2, 1, 1.2])
    with col:
        st.markdown("""
            <div class="dc-card">
              <p class="dc-title">AI Drilling Command Center</p>
              <div class="dc-subtitle">Secure access • Real-time assistance • Operational clarity</div>
            </div>
        """, unsafe_allow_html=True)
        st.write("")
        name = st.text_input("Enter your name", placeholder="e.g., Ahmed", label_visibility="collapsed")
        st.write("")
        col1, col2 = st.columns(2)
        if col1.button("Login") and name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.logged_in = True
            st.session_state.active_section = "menu"
            st.rerun()
        if col2.button("Clear"):
            st.session_state.user_name = ""
            st.rerun()

def render_shell():
    st.markdown(f"""
        <div class="dc-card" style="display:flex;justify-content:space-between;align-items:center;">
          <div>
            <p class="dc-title">AI Drilling Command Center</p>
            <div class="dc-subtitle">Operator: <b>{st.session_state.user_name}</b></div>
          </div>
          <div style="text-align:right;">
            <div class="dc-subtitle">Status: <b>Online</b></div>
          </div>
        </div>
    """, unsafe_allow_html=True)
    st.write("")

    a, b = st.columns(2)
    for label, section, col in zip(
        ["ChatBot", "Benchmarking"],
        ["chatbot", "benchmark"],
        [a, b]):

        with col:
            st.markdown('<div class="dc-card">', unsafe_allow_html=True)
            st.subheader(label)
            st.caption("Module interface")
            if st.button(f"Open {label}", use_container_width=True):
                st.session_state.active_section = section
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.session_state.active_section = None
        st.session_state.messages = []
        st.session_state.selected_wells = []
        st.session_state.assistant = None
        st.session_state.bp_show_plot = False
        st.rerun()

def render_chatbot():
    from chat_logic import DrillingAssistant

    st.markdown('<div class="dc-card">', unsafe_allow_html=True)
    st.subheader("ChatBot")
    st.caption("Ask about operations, safety steps, drilling parameters, lessons learned, stuck pipe, losses, etc.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    if st.session_state.assistant is None:
        st.session_state.assistant = DrillingAssistant()
    assistant = st.session_state.assistant

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_text = st.chat_input("Type your question…")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.spinner("Analyzing..."):
            response = assistant.chat_streamlit(user_text)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

    st.write("")
    c1, c2 = st.columns(2)
    if c1.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    if c2.button("Back to Menu"):
        st.session_state.active_section = "menu"
        st.rerun()

def render_benchmark_selection():
    st.markdown('<div class="dc-card">', unsafe_allow_html=True)
    st.subheader("Benchmarking")
    st.caption("Select wells and benchmarking view type")
    st.markdown('</div>', unsafe_allow_html=True)

    df = load_kpi()
    if df.empty or "Well" not in df.columns:
        st.error("kpi.csv not found / empty, or missing 'Well' column. Please add/repair the file.")
        if st.button("Back to Menu"):
            st.session_state.active_section = "menu"
            st.rerun()
        return

    wells = st.multiselect("Select wells", df["Well"].unique())
    if wells:
        col1, col2 = st.columns(2)
        if col1.button("View Table"):
            st.session_state.selected_wells = wells
            st.session_state.active_section = "benchmark_table"
            st.rerun()
        if col2.button("View Plot"):
            st.session_state.selected_wells = wells
            st.session_state.active_section = "benchmark_plot"
            st.session_state.bp_show_plot = False
            st.rerun()

    if st.button("Back to Menu"):
        st.session_state.active_section = "menu"
        st.rerun()

def render_benchmark_table():
    st.subheader("Benchmarking Table")
    df = load_kpi()
    if df.empty:
        st.error("kpi.csv not found / empty.")
        if st.button("Back"):
            st.session_state.active_section = "benchmark"
            st.rerun()
        return

    sub = df[df["Well"].isin(st.session_state.selected_wells)]
    st.dataframe(sub.reset_index(drop=True))

    if st.button("Back"):
        st.session_state.active_section = "benchmark"
        st.rerun()

def render_benchmark_plot():
    st.subheader("Benchmarking Plot")

    df = load_kpi()
    if df.empty:
        st.error("kpi.csv not found / empty.")
        if st.button("Back"):
            st.session_state.active_section = "benchmark"
            st.session_state.bp_show_plot = False
            st.rerun()
        return

    sub = df[df["Well"].isin(st.session_state.selected_wells)].copy()
    if sub.empty:
        st.warning("No data for selected wells.")
        if st.button("Back"):
            st.session_state.active_section = "benchmark"
            st.session_state.bp_show_plot = False
            st.rerun()
        return

    numeric_cols = sub.select_dtypes(include="number").columns.tolist()
    metrics = [c for c in numeric_cols if c != "ERROR"]

    if not metrics:
        st.warning("No numeric KPI columns available to plot.")
        if st.button("Back"):
            st.session_state.active_section = "benchmark"
            st.session_state.bp_show_plot = False
            st.rerun()
        return

    kind = st.selectbox("Select chart type", ["bar", "line", "scatter", "box"])
    x_options = ["Well"] + metrics
    x = st.selectbox("X-axis", x_options)
    y = st.selectbox("Y-axis", metrics)

    # Effects AUTO ON
    use_highlight = True
    use_gradient = (kind == "bar")
    use_glow = (kind in ["line", "scatter"])
    use_animate = (kind == "line")

    # Plot button
    colp1, _ = st.columns([1, 5])
    with colp1:
        if st.button("Plot"):
            st.session_state.bp_show_plot = True

    if not st.session_state.bp_show_plot:
        st.info("Select options then press **Plot** to display the chart.")
        if st.button("Back"):
            st.session_state.active_section = "benchmark"
            st.session_state.bp_show_plot = False
            st.rerun()
        return

    def draw_plot(ax):
        x_is_well = (x == "Well")

        plot_df = sub.copy()
        if not x_is_well:
            plot_df = plot_df[[x, y]].copy()
            plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
        else:
            plot_df = plot_df[["Well", y]].copy()

        plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y])
        if not x_is_well:
            plot_df = plot_df.dropna(subset=[x, y])

        if plot_df.empty:
            st.warning("Selected X/Y produced no valid numeric data to plot.")
            return

        if x_is_well:
            x_vals = plot_df["Well"].astype(str).tolist()
        else:
            x_vals = plot_df[x].tolist()
        y_vals = plot_df[y].tolist()

        if kind == "bar":
            if x_is_well:
                x_labels = x_vals
                y_values = np.array(y_vals, dtype=float)
                if use_gradient:
                    gradient_bar(ax, x_labels, y_values, highlight=use_highlight)
                else:
                    bar_with_highlight(ax, x_labels, y_values, highlight=use_highlight)
            else:
                ax.bar(x_vals, y_vals, color=TEAL_MAIN, edgecolor=TEAL_EDGE, alpha=0.85)
                if use_highlight and len(y_vals):
                    best_i = int(np.argmax(y_vals))
                    worst_i = int(np.argmin(y_vals))
                    ax.scatter([x_vals[best_i]], [y_vals[best_i]], s=140, color=GREEN_BEST, zorder=4)
                    ax.scatter([x_vals[worst_i]], [y_vals[worst_i]], s=140, color=RED_WORST, zorder=4)

        elif kind == "line":
            if use_glow:
                glow_line(ax, x_vals, y_vals, is_categorical=x_is_well)
            else:
                if x_is_well:
                    x_pos = np.arange(len(x_vals))
                    ax.plot(x_pos, y_vals, color=TEAL_MAIN, linewidth=2.5, marker="o")
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_vals, color=THEME_TEXT)
                else:
                    ax.plot(x_vals, y_vals, color=TEAL_MAIN, linewidth=2.5, marker="o")

            if use_highlight and len(y_vals):
                best_i = int(np.argmax(y_vals))
                worst_i = int(np.argmin(y_vals))
                if x_is_well:
                    bx = best_i
                    wx = worst_i
                else:
                    bx = x_vals[best_i]
                    wx = x_vals[worst_i]
                ax.scatter([bx], [y_vals[best_i]], s=160, color=GREEN_BEST, zorder=5)
                ax.scatter([wx], [y_vals[worst_i]], s=160, color=RED_WORST, zorder=5)

        elif kind == "scatter":
            if use_glow:
                scatter_with_glow(ax, x_vals, y_vals, is_categorical=x_is_well)
            else:
                if x_is_well:
                    x_pos = np.arange(len(x_vals))
                    ax.scatter(x_pos, y_vals, s=110, color=TEAL_MAIN, edgecolors=TEAL_EDGE, linewidths=1.2)
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(x_vals, color=THEME_TEXT)
                else:
                    ax.scatter(x_vals, y_vals, s=110, color=TEAL_MAIN, edgecolors=TEAL_EDGE, linewidths=1.2)

            if use_highlight and len(y_vals):
                best_i = int(np.argmax(y_vals))
                worst_i = int(np.argmin(y_vals))
                if x_is_well:
                    bx = best_i
                    wx = worst_i
                else:
                    bx = x_vals[best_i]
                    wx = x_vals[worst_i]
                ax.scatter([bx], [y_vals[best_i]], s=220, color=GREEN_BEST, zorder=5)
                ax.scatter([wx], [y_vals[worst_i]], s=220, color=RED_WORST, zorder=5)

        elif kind == "box":
            if not x_is_well:
                st.info("Box plot works best when X-axis is 'Well'. Switching X-axis to 'Well'.")
                plot_df = sub[["Well", y]].copy()
                plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
                plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y])

            wells = plot_df["Well"].astype(str).unique().tolist()
            data = [plot_df[plot_df["Well"].astype(str) == w][y].dropna().values for w in wells]

            bp = ax.boxplot(data, patch_artist=True, labels=wells)
            for box in bp["boxes"]:
                box.set(facecolor=TEAL_MAIN, alpha=0.55, edgecolor=TEAL_EDGE, linewidth=1.4)
            for whisker in bp["whiskers"]:
                whisker.set(color=TEAL_EDGE, linewidth=1.2)
            for cap in bp["caps"]:
                cap.set(color=TEAL_EDGE, linewidth=1.2)
            for med in bp["medians"]:
                med.set(color="white", linewidth=1.4)

    if use_animate and kind == "line":
        placeholder = st.empty()
        fig, ax = plt.subplots(figsize=(10, 4))
        style_axes_dark(fig, ax, title=f"{y} vs {x}", xlabel=x, ylabel=y)

        x_is_well = (x == "Well")

        plot_df = sub.copy()
        if not x_is_well:
            plot_df = plot_df[[x, y]].copy()
            plot_df[x] = pd.to_numeric(plot_df[x], errors="coerce")
        else:
            plot_df = plot_df[["Well", y]].copy()

        plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[y])
        if not x_is_well:
            plot_df = plot_df.dropna(subset=[x, y])

        if plot_df.empty:
            st.warning("Selected X/Y produced no valid numeric data to animate.")
            plt.close(fig)
        else:
            if x_is_well:
                x_vals = plot_df["Well"].astype(str).tolist()
            else:
                x_vals = plot_df[x].tolist()
            y_vals = plot_df[y].tolist()

            for i in range(1, len(y_vals) + 1):
                ax.clear()
                style_axes_dark(fig, ax, title=f"{y} vs {x} (Live)", xlabel=x, ylabel=y)

                if use_glow:
                    glow_line(ax, x_vals[:i], y_vals[:i], is_categorical=x_is_well)
                else:
                    if x_is_well:
                        xp = np.arange(i)
                        ax.plot(xp, y_vals[:i], color=TEAL_MAIN, linewidth=2.5, marker="o")
                        ax.set_xticks(np.arange(len(x_vals)))
                        ax.set_xticklabels(x_vals, color=THEME_TEXT)
                    else:
                        ax.plot(x_vals[:i], y_vals[:i], color=TEAL_MAIN, linewidth=2.5, marker="o")

                placeholder.pyplot(fig, clear_figure=False, bbox_inches=None)
                time.sleep(0.12)

            plt.close(fig)

    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        style_axes_dark(fig, ax, title=f"{y} vs {x}", xlabel=x, ylabel=y)

        draw_plot(ax)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True, bbox_inches=None)

    if st.button("Back"):
        st.session_state.active_section = "benchmark"
        st.session_state.bp_show_plot = False
        st.rerun()

# -------------------------
# MAIN
# -------------------------
def main():
    inject_css()
    init_state()

    if not st.session_state.logged_in:
        render_login()
    elif st.session_state.active_section == "menu":
        render_shell()
    elif st.session_state.active_section == "chatbot":
        render_chatbot()
    elif st.session_state.active_section == "benchmark":
        render_benchmark_selection()
    elif st.session_state.active_section == "benchmark_table":
        render_benchmark_table()
    elif st.session_state.active_section == "benchmark_plot":
        render_benchmark_plot()
    else:
        st.session_state.active_section = "menu"
        st.rerun()

if __name__ == "__main__":
    main()
