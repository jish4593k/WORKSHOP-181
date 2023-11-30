import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import Tk, ttk, messagebox
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

# Load data using pandas
df = pd.read_csv("medium_data.csv")
data = df["reading_time"].tolist()

# Plot the population distribution using Plotly
fig_population = go.Figure()
fig_population.add_trace(go.Histogram(x=data, histnorm="probability", name="Population Distribution"))
fig_population.update_layout(title_text="Population Distribution", xaxis_title="reading_time", yaxis_title="Probability")
fig_population.show()

print("Population mean:", np.mean(data))

def random_set_of_mean(counter):
 
    dataset = np.random.choice(data, counter)
    mean = np.mean(dataset)
    return mean

def show_fig(mean_list):

    plt.figure(figsize=(8, 5))
    sns.histplot(mean_list, kde=True)
    plt.title("Distribution of Sample Means")
    plt.xlabel("reading_time")
    plt.ylabel("Frequency")
    plt.show()

def setup():
    mean_list = [random_set_of_mean(10) for _ in range(100)]


    show_fig(mean_list)
