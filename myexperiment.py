import streamlit as st
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt


url_gps = "https://raw.githubusercontent.com/jasa89/fysiikanlopputy-/refs/heads/main/My%20Experiment1/Location.csv"
url_acc = "https://raw.githubusercontent.com/jasa89/fysiikanlopputy-/refs/heads/main/My%20Experiment1/Linear%20Acceleration.csv"

df_gps =pd.read_csv(url_gps)
df_acc = pd.read_csv(url_acc)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_steps(df_acc):
    data = df_acc['Linear Acceleration z (m/s^2)']
    t = df_acc['Time (s)']
    T_tot = t.iloc[-1] - t.iloc[0]
    fs = len(t) / T_tot
    data_filt = butter_bandpass_filter(data, 0.5, 3.0, fs)
    steps = 0
    for i in range(len(data_filt) - 1):
        if data_filt[i] * data_filt[i + 1] < 0:
            steps += 0.5
    return int(round(steps)), t, data, data_filt 

def calculate_steps_ftt(df_acc):
    signal = df_acc['Linear Acceleration z (m/s^2)'] 
    t = df_acc['Time (s)'] 
    N = len(signal) 
    dt = np.max(t)/N 
    fourier = np.fft.fft(signal,N) 
    psd = (np.abs(fourier)**2)/N 
    freq = np.fft.fftfreq(N,dt) 
    L = np.arange(1,int(N/2)) 
    f_max = freq[L][psd[L] == np.max(psd[L])][0] 
    steps =  f_max*np.max(t) 
    return round((steps)), f_max, fourier, psd ,freq


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r

df_gps['Distance_km'] = 0.0

for i in range(len(df_gps) - 1):
    df_gps.loc[i+1, 'Distance_km'] = haversine(
        df_gps.loc[i, 'Longitude (°)'], df_gps.loc[i, 'Latitude (°)'],
        df_gps.loc[i+1, 'Longitude (°)'], df_gps.loc[i+1, 'Latitude (°)']
    )
df_gps['Total_distance_km'] = df_gps['Distance_km'].cumsum()
total_distance_m = df_gps['Total_distance_km'].iloc[-1] * 1000

steps_dt, t, data, data_filt = calculate_steps(df_acc)
steps_ftt, dominant_freq, fourier_vals, psd_vals, freq_vals = calculate_steps_ftt(df_acc)
step_length = total_distance_m / steps_dt if steps_dt > 0 else np.nan


st.header("Päivän kävelymatka")
st.write("Askelmäärä suodatetusta datasta:", steps_dt)
st.write("Askelmäärä Fourier-analyysin avulla", steps_ftt)
st.write("Keskinopeus on :", round(df_gps['Velocity (m/s)'].mean(),2),'m/s' ) 
st.write("Kokonaismatka:", round(total_distance_m / 1000, 2), "km") 
st.write("Askelpituus: ", round(step_length*100,2), "cm") 
st.subheader("Suodatettu kiihtyvyysdata (z-komponentti)")
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(t, data_filt)
ax.set_xlabel("Aika (s)")
ax.set_ylabel("Kiihtyvyys (m/s²)")
ax.grid()
ax.legend()
st.pyplot(fig)

st.subheader("Tehospektri")
mask = freq_vals > 0
freq_pos = freq_vals[mask]
psd_pos = psd_vals[mask]
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(freq_pos, psd_pos)
ax.set_xlabel("Taajuus [Hz]")
ax.set_ylabel("Teho")
ax.set_xlim(0, 10)
ax.grid()
st.pyplot(fig)

st.subheader("Reitin kartta")
df_gps = df_gps[df_gps['Horizontal Accuracy (m)'] <= 5].reset_index(drop=True)
lat1 = df_gps['Latitude (°)'].mean()
long1 = df_gps['Longitude (°)'].mean()
my_map = folium.Map(location=[lat1, long1], zoom_start=15)
route = list(zip(df_gps['Latitude (°)'], df_gps['Longitude (°)']))
folium.PolyLine(route, color='red', weight=3).add_to(my_map)
folium.Marker(route[0], popup="Alku", icon=folium.Icon(color='green')).add_to(my_map)
folium.Marker(route[-1], popup="Loppu", icon=folium.Icon(color='red')).add_to(my_map)
st_data = st_folium(my_map, width=700, height=500)

