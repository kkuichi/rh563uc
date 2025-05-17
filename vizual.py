#Vizualizacia pre ročné zhluky + variancia
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import calendar
import matplotlib.pyplot as plt
from scipy.stats import entropy

#GRAF "Priemerná mesačná spotreba a variácia podľa ročného zhluku"
#načitanie dát
df_rok = pd.read_csv("yearly_clustered_k4.csv")
df_den = pd.read_csv("daily_15min_vectors.csv")

df_den["day"] = pd.to_datetime(df_den["day"])
df_den["month"] = df_den["day"].dt.month
df_den["year"] = df_den["day"].dt.year

#výpočet dennej spotreby a spojenie s ročnými zhlukmi
cols_15min = [str(i) for i in range(96)]
df_den["daily_sum"] = df_den[cols_15min].sum(axis=1)
df = df_den.merge(df_rok[["dataid", "year", "cluster"]], on=["dataid", "year"], how="inner")

colours = [
    "#64B5F6",  #zhluk 0
    "#BA68C8",  #1
    "#FFB74D",  #2
    "#E57373",  #3
]
zhluky_poc = df["cluster"].nunique()


fig, axes = plt.subplots(2, 2, figsize=(14, 6))
axes = axes.flatten()
fig.suptitle("Priemerná mesačná spotreba a variácia podľa ročného zhluku", fontsize=14)

for cluster_id in range(zhluky_poc):
    ax = axes[cluster_id]
    df_cluster = df[df["cluster"] == cluster_id]

    #suma denneho spotreby po mesiacoch
    monthly_sum = df_cluster.groupby(["dataid", "month"])["daily_sum"].sum().reset_index()

    #riadky = domy, stlpce = mesiace, hodnoty = mesačný sučet
    pivot = monthly_sum.pivot(index="dataid", columns="month", values="daily_sum")

    #priemer a štandardna odchýlka medzi domácnosťami
    month_mean = pivot.mean()
    month_odchyl = pivot.std()*1

    months = list(range(1, 13))
    labels = [calendar.month_abbr[m] for m in months]
    color = colours[cluster_id]

    ax.plot(months, month_mean, label="Priemer", color=color)
    ax.fill_between(months, month_mean - month_odchyl, month_mean + month_odchyl,
                    color=color, alpha=0.25, label="±1 std")

    ax.set_title(f"Ročný Zhkuk {cluster_id}")
    ax.set_xticks(months)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Mesiac")
    ax.set_ylabel("Spotreba (kWh)")
    ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()



#GRAF "Rozloženie týždňových zhlukov v rámci ročných zhlukov"
#načitanie dát
df = pd.read_csv("yearly_clustered_k4.csv")
tyz_cols = [f"w{i}" for i in range(52)]

#vypočet frekvencii
zhluk_freq = {}
for zhlk_id in sorted(df["cluster"].unique()):
    subset = df[df["cluster"] == zhlk_id]
    all_weeks = subset[tyz_cols].values.flatten()
    all_weeks = all_weeks[all_weeks != -1]
    value_counts = pd.Series(all_weeks).value_counts(normalize=True).sort_index()
    zhluk_freq[zhlk_id] = value_counts

#tabulka frekvencii
df_freq = pd.DataFrame(zhluk_freq).fillna(0)
df_freq.index.name = "weekly_cluster"
df_freq.columns.name = "yearly_cluster"

print("Frekvencie týždňových zhlukov v jednotlivých ročnych zhlukoch:")
print(df_freq.round(3))

colours = [
    "#B3E5FC",  # 0 min spotreba
    "#29B6F6",  # 1
    "#FFEB3B",  # 2
    "#FB8C00",  # 3
    "#D32F2F",  # 4 max spotreba
]

df_freq.T.plot(kind="bar", stacked=True, figsize=(8, 5), color=colours)
plt.title("Rozloženie týždňových zhlukov v rámci ročných zhlukov")
plt.xlabel("Ročný zhluk")
plt.ylabel("Podiel")
plt.xticks(rotation=0)
plt.legend(title="Týždňový zhluk", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()



#GRAF "Entropia týždenných zhlukov v rámci ročných zhlukov"
#načitanie dát
df = pd.read_csv("yearly_clustered_k4.csv")

tyz_cols = [f"w{i}" for i in range(52)]

#výpočet entropie
def entropia_vyp(row):
    hodnota = row[tyz_cols].values.astype(int)
    hodnota = hodnota[hodnota != -1]  #filtrujeme -1
    pocet = np.bincount(hodnota, minlength=5)
    pravdep = pocet / pocet.sum()
    return entropy(pravdep, base=2)

#entropia pre každý row
df["entropy"] = df.apply(entropia_vyp, axis=1)

colours_entrop = {
    "0": "#4FC3F7",   #min spotreba
    "1": "#BA68C8",
    "2": "#FFB74D",
    "3": "#E57373",   #max spotreba
}

df["cluster"] = df["cluster"].astype(str)

sns.boxplot(x="cluster", y="entropy", data=df, hue="cluster",
    palette=colours_entrop, order=["0", "1", "2", "3"], width=0.6,
    fliersize=2, legend=False)

#vodorovné čiary
plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
plt.axhline(2.0, color="gray", linestyle="--", linewidth=1)

plt.title("Entropia týždenných zhlukov v rámci ročných zhlukov")
plt.xlabel("Ročný zhluk")
plt.ylabel("Entropia (bit)")
plt.tight_layout()
plt.show()



#GRAF "frekvencia denných zhlukov v rámci ročných zhlukov"
#načitanie dát
df_rocny = pd.read_csv("yearly_clustered_k4.csv")
df_den = pd.read_csv("daily_clusters_6.csv")

df_den["day"] = pd.to_datetime(df_den["day"])
df_den["year"] = df_den["day"].dt.year

#join podla dataid a year
df_rok_short = df_rocny[["dataid", "year", "cluster"]]  #cluster = ročny
df = df_den.merge(df_rok_short, on=["dataid", "year"], how="inner")  #cluster_x = denny, cluster_y = ročny

#nastavenie počtu dennych zhlukov
poc_den_zhlukov = 6
all_zhluky = sorted(df["cluster_y"].unique())

colours_day = LinearSegmentedColormap.from_list("custom_gradient", [
    "#cceeff", #min spotreba
    "#99ccff",
    "#ffcc99",
    "#ff9966",
    "#ff6666",
    "#cc0000" #max spotreba
])
colour_sada = [colours_day(i / (poc_den_zhlukov - 1)) for i in range(poc_den_zhlukov)]
#filtrovanie neprazdnych zhlukov
plot_data = []
for zhluk_id in all_zhluky:
    subset = df[df["cluster_y"] == zhluk_id]
    freq = subset["cluster_x"].value_counts(normalize=True).sort_index() * 100  # percentá
    freq = freq.reindex(range(poc_den_zhlukov), fill_value=0)
    if freq.sum() > 0:
        plot_data.append((zhluk_id, freq))

#vizualiácia
p = len(plot_data)
rows = (p + 1) // 2
fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))
axes = axes.flatten()
fig.suptitle("Frekvencia denných zhlukov v rámci ročných zhlukov", fontsize=14)

for i, (zhluk_id, freq) in enumerate(plot_data):
    ax = axes[i]
    freq = freq.rename_axis("denný_zhluk").reset_index(name="percento")
    sns.barplot(
        data=freq, x="denný_zhluk", y="percento",
        ax=ax, hue="denný_zhluk", dodge=False,
        palette=colour_sada, legend=False
    )
    ax.set_title(f"Ročný zhluk {zhluk_id}")
    ax.set_xlabel("Typ dňa")
    ax.set_ylabel("Podiel (%)")
    ax.set_ylim(0, 100)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



