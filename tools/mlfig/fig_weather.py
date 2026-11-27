"""天候変換画像の例 (ml_weather_example.png)。元画像 1 枚と、同一画像を画像生成モデルで 4 種の天候へ変換した画像。"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

B = os.path.expanduser('~/aiformula_dataset/real/アノテーション済み_1021/')
OUT = os.path.expanduser('~/master_thesis/images/ml_weather_example.png')
STEM = 'run_20261027_152143_000244'
plt.rcParams.update({'font.size': 10, 'text.color': '#0b0b0b'})
W = [('morning_sunny', 'Morning, sunny'), ('noon_rain', 'Noon, rainy'), ('evening_cloudy', 'Evening, cloudy'), ('night_fog', 'Night, foggy')]
items = [('(a) Original', B + f'base_261027_annotation/images/val/{STEM}.png')]
for k, (key, name) in zip('bcde', W):
    items.append((f'({k}) {name}', B + f'base_261027_annotation_cosmos_gen/images/val/{STEM}__aug_{key}.png'))
fig, axs = plt.subplots(2, 3, figsize=(11.0, 4.5))
for ax in axs.flat:
    ax.axis('off')
for ax, (t, p) in zip(axs.flat, items):
    ax.imshow(mpimg.imread(p)[12:372])
    ax.set_title(t, loc='left', fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, wspace=0.03, hspace=0.12)
fig.savefig(OUT, dpi=200, facecolor='white')
