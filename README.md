QD実験用のリポジトリ

生値データの描画
python plot_qd_data.py --show



QD重心位置の描画
python plot_all_centroid_ave.py --focus 1200 --show
指定付き
python plot_all_centroid_ave.py --focus 1200 1200 1400 1600 1800 2000 2200 --show


QDシミュレータとの比較
# 例: 焦点12.00mm（1200）で def=+0.00mm を比較
python compare_defocus.py --defocus-mm 0 --focus 1200 --show
python compare_defocus.py --defocus-mm 0 2 4 --focus 1200 --show
で実行可能
--series measured reference flatspot
で、3種類のうちどれを描画するか選べる。

K導出
python compare_defocus.py --defocus-mm 0 2 4 6 8 10 \
  --series measured reference \
  --report-metrics --metrics-window-mm 0.02 \
  --focal-mm 75 --beam-diameter-mm 2.27 \
  --metrics-csv plots/compare/metrics.csv