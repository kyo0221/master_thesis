# 2章の図の生成スクリプト

- `topomap_rviz/`: 運営提供トポロジカルマップ(formula_ws/src/map/topomap_server/config/aiformula_topomap.osm)を
  MarkerArray で publish し，rviz2 (`topomap.rviz`) で表示して撮影 → `images/topomap_rviz.png`．
  `ROS_DOMAIN_ID` を分け，`xwd -id <rviz window>` で撮影し，背景色(48,48,48)の領域を切り出す．
- `detection/`: RF-DETR(Large) の ONNX (`~/yoloe_eval/models/rfdetr_l_far2_onnx/rfdetr-large.onnx`) を
  onnxruntime(CPU) で実行し，検出枠を描画 → `images/det_example_full.png`, `images/det_example_crop.png`．
  前処理は 640x640 引き伸ばし・RGB・ImageNet 正規化，閾値 0.3．
- TikZ 図: `images/fig_network.tex`(vision_planner n 版の構造), `fig_smooth.tex`, `fig_frenet.tex`, `fig_chassis.tex`(dell_ws chassis_driver).
