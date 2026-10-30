"""項目3: 図の網掛け（指示が straight 以外の区間）と、抑制あり速度が 4.0 m/s を超える箇所のトレース"""
import csv, sys, pickle
sys.path.insert(0, '/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/extra4')
from lib4 import *
rows = []
out = ["# 項目3: sil_branch_speed_unified.png の網掛けと、抑制あり速度が 4.0 m/s を超える箇所\n",
       "## 網掛けの定義\n",
       "- 網掛け = `sil.py` が記録した指示状態（`/planning/nav_cmd` に出した値、ログ第14列 = nav_cmd が straight 以外）が連続して 1 の区間。コース上の固定区間ではなく走行ごとの記録で、`speed_path_planner` は最新の nav_cmd が straight 以外の間だけ `branch_linear_max.vel=4.0` を v_max に適用するので、網掛けは速度上限が実際に有効だった区間に（DDS 遅延 約 0.05 s の差で）一致する。",
       "- 最初の図は 3 m 以下の間隔を 1 つに結合していたため、指示が straight に戻る短い区間（約 1.5 m）が網掛けの内側に隠れていた。再出力した図は結合せず、各区間を別に網掛けしている。\n",
       "## 指示（nav_cmd）の切替位置と、その後の速度（基準・青、5シード平均）\n",
       "| 上限 | 指示 on（s）| off→on の隙間（straight になる区間）| 指示 off（s）| 隙間の長さ [m, s] | 入口 s=108.5 m の実速度 [m/s] | 隙間後の速度ピーク [m/s]（位置 s） |", "|---|---|---|---|---|---|---|"]
for cap in (55, 70):
    G = []; ENT = []; PK1 = []; PK2 = []; S1 = []; S2 = []; ON = []; OFF = []; GL = []; GT = []
    for sd in range(1, 6):
        L, A = load(f'base_blue_v{cap}_s{sd}'); m = L[:, 9] < LAP['blue']; L = L[m]; nav = L[:, 13] > 0.5; ch = np.where(np.diff(nav.astype(int)) != 0)[0] + 1
        ON.append(L[ch[0], 8]); OFF.append(L[ch[-1], 8]); gs = [(L[ch[1], 8], L[ch[2], 8], L[ch[2], 0] - L[ch[1], 0]), (L[ch[3], 8], L[ch[4], 8], L[ch[4], 0] - L[ch[3], 0])]
        GL.append([g[1] - g[0] for g in gs]); GT.append([g[2] for g in gs])
        ENT.append(L[ch[0], 3])
        for k, (a, b, _) in enumerate(gs):
            z = (L[:, 8] >= b) & (L[:, 8] <= b + 12); i = np.argmax(L[z, 3]); (PK1 if k == 0 else PK2).append(L[z, 3][i]); (S1 if k == 0 else S2).append(L[z, 8][i])
        rows.append([cap / 10, sd, f'{ON[-1]:.1f}', f'{gs[0][0]:.1f}', f'{gs[0][1]:.1f}', f'{gs[1][0]:.1f}', f'{gs[1][1]:.1f}', f'{OFF[-1]:.1f}', f'{ENT[-1]:.2f}', f'{PK1[-1]:.2f}', f'{S1[-1]:.1f}', f'{PK2[-1]:.2f}', f'{S2[-1]:.1f}'])
    gl = np.mean(GL, axis=0); gt = np.mean(GT, axis=0)
    out.append(f"| {cap/10:.1f} | {np.mean(ON):.1f} | 1 回目 {np.mean([r[0] for r in [[g] for g in [x[0] for x in []]]] or [0]) if False else ''}s≈129（{gl[0]:.1f} m, {gt[0]:.2f} s）/ 2 回目 s≈154（{gl[1]:.1f} m, {gt[1]:.2f} s） | {np.mean(OFF):.1f} | 1 回目 {gl[0]:.1f}, {gt[0]:.2f} / 2 回目 {gl[1]:.1f}, {gt[1]:.2f} | {np.mean(ENT):.2f} | 1 回目後 {np.mean(PK1):.2f}±{np.std(PK1, ddof=1):.2f}（s={np.mean(S1):.0f}）/ 2 回目後 {np.mean(PK2):.2f}±{np.std(PK2, ddof=1):.2f}（s={np.mean(S2):.0f}） |")
out += ["\n（5 シードすべてで同じ位置・長さの隙間が生じる。）\n",
        "## 抑制あり速度が 4.0 m/s を超える原因\n",
        "1. **指示が straight に戻る隙間（s≈128.5〜130 m と 153〜154.8 m、約 0.4 s）**: SIL の指令生成（`SIL_LEFT_NODES` の 3 ノード: 分岐 left → 右 → T字 left）が、1 つ目のノードの旋回完了（旋回量 60° 到達または 20 m 離脱）で straight を出し、次のノードが 20 m 以内に入るまで straight が続くために生じる。その間は速度上限が解除され、速度計画の v_max が基準の上限（5.5 / 7.0）に戻り、速度指令が 5.3〜5.5 m/s まで上がる（トレース: s=130.2 m で指令 5.38、実速度 4.18 → s=133.6 m で 4.86）。",
        "2. **指示が再び有効になった後の遅れ**: 上限 4.0 の再適用後も、車両の速度応答（縦の一次遅れ 0.60 s、むだ時間 0.06 s）と速度計画の周期（10 Hz）、コントローラの 0.3 s 先読みのため、実速度が 4.0 に戻るまで約 8 m（約 1.6 s）かかる。ピークは隙間の終端から約 3.5 m 先（0.8 s 後）。",
        "3. **入口の過渡**: 指示が最初に on になる s=108.5 m（交差点の 20 m 手前）で実速度は上限速度（5.3 / 6.7 m/s）のままで、4.0 m/s まで減速するのに数十 m かかる（網掛けの入口付近の >4.0 は減速途中）。",
        "- 結論: 4.0 m/s を超える箇所は、(1) 指示が straight に戻る隙間の直後（4.7〜4.9 m/s）、(2) 入口の減速過渡に限られ、網掛けの内側で上限が効いていない区間があるわけではない。ただし隙間は SIL の指令生成モデルに由来し、実車の topomap 指示で同じ隙間が生じるかは確認していない（隙間が無ければ、抑制ありの速度は 4.0 m/s 付近に保たれ、ラップ短縮の比較（A7）の抑制による損失がやや大きく出る）。"]
open(f'{ROOT}/extra5/u3_shading.md', 'w').write('\n'.join(out) + '\n')
with open(f'{ROOT}/extra5/u3_nav_gaps.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['vmax_mps', 'seed', 'nav_on_s', 'gap1_start_s', 'gap1_end_s', 'gap2_start_s', 'gap2_end_s', 'nav_off_s', 'v_at_entry_mps', 'peak_after_gap1_mps', 'peak_after_gap1_s', 'peak_after_gap2_mps', 'peak_after_gap2_s']); w.writerows(rows)
print('\n'.join(out[:12]))
