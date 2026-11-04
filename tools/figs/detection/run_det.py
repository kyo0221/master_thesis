import sys, glob, os, numpy as np, cv2, onnxruntime as ort
onnx, indir, outdir = sys.argv[1:4]
sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0].name
names = ['pylon', 'dynamic obstacle', 'red signal', 'green signal']
colors = [(0,165,255), (255,0,255), (0,0,255), (0,200,0)]
mean = np.array([0.485,0.456,0.406], np.float32); std = np.array([0.229,0.224,0.225], np.float32)
for path in sorted(glob.glob(os.path.join(indir, '*.png'))):
    img = cv2.imread(path); h, w = img.shape[:2]
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (640, 640)).astype(np.float32) / 255.0
    x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
    outs = sess.run(None, {inp: x})
    dets = [o for o in outs if o.shape[-1] == 4][0][0]
    logits = [o for o in outs if o.shape[-1] == 5][0][0]
    prob = 1.0 / (1.0 + np.exp(-logits)); cls = prob.argmax(1); score = prob.max(1)
    keep = score >= 0.3
    found = []
    for (cx, cy, bw, bh), c, s in zip(dets[keep], cls[keep], score[keep]):
        x1, y1, x2, y2 = int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)
        if c > 3: continue
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[c], 2)
        label = f'{names[c]} {s:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), colors[c], -1)
        cv2.putText(img, label, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        found.append((names[c], round(float(s), 2), x1, y1, x2, y2))
    out = os.path.join(outdir, os.path.basename(path).replace('.png', '_det.png'))
    cv2.imwrite(out, img)
    print(os.path.basename(path), f'{w}x{h}', found)
