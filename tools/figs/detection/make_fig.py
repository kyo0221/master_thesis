import numpy as np, cv2, onnxruntime as ort, sys
onnx, src, out_full, out_crop = sys.argv[1:5]
sess = ort.InferenceSession(onnx, providers=['CPUExecutionProvider']); inp = sess.get_inputs()[0].name
names = ['Pylon', 'Dynamic obstacle', 'Red signal', 'Green signal']; colors = [(0,165,255), (255,0,255), (0,0,255), (0,200,0)]
mean = np.array([0.485,0.456,0.406], np.float32); std = np.array([0.229,0.224,0.225], np.float32)
img = cv2.imread(src); h, w = img.shape[:2]
x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (640, 640)).astype(np.float32) / 255.0
x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
outs = sess.run(None, {inp: x})
dets = [o for o in outs if o.shape[-1] == 4][0][0]; logits = [o for o in outs if o.shape[-1] == 5][0][0]
prob = 1/(1+np.exp(-logits)); cls = prob.argmax(1); score = prob.max(1); keep = score >= 0.3
boxes = [((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h, int(c), float(s)) for (cx,cy,bw,bh),c,s in zip(dets[keep],cls[keep],score[keep]) if c <= 3]
def draw(canvas, scale, ox, oy, thick, fs):
    for x1,y1,x2,y2,c,s in boxes:
        p1 = (int((x1-ox)*scale), int((y1-oy)*scale)); p2 = (int((x2-ox)*scale), int((y2-oy)*scale))
        cv2.rectangle(canvas, p1, p2, colors[c], thick)
        label = f'{names[c]} {s:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        ty = p1[1]-4 if p1[1]-th-8 > 0 else p2[1]+th+8
        cv2.rectangle(canvas, (p1[0], ty-th-4), (p1[0]+tw+4, ty+2), colors[c], -1)
        cv2.putText(canvas, label, (p1[0]+2, ty-1), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), thick, cv2.LINE_AA)
full = img.copy(); draw(full, 1.0, 0, 0, 1, 0.4)
# クロップ範囲（検出枠を含む領域に余白）
xs = [b[0] for b in boxes]+[b[2] for b in boxes]; ys = [b[1] for b in boxes]+[b[3] for b in boxes]
cx0, cy0, cx1, cy1 = int(min(xs))-40, int(min(ys))-30, int(max(xs))+40, int(max(ys))+20
cx0, cy0 = max(cx0,0), max(cy0,0); cx1, cy1 = min(cx1,w), min(cy1,h)
cv2.rectangle(full, (cx0,cy0), (cx1,cy1), (255,255,255), 1)
crop = img[cy0:cy1, cx0:cx1]; sc = 4.0
crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC); draw(crop, sc, cx0, cy0, 2, 0.6)
cv2.imwrite(out_full, full); cv2.imwrite(out_crop, crop)
print(boxes); print('crop', cx0, cy0, cx1, cy1, crop.shape)
