// Dev smoke for annotate.html (Node, no browser): runs the EXACT page script
// through init + file load + polygon tracing + export, asserting the schema.
// Usage: node eval/mask_pilot/smoke_harness.js
const fs = require("fs");
const path = __dirname + "/annotate.html";
const html = fs.readFileSync(path, "utf8");
const src = html.split("<script>")[1].split("</script>")[0];

function el(id) {
  return {
    id, value: "", files: [], textContent: "", title: "",
    width: 300, height: 150, children: [],
    onclick: null, ondblclick: null, onchange: null, onmousemove: null,
    listeners: {},
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
    click() { if (this.onclick) this.onclick({}); },
    getBoundingClientRect() { return {left: 0, top: 0}; },
  };
}
const ids = {};
const cv = el("cv");
cv.getContext = () => new Proxy({}, { get: (t, k) => (k === "canvas" ? cv : (...a) => {}) });
Object.defineProperty(cv, "clientWidth", { get() { return cv.width; } });
ids.cv = cv;
["imgSel", "file", "close", "undo", "thin", "delInst", "neg", "export",
 "warn", "info", "list", "scroll"].forEach(id => { ids[id] = el(id); });
ids.imgSel.value = "cr5";
global.document = {
  getElementById: id => ids[id],
  createElement: tag => el("new-" + tag),
  addEventListener(t, f) { (this._kl = this._kl || {})[t] = f; },
};
let exported = null;
global.Blob = class { constructor(parts) { this.parts = parts; } };
global.URL = { createObjectURL: o => {
  if (o && o.parts) exported = o.parts.join("");
  return "blob:fake";
} };
global.Image = class {
  set src(u) { this.naturalWidth = 554; this.naturalHeight = 554; this.onload(); }
};
global.alert = m => { throw new Error("ALERT: " + m); };

// init (page script runs on load)
eval(src);

// drive: select cr5 -> pick file -> 3 clicks -> Enter closes -> export
ids.imgSel.onchange();
ids.file.files = [{}];
ids.file.onchange({ target: { files: [{}] } });
const click = (x, y) => cv.onclick({ clientX: x, clientY: y });
click(10, 10); click(60, 12); click(30, 70);
document._kl.keydown({ key: "Enter" });
ids.export.click();

const out = JSON.parse(exported);
const assert = require("assert");
assert.strictEqual(out.version, 1);
assert.strictEqual(out.images.length, 7);
const cr5 = out.images.find(i => i.image_id === "cr5");
assert.strictEqual(cr5.instances.length, 1);
assert.deepStrictEqual(cr5.instances[0].polygons, [[[10, 10], [60, 12], [30, 70]]]);
assert.strictEqual(cr5.instances[0].instance_id, "cr5-c1");
assert.strictEqual(out.images.find(i => i.image_id === "in1").negative, true);
assert.strictEqual(out.images.find(i => i.image_id === "in1").instances.length, 0);
console.log("SMOKE PASS: init + file load + 3-vertex polygon + Enter-close +",
  "schema export verified (" + out.images.length + " images).");
