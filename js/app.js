// The page itself: image loading, the box, overlays and the report. All analysis is in the other js/ files.
(function () {
'use strict';
const $ = s => document.querySelector(s);
const LAYERS = [
  {id:'nodule', name:'Nodule outline', color:'var(--ov-nodule)', on:true},
  {id:'box',    name:'Your box',       color:'var(--ov-box)',    on:true},
  {id:'ring',   name:'Reference tissue',color:'var(--ov-ring)',  on:false},
  {id:'dark',   name:'Cystic spaces',  color:'var(--ov-dark)',   on:false},
  {id:'foci',   name:'Bright foci',    color:'var(--ov-foci)',   on:false},
];
const state = {file:null, gray:null, img:null, box:null, result:null, drag:null, isSample:false,
               layers:Object.fromEntries(LAYERS.map(l=>[l.id,l.on]))};

/* ---------- upload ---------- */
const drop = $('#drop'), fileIn = $('#file');
drop.onclick = () => fileIn.click();
drop.ondragover = e => {e.preventDefault(); drop.classList.add('over');};
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => {e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);};
fileIn.onchange = () => {if (fileIn.files[0]) loadFile(fileIn.files[0]);};
$('#reset').onclick = () => {
  state.file = state.gray = state.img = state.result = state.box = null;
  state.isSample = false;
  drop.hidden = false; $('#samplerow').hidden = false; $('#viewer').hidden = true;
  for (const id of ['scorecard','catcard','featcard','methodcard']) $('#'+id).hidden = true;
  $('#error').hidden = true; $('#imgmeta').textContent = ''; fileIn.value = '';
};

function showError(msg){ const e = $('#error'); e.textContent = msg; e.hidden = false; }

function escapeHtml(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function setEngine(cls, html){ $('#engine').className = 'engine ' + cls; $('#enginetxt').innerHTML = html; }

/* ---------- loading an image ---------- */
// The picture shown is built from the very grayscale pixels that will be analysed (after the
// 640 px limit), so the box and every overlay sit on exactly what the model measures.
function grayCanvas(p){
  const c = document.createElement('canvas'); c.width = p.w; c.height = p.h;
  const im = new ImageData(p.w, p.h), d = im.data;
  for (let i = 0, k = 0; i < p.gray.length; i++, k += 4){ d[k] = d[k+1] = d[k+2] = p.gray[i]; d[k+3] = 255; }
  c.getContext('2d').putImageData(im, 0, 0);
  return c;
}

async function loadFile(f, opts = {}){
  $('#error').hidden = true;
  let dec, prep;
  try { dec = await IWSP.decodeFile(f); prep = IWSP.limitSize(dec.gray, dec.h, dec.w); }
  catch (err) { return showError(err.message || String(err)); }
  state.file = f; state.gray = prep; state.img = grayCanvas(prep); state.isSample = !!opts.sample;
  state.W = prep.w; state.H = prep.h;
  state.box = opts.box ? opts.box.slice() : [state.W*0.2, state.H*0.2, state.W*0.8, state.H*0.8];
  state.result = null;
  drop.hidden = true; $('#samplerow').hidden = true; $('#viewer').hidden = false;
  for (const id of ['scorecard','catcard','featcard','methodcard']) $('#'+id).hidden = true;
  $('#status').textContent = '';
  $('#imgmeta').textContent = (opts.sample ? 'sample · ' : '') + `${dec.w}×${dec.h} px` +
     (prep.w !== dec.w || prep.h !== dec.h ? ` → analysed at ${prep.w}×${prep.h}` : '');
  buildToggles(); draw();
  return true;
}

async function sampleFile(){
  const blob = await (await fetch(IWSP.MODEL.sample.data_url)).blob();
  return new File([blob], IWSP.MODEL.sample.name, {type:'image/png'});
}
$('#samplenote').textContent = IWSP.MODEL.sample.note;
$('#sample').onclick = () => runSample().catch(err => showError(err.message || String(err)));
async function runSample(){
  if (await loadFile(await sampleFile(), {sample:true, box:IWSP.MODEL.sample.box})) await analyse();
}

/* ---------- canvas ---------- */
const cv = $('#cv'), ctx = cv.getContext('2d');
function draw(){
  if (!state.img) return;
  cv.width = state.W; cv.height = state.H;
  ctx.drawImage(state.img, 0, 0);
  const ev = state.result && state.result.evidence;
  const css = getComputedStyle(document.documentElement);
  const col = n => css.getPropertyValue(n).trim();

  if (ev && state.layers.ring){
    ctx.save(); ctx.fillStyle = col('--ov-ring') + '38'; ctx.beginPath();
    for (const p of ev.ring_outer) poly(p);
    for (const p of ev.nodule) poly(p);
    ctx.fill('evenodd'); ctx.restore();
  }
  if (ev && state.layers.dark){
    ctx.save(); ctx.fillStyle = col('--ov-dark') + '55'; ctx.strokeStyle = col('--ov-dark');
    ctx.lineWidth = 1; ctx.beginPath(); for (const p of ev.dark) poly(p);
    ctx.fill(); ctx.stroke(); ctx.restore();
  }
  if (state.box && state.layers.box){
    const [x1,y1,x2,y2] = state.box;
    ctx.save(); ctx.strokeStyle = col('--ov-box'); ctx.lineWidth = 1.5; ctx.setLineDash([5,4]);
    ctx.strokeRect(x1, y1, x2-x1, y2-y1); ctx.restore();
  }
  if (ev && state.layers.nodule){
    ctx.save(); ctx.strokeStyle = col('--ov-nodule'); ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath(); for (const p of ev.nodule) poly(p); ctx.stroke(); ctx.restore();
  }
  if (ev && state.layers.foci){
    ctx.save(); ctx.strokeStyle = col('--ov-foci'); ctx.lineWidth = 1.5;
    for (const f of ev.foci){ ctx.beginPath(); ctx.arc(f.x, f.y, 5, 0, 6.2832); ctx.stroke(); }
    ctx.restore();
  }
}
function poly(pts){
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
  ctx.closePath();
}
function at(e){
  const r = cv.getBoundingClientRect();
  const k = cv.width / r.width;
  return [Math.max(0, Math.min(cv.width,  (e.clientX - r.left) * k)),
          Math.max(0, Math.min(cv.height, (e.clientY - r.top)  * k))];
}
cv.addEventListener('pointerdown', e => {
  if (!state.img) return;
  cv.setPointerCapture(e.pointerId); state.drag = at(e);
});
cv.addEventListener('pointermove', e => {
  if (!state.drag) return;
  const [x,y] = at(e), [x0,y0] = state.drag;
  state.box = [Math.min(x0,x), Math.min(y0,y), Math.max(x0,x), Math.max(y0,y)];
  draw();
});
cv.addEventListener('pointerup', () => {
  if (state.drag && state.box && (state.box[2]-state.box[0] < 10 || state.box[3]-state.box[1] < 10))
    state.box = [state.W*0.2, state.H*0.2, state.W*0.8, state.H*0.8];
  state.drag = null; draw();
});

function buildToggles(){
  const t = $('#toggles'); t.innerHTML = '';
  for (const l of LAYERS){
    const b = document.createElement('label');
    b.className = 'tg' + (state.layers[l.id] ? ' on' : '');
    b.style.color = l.color;
    b.innerHTML = `<span class="sw"></span><span style="color:var(--fg)">${l.name}</span>`;
    b.onclick = () => { state.layers[l.id] = !state.layers[l.id];
      b.classList.toggle('on', state.layers[l.id]); draw(); };
    t.appendChild(b);
  }
}

/* ---------- analyse ---------- */
$('#run').onclick = () => analyse();

function analyse(){
  if (!state.gray) return Promise.resolve();
  const btn = $('#run'); btn.disabled = true;
  $('#status').textContent = 'Segmenting and measuring…'; $('#error').hidden = true;
  // let the browser paint the status line before the (synchronous) computation starts
  return new Promise(done => setTimeout(() => {
    try {
      const t0 = performance.now(), g = state.gray;
      const data = IWSP.analyseGray(g.gray, g.h, g.w, state.box);
      state.result = data; state.box = data.box;
      $('#status').textContent = `done in ${((performance.now() - t0) / 1000).toFixed(2)} s`;
      render(data); draw();
    } catch (err) {
      $('#status').textContent = '';
      showError(err.message || String(err));
    } finally { btn.disabled = false; done(); }
  }, 30));
}

/* ---------- self-check ---------- */
// On demand, the page analyses the bundled sample and compares all measurements and the
// TI-RADS call with what the original Python pipeline produced for the same image and box.
function offerSelfCheck(){
  setEngine('', 'Not yet checked in this browser. <button type="button" class="linkbtn" id="selfcheck">Run self-check</button>');
  $('#selfcheck').onclick = () => { runSelfCheck(); };
}
const runSelfCheck = () => (async () => {
  setEngine('loading', 'Running the self-check&hellip;');
  await new Promise(r => setTimeout(r, 60));               // let the status line paint first
  const M = IWSP.MODEL, g = M.golden, T = M.tolerance;
  const dec = await IWSP.decodeFile(await sampleFile()), p = IWSP.limitSize(dec.gray, dec.h, dec.w);
  const res = IWSP.analyseGray(p.gray, p.h, p.w, M.sample.box);
  let worst = 0, worstName = '', bad = [];
  for (const [k, ref] of Object.entries(g.features)){
    const v = res.features[k];
    if (ref === null || !Number.isFinite(v)){ if (!(ref === null && !Number.isFinite(v))) bad.push(k); continue; }
    const err = Math.abs(v - ref) / (T.abs / T.rel + Math.abs(ref));
    if (err > worst){ worst = err; worstName = k; }
    if (err > T.rel) bad.push(k);
  }
  const chosen = Object.fromEntries(Object.entries(res.categories).map(([k, c]) => [k, c.chosen]));
  const sameCall = res.tr === g.tr && res.total_points === g.total_points && JSON.stringify(chosen) === JSON.stringify(g.chosen);
  const n = Object.keys(g.features).length, c = {ok: bad.length === 0 && sameCall, sameCall, bad, worst, worstName, n};
  if (c.ok) setEngine('ok', `Self-check passed: all ${n} measurements and the TI-RADS call match the reference
      Python pipeline <span class="mono">(largest difference ${worst.toExponential(1)})</span>`);
  else setEngine('warn', `<strong>Self-check mismatch.</strong> In this browser ${bad.length} of ${n} measurements differ from the
      reference pipeline${sameCall ? '' : ', and the sample scores a different TI-RADS level'}. Results may not match the validated model.`);
  return c;
})().catch(err => {
  setEngine('bad', `<strong>The self-check could not run.</strong> ${escapeHtml(err.message || String(err))}`);
  return {ok:false, error:String(err)};
});
offerSelfCheck();
$('#buildinfo').textContent = `JavaScript port of segment.py (md5 ${IWSP.MODEL.source_md5['segment.py'].slice(0,8)}) and features.py ` +
    `(md5 ${IWSP.MODEL.source_md5['features.py'].slice(0,8)}) · model data exported ${IWSP.MODEL.built}`;

/* ---------- report ---------- */
const RISK = {1:['Benign','No follow-up required under ACR TI-RADS.'],
              2:['Not suspicious','Follow-up only at larger sizes.'],
              3:['Mildly suspicious','FNA at ≥25 mm, follow at ≥15 mm.'],
              4:['Moderately suspicious','FNA at ≥15 mm, follow at ≥10 mm.'],
              5:['Highly suspicious','FNA at ≥10 mm, follow at ≥5 mm.']};

function rel(auc, n){
  if (auc == null) return ['', '', ''];
  if (n < 20) return ['low', `only ${n} validation case${n === 1 ? '' : 's'}`,
                      'Too few cases in the validation split to judge this descriptor. Treat it as unverified.'];
  const c = auc >= 0.85 ? 'good' : auc >= 0.75 ? 'mid' : 'low';
  return [c, `AUC ${auc.toFixed(2)} \u00b7 n=${n}`,
          `Ranks a positive above a negative ${(auc*100).toFixed(0)}% of the time on ${n} held-out cases.`];
}

function render(d){
  const tr = d.tr;
  const box = $('#trbox');
  box.innerHTML = `
    <div class="trbadge" style="--c:var(--tr${tr})">
      <div class="k">TI-RADS</div><div class="v">${tr}</div>
    </div>
    <div class="trmeta">
      <div class="risk">${RISK[tr][0]}</div>
      <p>${RISK[tr][1]}</p>
      <p style="margin-top:6px">${d.total_points} ACR point${d.total_points === 1 ? '' : 's'} in total${
        d.spongiform_override ? ' — scored 0 because the nodule reads as spongiform' : ''}.</p>
    </div>`;

  const order = ['composition','echogenicity','shape','margin','foci'];
  const sum = $('#sum'); sum.innerHTML = '';
  order.forEach((g, i) => {
    if (!(g in d.points)) return;
    if (i) sum.insertAdjacentHTML('beforeend', '<span class="op">+</span>');
    sum.insertAdjacentHTML('beforeend',
      `<span class="term"><i>${d.categories[g].title}</i><b>${d.points[g]}</b></span>`);
  });
  sum.insertAdjacentHTML('beforeend',
    `<span class="op">=</span><span class="term eqtotal"><i>total</i><b>${d.total_points}</b></span>
     <span class="op">→</span><span class="term eqtotal"><i>level</i><b>TR${tr}</b></span>`);

  const host = $('#cats'); host.innerHTML = '';
  for (const g of order){
    const c = d.categories[g]; if (!c) continue;
    const picked = c.chosen.length ? c.chosen.map(k => d.probs[k].title).join(' + ')
                                   : 'None detected';
    const lead = c.chosen[0] ? d.probs[c.chosen[0]] : null;
    const [rc, rt, rtip] = lead ? rel(lead.auc, lead.n_pos) : ['','',''];
    const el = document.createElement('div');
    el.className = 'cat';
    el.innerHTML = `
      <button class="cat-h" aria-expanded="false">
        <span class="chev">&#9656;</span>
        <span class="nm">${c.title}</span>
        <span class="pick">${picked}</span>
        ${rt ? `<span class="pill ${rc}" title="${rtip}">${rt}</span>` : ''}
        <span class="pts">${d.points[g]} pt${d.points[g] === 1 ? '' : 's'}</span>
      </button>
      <div class="cat-b">${catBody(c, d)}</div>`;
    const btn = el.querySelector('.cat-h');
    btn.onclick = () => { const open = el.classList.toggle('open');
                          btn.setAttribute('aria-expanded', open); };
    host.appendChild(el);
  }
  const tb = $('#ftbody'); tb.innerHTML = '';
  for (const [k, v] of Object.entries(d.features)){
    const tr2 = document.createElement('tr');
    tr2.innerHTML = `<td class="k mono">${k}</td>
      <td class="n mono">${fmt(v)}</td>
      <td class="d">${d.feature_docs[k] || ''}</td>`;
    tb.appendChild(tr2);
  }
  $('#method').textContent = d.method;
  for (const id of ['scorecard','catcard','featcard','methodcard']) $('#'+id).hidden = false;
}

function catBody(c, d){
  let h = '<div class="rank">';
  for (const k of c.ranked){
    const p = d.probs[k], sel = c.chosen.includes(k);
    h += `<div class="lab ${sel ? 'sel' : ''}">${p.title}
            <span style="color:var(--dim)"> &middot; ${c.points_table[k]} pt</span></div>
          <div class="mono" style="font-size:12px;color:${sel ? 'var(--fg)' : 'var(--muted)'}">
            ${(p.p*100).toFixed(1)}%</div>
          <div class="bar"><i class="${sel ? 'sel' : ''}" style="width:${Math.max(1.5, p.p*100).toFixed(1)}%"></i></div>`;
  }
  h += '</div>';
  const lead = c.chosen[0] ? d.probs[c.chosen[0]] : d.probs[c.ranked[0]];
  if (!lead) return h;
  const note = c.type === 'exclusive'
    ? 'The highest probability in this category wins, as ACR allows only one.'
    : 'Each focus type counts on its own at a probability of 50% or more, and the points add up.';
  h += `<div class="why"><h3>What drove &ldquo;${lead.title}&rdquo;</h3>
        <p style="margin:-4px 0 10px;color:var(--muted);font-size:12px">${note}
        Bars show each measurement's push on the log-odds; they sum with the intercept
        (${lead.intercept.toFixed(2)}) to ${lead.logit.toFixed(2)}.</p>`;
  const max = Math.max(...lead.drivers.map(x => Math.abs(x.contribution)), 0.001);
  for (const dr of lead.drivers){
    const pos = dr.contribution >= 0, w = Math.abs(dr.contribution) / max * 50;
    h += `<div class="drv">
      <div class="drv-t"><code class="mono">${dr.feature}</code>
        <span class="drv-n mono ${pos ? 'p' : 'n'}">${pos ? '+' : ''}${dr.contribution.toFixed(2)}</span></div>
      <div class="cbar"><span class="mid"></span>
        <i style="${pos ? 'left:50%' : `right:50%`};width:${w}%;background:${pos ? 'var(--pos)' : 'var(--neg)'}"></i></div>
      <p>${dr.doc}</p>
      <div class="val mono">measured ${fmt(dr.value)} &middot; ${dr.z >= 0 ? '+' : ''}${dr.z.toFixed(2)} SD vs training mean</div>
    </div>`;
  }
  return h + '</div>';
}

function fmt(v){
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 100000)) return v.toExponential(2);
  return (Math.round(v * 1000) / 1000).toString();
}

if (location.hash === '#sample') runSample().catch(err => showError(err.message || String(err)));

$('#fsearch').oninput = e => {
  const q = e.target.value.toLowerCase();
  for (const tr of $('#ftbody').children)
    tr.hidden = q && !tr.textContent.toLowerCase().includes(q);
};
window.IWSP_APP = {state, runSample, analyse, loadFile, runSelfCheck};
})();
