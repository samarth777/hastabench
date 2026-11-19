/* HastaBench report: charts, grids and the 3-D replay explorer. Data: window.HB (summary.js). */
(function () {
  const HB = window.HB;
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const INK = css('--ink'), INK2 = css('--ink-2'), INK3 = css('--ink-3'), GRID = css('--grid');
  const FAM = [
    { k: 'Anthropic', test: (m) => /claude/i.test(m), c: css('--s1') },
    { k: 'OpenAI', test: (m) => /gpt/i.test(m), c: css('--s2') },
    { k: 'Google', test: (m) => /gemini/i.test(m), c: css('--s3') },
    { k: 'Moonshot', test: (m) => /kimi/i.test(m), c: css('--s4') },
  ];
  const fam = (m) => FAM.find((f) => f.test(m)) || { k: 'other', c: INK3 };
  const NS = 'http://www.w3.org/2000/svg';
  const sv = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const h = (tag, attrs = {}, html = '') => { const e = document.createElement(tag); for (const k in attrs) e.setAttribute(k, attrs[k]); e.innerHTML = html; return e; };
  const f1 = (x) => (x == null || isNaN(x) ? '–' : x.toFixed(1));
  const money = (x) => (x == null ? '–' : x < 0.1 ? '$' + x.toFixed(3) : '$' + x.toFixed(2));
  const tip = $('tip');
  function showTip(evt, html) {
    tip.innerHTML = html; tip.style.opacity = 1;
    const x = evt.clientX + 14, y = evt.clientY + 14;
    tip.style.left = Math.min(x, window.innerWidth - tip.offsetWidth - 10) + 'px'; tip.style.top = y + 'px'; tip.style.position = 'fixed';
  }
  const hideTip = () => (tip.style.opacity = 0);

  const models = HB.models, refs = HB.refs;
  const refByModel = Object.fromEntries(refs.map((r) => [r.model, r]));
  const T = HB.targets;

  // ------------------------------------------------------------ tiles + headline
  (function tiles() {
    const best = models.filter((d) => d.effort === 'default')[0] || models[0];
    const solvedAny = new Set(); models.forEach((m) => m.solved_targets.forEach((t) => solvedAny.add(t)));
    const cmaS = refByModel['REF_cma_score'], mean = refByModel['REF_mean_shape'];
    const items = [
      [f1(best.mean), `best mean HastaScore (${best.name})`],
      [`${best.solved}/28`, `hastas solved by the best model`],
      [`${solvedAny.size}/28`, `hastas solved by at least one of ${models.length} model configurations`],
      [cmaS ? f1(cmaS.mean) : '–', `blind CMA-ES on the hidden score (2,000 poses)`],
    ];
    $('tiles').innerHTML = items.map(([v, l]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
    const beat = mean ? models.filter((m) => m.mean > mean.mean).length : 0;
    $('headline').innerHTML = `<b>Headline.</b> ${best.name} leads with a mean HastaScore of ${f1(best.mean)} [${f1(best.ci[0])}, ${f1(best.ci[1])}] and solves ${best.solved} of 28 hastas. ` +
      `Altogether the ${models.length} model configurations solve ${solvedAny.size} distinct hastas. ` +
      (mean ? `${beat} of ${models.length} configurations beat the featureless average-hasta prior (${f1(mean.mean)}). ` : '') +
      (cmaS ? `A blind search that may query the hidden score reaches ${f1(cmaS.mean)} and solves ${cmaS.solved}, so most targets are reachable within the rules. Models get there far less often.` : '');
  })();

  // ------------------------------------------------------------ findings (computed from the data)
  (function findings() {
    const def = models.filter((d) => d.effort === 'default');
    const best = def[0], li = [];
    const solvedBy = {}; def.forEach((d) => d.solved_targets.forEach((t) => (solvedBy[t] = (solvedBy[t] || 0) + 1)));
    const never = HB.all28.filter((t) => !solvedBy[t]);
    const meanT = (t) => def.reduce((a, d) => a + (d.per_target[t] ?? 0), 0) / def.length;
    const hardest = HB.all28.slice().sort((a, b) => meanT(a) - meanT(b)).slice(0, 3);
    li.push(`<b>No model is reliable.</b> The best, ${best.name}, averages ${f1(best.mean)} and solves ${best.solved} of 28 hastas. ${never.length} of 28 hastas are solved by no model at default effort; the hardest on average are ${hardest.join(', ')}.`);
    const cs = refByModel['REF_cma_score'], ci = refByModel['REF_cma_image'], ms = refByModel['REF_mean_shape'];
    if (cs && ci) li.push(`<b>The targets are reachable.</b> A blind search that may query the hidden score solves ${cs.solved}/28 (mean ${f1(cs.mean)}). The same search guided only by image silhouettes reaches ${f1(ci.mean)}, so silhouettes alone are a weak signal. ${def.filter((d) => d.mean > ci.mean).length} of ${def.length} models beat it, which suggests they do read the images.`);
    if (ms) li.push(`<b>A featureless prior is competitive.</b> The average finger configuration of all 28 hastas, at the right orientation, scores ${f1(ms.mean)}. ${def.filter((d) => d.mean <= ms.mean).length} of ${def.length} models do no better.`);
    let H = 0, Hu = 0, N = 0; def.forEach((d) => { const q = HB.quality[d.key] || {}; H += q.helped || 0; Hu += q.hurt || 0; N += (q.helped || 0) + (q.hurt || 0) + (q.neutral || 0); });
    const lost = def.reduce((a, d) => a + (d.lost_gt2 || 0), 0), eps = def.reduce((a, d) => a + d.n, 0);
    li.push(`<b>Moves are poorly aimed.</b> Of ${N.toLocaleString()} accepted moves, ${Math.round((100 * H) / N)}% raised the hidden score and ${Math.round((100 * Hu) / N)}% lowered it. In ${lost} of ${eps} episodes the submitted pose scored more than 2 points below the episode's best.`);
    let fo = 0, fs = 0, fc = 0, nf = 0, th = 0; def.forEach((d) => { const f = HB.failure[d.key] || {}; fo += f.orientation || 0; fs += f.shape || 0; fc += f.contacts || 0; nf += f.n_fail || 0; th += f.finger_thumb || 0; });
    li.push(`<b>The thumb is the bottleneck.</b> Among unsolved episodes, ${Math.round((100 * th) / nf)}% miss with the thumb by more than 2 cm and ${Math.round((100 * fc) / nf)}% fail a required contact (such as fingertip on thumb tip). Only ${Math.round((100 * fo) / nf)}% fail on hand orientation.`);
    let named = 0, nn = 0, reads = 0; def.forEach((d) => { const k = HB.knowledge[d.key] || {}; named += k.named || 0; nn += k.n || 0; reads += k.read_eps || 0; });
    li.push(`<b>Models rarely identify the gesture, and rarely ask.</b> Only ${named} of ${nn} plans name the correct hasta, and the classical description was read in just ${reads} episodes, although a read costs only 6% of the score.`);
    if (HB.two_models.length) { const b2 = HB.two_models[0]; const s2 = HB.two_models.reduce((a, d) => a + d.solved, 0);
      li.push(`<b>Two hands are much harder.</b> On the five samyuta hastas the best model, ${b2.name}, averages ${f1(b2.mean)}, and ${s2} two-hand episodes are solved in total.`); }
    const byBase = {}; models.forEach((d) => (byBase[d.base] = byBase[d.base] || {})[d.effort] = d);
    const ef = Object.values(byBase).filter((g) => g.default && g.high);
    if (ef.length) li.push((ef.every((g) => Math.abs(g.high.mean - g.default.mean) < 5) ? `<b>Extra reasoning effort barely helps.</b> ` : `<b>Effort helps unevenly.</b> `) + ef.map((g) => `${g.default.name}: ${f1(g.default.mean)} → ${f1(g.high.mean)} at high effort (${money(g.default.cost)} → ${money(g.high.cost)} per episode)`).join('; ') + '.');
    $('findings').innerHTML = li.map((x) => `<li style="margin-bottom:.5em">${x}</li>`).join('');
  })();

  // ------------------------------------------------------------ Figure 1: score vs cost
  (function fig1() {
    const W = 1100, H = 430, m = { l: 56, r: 230, t: 16, b: 46 };
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Mean score against cost per episode' });
    const pts = models.filter((d) => d.cost != null);
    const lo = Math.log10(Math.min(...pts.map((d) => d.cost)) / 1.6), hi = Math.log10(Math.max(...pts.map((d) => d.cost)) * 1.6);
    const X = (c) => m.l + ((Math.log10(c) - lo) / (hi - lo)) * (W - m.l - m.r);
    const Y = (s) => m.t + (1 - s / 100) * (H - m.t - m.b);
    for (let s = 0; s <= 100; s += 20) {
      sv('line', { x1: m.l, x2: W - m.r, y1: Y(s), y2: Y(s), stroke: GRID }, svg);
      sv('text', { x: m.l - 8, y: Y(s) + 4, 'text-anchor': 'end', fill: INK2 }, svg).textContent = s;
    }
    [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3].forEach((c) => {
      if (Math.log10(c) < lo || Math.log10(c) > hi) return;
      sv('line', { x1: X(c), x2: X(c), y1: H - m.b, y2: H - m.b + 4, stroke: INK3 }, svg);
      sv('text', { x: X(c), y: H - m.b + 18, 'text-anchor': 'middle', fill: INK2 }, svg).textContent = '$' + c;
    });
    sv('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: INK3 }, svg);
    sv('text', { x: (m.l + W - m.r) / 2, y: H - 8, 'text-anchor': 'middle', fill: INK2 }, svg).textContent = 'mean API cost per episode (log scale)';
    sv('text', { x: 14, y: Y(50), transform: `rotate(-90 14 ${Y(50)})`, 'text-anchor': 'middle', fill: INK2 }, svg).textContent = 'mean HastaScore';
    // reference lines
    const refOrder = ['REF_ik_oracle', 'REF_cma_score', 'REF_cma_image', 'REF_mean_shape', 'REF_rest'];
    let lastY = -99;
    refOrder.map((k) => refByModel[k]).filter(Boolean).sort((a, b) => b.mean - a.mean).forEach((r) => {
      const y = Y(r.mean);
      sv('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: INK3, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' }, svg);
      const ly = Math.max(y + 4, lastY + 14); lastY = ly;
      sv('text', { x: W - m.r + 8, y: ly, fill: INK2 }, svg).textContent = `${r.name} · ${f1(r.mean)}`;
    });
    // connect effort levels of the same model
    const byBase = {}; pts.forEach((d) => (byBase[d.base] = byBase[d.base] || []).push(d));
    Object.values(byBase).forEach((g) => { if (g.length < 2) return; g.sort((a, b) => a.cost - b.cost);
      sv('path', { d: g.map((d, i) => (i ? 'L' : 'M') + X(d.cost) + ' ' + Y(d.mean)).join(''), fill: 'none', stroke: fam(g[0].model).c, 'stroke-width': 2, opacity: 0.6 }, svg); });
    // model dots with labels (simple collision avoidance)
    const placed = [];
    pts.slice().sort((a, b) => b.mean - a.mean).forEach((d) => {
      const x = X(d.cost), y = Y(d.mean), c = fam(d.model).c;
      sv('line', { x1: x, x2: x, y1: Y(d.ci[0]), y2: Y(d.ci[1]), stroke: c, 'stroke-width': 2, opacity: 0.35 }, svg);
      sv('circle', { cx: x, cy: y, r: 7, fill: c, stroke: '#fcfcfb', 'stroke-width': 2 }, svg);
      const tw = d.name.length * 6.6 + 4, th = 13;
      const cands = [[x + 11, y + 4], [x - 11 - tw, y + 4], [x + 11, y - 10], [x + 11, y + 18], [x - 11 - tw, y - 10], [x - 11 - tw, y + 18], [x - tw / 2, y - 14], [x - tw / 2, y + 24]];
      const boxOf = ([lx, ly]) => ({ x0: lx, x1: lx + tw, y0: ly - th + 3, y1: ly + 3 });
      const clash = (b) => placed.some((p) => !(b.x1 < p.x0 || b.x0 > p.x1 || b.y1 < p.y0 || b.y0 > p.y1)) ||
        pts.some((o) => { const ox = X(o.cost), oy = Y(o.mean); return ox > b.x0 - 8 && ox < b.x1 + 8 && oy > b.y0 - 8 && oy < b.y1 + 8; });
      let pick = cands.find((c) => !clash(boxOf(c))) || cands[0];
      placed.push(boxOf(pick));
      const [lx, ly] = pick;
      sv('text', { x: lx, y: ly, fill: INK }, svg).textContent = d.name;
      const hit = sv('circle', { cx: x, cy: y, r: 16, fill: 'transparent', style: 'cursor:pointer' }, svg);
      hit.addEventListener('mousemove', (e) => showTip(e, `<b>${d.name}</b><br>mean ${f1(d.mean)} [${f1(d.ci[0])}, ${f1(d.ci[1])}]<br>solved ${d.solved}/28 · ${money(d.cost)}/episode<br>${f1(d.time_min)} min/episode`));
      hit.addEventListener('mouseleave', hideTip);
    });
    const lg = h('div', { class: 'legend' }, FAM.filter((f) => models.some((m) => f.test(m.model))).map((f) => `<span><i class="sw" style="background:${f.c}"></i>${f.k}</span>`).join('') +
      `<span><i class="lg-line lg-dash"></i>reference baseline</span>`);
    $('fig1').appendChild(lg); $('fig1').appendChild(svg);
  })();

  // ------------------------------------------------------------ leaderboard
  (function lb() {
    const rows = models.map((d) => `<tr><td><i class="sw" style="background:${fam(d.model).c}"></i>${d.name}</td>
      <td class="num"><b>${f1(d.mean)}</b></td><td class="num">${f1(d.ci[0])}–${f1(d.ci[1])}</td><td class="num">${d.solved}/28</td>
      <td class="num">${f1(d.best)}</td><td class="num">${f1(d.lost)}</td><td class="num">${f1(d.calls)}</td><td class="num">${f1(d.looks)}</td>
      <td class="num">${d.reads.toFixed(2)}</td><td class="num">${money(d.cost)}</td><td class="num">${f1(d.time_min)}</td></tr>`).join('');
    const rr = refs.map((d) => `<tr class="ref"><td>${d.name}</td><td class="num">${f1(d.mean)}</td><td class="num">${f1(d.ci[0])}–${f1(d.ci[1])}</td><td class="num">${d.solved}/28</td><td colspan="7"></td></tr>`).join('');
    $('lb').innerHTML = `<tr><th>Model</th><th class="num">Score</th><th class="num">95% CI</th><th class="num">Solved</th><th class="num">Best</th><th class="num">Lost</th>
      <th class="num">Calls</th><th class="num">Looks</th><th class="num">Reads</th><th class="num">$/ep</th><th class="num">min/ep</th></tr>` + rows + rr;
  })();

  // ------------------------------------------------------------ grids
  const zoom = $('zoom');
  function grid(container, order, rowsM, rowsR, split) {
    const tb = h('table');
    let html = '<tr><th></th>' + order.map((t) => `<th class="col">${t}</th>`).join('') + '</tr>';
    html += '<tr><th>target</th>' + order.map((t) => `<td><span class="cell tgt" data-img="img/target/${t}_front.jpg"><img src="img/target/${t}_front.jpg"></span></td>`).join('') + '</tr>';
    const row = (d, isRef) => '<tr><th>' + d.name + '</th>' + order.map((t) => {
      const s = d.per_target[t];
      if (s == null) return '<td></td>';
      const solved = d.solved_targets.includes(t);
      const src = `img/final/${d.key}_${t}.jpg`;
      return `<td><span class="cell${solved ? ' solved' : ''}" data-img="${src}" data-key="${isRef ? '' : d.key}" data-t="${t}" title="${d.name} · ${t} · ${f1(s)}"><img src="${src}"><span class="b">${Math.round(s)}</span></span></td>`;
    }).join('') + '</tr>';
    html += rowsM.map((d) => row(d, false)).join('') + rowsR.map((d) => row(d, true)).join('');
    tb.innerHTML = html; container.appendChild(tb);
    container.querySelectorAll('.cell').forEach((c) => {
      c.addEventListener('mousemove', (e) => { zoom.style.display = 'block'; zoom.style.background = `#c9ced6 url(${c.dataset.img}) center/cover`; zoom.style.left = Math.min(e.clientX + 18, innerWidth - 240) + 'px'; zoom.style.top = Math.min(e.clientY + 18, innerHeight - 240) + 'px'; });
      c.addEventListener('mouseleave', () => (zoom.style.display = 'none'));
      if (c.dataset.key) c.addEventListener('click', () => { openReplay(c.dataset.key, c.dataset.t); $('stage').scrollIntoView({ behavior: 'smooth', block: 'center' }); });
    });
  }
  const gridRefs = refs.filter((r) => ['REF_cma_image', 'REF_cma_score', 'REF_mean_shape'].includes(r.model));
  grid($('grid'), HB.order, models, gridRefs, 'one');
  if (HB.two_models.length) grid($('grid2'), HB.order_two, HB.two_models, HB.two_refs.filter((r) => r.model !== 'REF_ik_oracle'), 'two');

  // ------------------------------------------------------------ Figure 4: per-call curves
  (function fig4() {
    let sel = models[0].key;
    const btns = $('curvebtns');
    models.forEach((d) => { const b = h('button', {}, d.name); b.onclick = () => { sel = d.key; draw(); }; b.dataset.k = d.key; btns.appendChild(b); });
    const W = 1100, H = 360, m = { l: 48, r: 110, t: 12, b: 40 };
    function draw() {
      btns.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.k === sel));
      $('fig4').innerHTML = '';
      const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
      const N = HB.budget;
      const X = (i) => m.l + (i / N) * (W - m.l - m.r), Y = (s) => m.t + (1 - s / 100) * (H - m.t - m.b);
      for (let s = 0; s <= 100; s += 20) { sv('line', { x1: m.l, x2: W - m.r, y1: Y(s), y2: Y(s), stroke: GRID }, svg); sv('text', { x: m.l - 8, y: Y(s) + 4, 'text-anchor': 'end', fill: INK2 }, svg).textContent = s; }
      for (let i = 0; i <= N; i += 10) sv('text', { x: X(i), y: H - m.b + 18, 'text-anchor': 'middle', fill: INK2 }, svg).textContent = i;
      sv('text', { x: (m.l + W - m.r) / 2, y: H - 6, 'text-anchor': 'middle', fill: INK2 }, svg).textContent = 'call number';
      const path = (arr) => arr.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1)).join('');
      models.forEach((d) => { if (d.key !== sel && HB.curves[d.key]) sv('path', { d: path(HB.curves[d.key].cur), fill: 'none', stroke: '#c9c8c2', 'stroke-width': 1.5 }, svg); });
      const c = HB.curves[sel], col = fam(models.find((d) => d.key === sel).model).c;
      sv('path', { d: path(c.best), fill: 'none', stroke: col, 'stroke-width': 2, 'stroke-dasharray': '6 4' }, svg);
      sv('path', { d: path(c.cur), fill: 'none', stroke: col, 'stroke-width': 2.5 }, svg);
      const nm = models.find((d) => d.key === sel).name;
      sv('text', { x: X(N) + 8, y: Y(c.best[N]) - 2, fill: INK }, svg).textContent = 'best so far';
      sv('text', { x: X(N) + 8, y: Y(c.cur[N]) + 12, fill: INK }, svg).textContent = 'current';
      // hover crosshair
      const cross = sv('line', { y1: m.t, y2: H - m.b, stroke: INK3, opacity: 0 }, svg);
      const hit = sv('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg);
      hit.addEventListener('mousemove', (e) => {
        const r = svg.getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * W;
        const i = Math.max(0, Math.min(N, Math.round(((px - m.l) / (W - m.l - m.r)) * N)));
        cross.setAttribute('x1', X(i)); cross.setAttribute('x2', X(i)); cross.setAttribute('opacity', 1);
        showTip(e, `<b>${nm}</b> after call ${i}<br>current ${f1(c.cur[i])} · best so far ${f1(c.best[i])}`);
      });
      hit.addEventListener('mouseleave', () => { cross.setAttribute('opacity', 0); hideTip(); });
      $('fig4').appendChild(h('div', { class: 'legend' }, `<span><i class="lg-line" style="border-color:${col}"></i>current pose</span><span><i class="lg-line lg-dash" style="border-color:${col}"></i>best so far</span><span><i class="lg-line" style="border-color:#c9c8c2"></i>other models (current)</span>`));
      $('fig4').appendChild(svg);
    }
    draw();
  })();

  // ------------------------------------------------------------ Figure 5: move quality
  (function fig5() {
    const W = 760, rowH = 26, m = { l: 150, r: 120, t: 22 };
    const H = m.t + rowH * models.length + 8;
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
    const cols = { helped: css('--s1'), neutral: '#c9c8c2', hurt: css('--s2') };
    const lg = h('div', { class: 'legend' }, Object.entries(cols).map(([k, c]) => `<span><i class="sw" style="background:${c}"></i>${k}</span>`).join(''));
    sv('text', { x: W - m.r + 10, y: 14, fill: INK2 }, svg).textContent = 'refused calls';
    models.forEach((d, i) => {
      const q = HB.quality[d.key] || {}, rf = HB.refusals[d.key] || {};
      const tot = (q.helped || 0) + (q.neutral || 0) + (q.hurt || 0);
      const y = m.t + i * rowH;
      sv('text', { x: m.l - 8, y: y + 15, 'text-anchor': 'end', fill: INK }, svg).textContent = d.name;
      let x = m.l;
      ['helped', 'neutral', 'hurt'].forEach((k) => {
        const w = tot ? ((q[k] || 0) / tot) * (W - m.l - m.r) : 0;
        if (w > 0) {
          const r = sv('rect', { x: x, y: y + 3, width: Math.max(w - 2, 0), height: rowH - 8, fill: cols[k], rx: 2 }, svg);
          r.addEventListener('mousemove', (e) => showTip(e, `<b>${d.name}</b><br>${k}: ${q[k] || 0} of ${tot} moves (${((100 * (q[k] || 0)) / tot).toFixed(0)}%)`));
          r.addEventListener('mouseleave', hideTip);
          if (w > 34) sv('text', { x: x + 5, y: y + 16, fill: k === 'neutral' ? INK : '#fff', 'font-size': 11 }, svg).textContent = Math.round((100 * (q[k] || 0)) / tot) + '%';
        }
        x += w;
      });
      const nref = Object.values(rf).reduce((a, b) => a + b, 0);
      const t = sv('text', { x: W - m.r + 10, y: y + 15, fill: INK2 }, svg); t.textContent = nref;
      const ttl = Object.entries(rf).map(([k, v]) => `${k}: ${v}`).join('<br>') || 'none';
      t.addEventListener('mousemove', (e) => showTip(e, `<b>${d.name}</b> refusals<br>${ttl}`)); t.addEventListener('mouseleave', hideTip);
    });
    $('fig5').appendChild(lg); $('fig5').appendChild(svg);
  })();

  // ------------------------------------------------------------ failure + knowledge tables
  (function tables() {
    const pct = (a, n) => (n ? Math.round((100 * a) / n) + '%' : '–');
    const barc = (a, n) => `<span class="bar" style="width:${n ? (50 * a) / n : 0}px"></span>${pct(a, n)}`;
    $('fail').innerHTML = '<tr><th>Model</th><th class="num">Unsolved</th><th>Orientation &gt; 20°</th><th>Shape off</th><th>Contacts unmet</th><th>Finger &gt; 20 mm (thumb · index · middle · ring · little)</th></tr>' +
      models.map((d) => { const f = HB.failure[d.key] || {}; const n = f.n_fail || 0;
        return `<tr><td>${d.name}</td><td class="num">${n}/${d.n}</td><td>${barc(f.orientation || 0, n)}</td><td>${barc(f.shape || 0, n)}</td><td>${barc(f.contacts || 0, n)}</td>
        <td class="mono">${['thumb', 'index', 'middle', 'ring', 'little'].map((k) => pct(f['finger_' + k] || 0, n)).join(' · ')}</td></tr>`; }).join('');
    $('know').innerHTML = '<tr><th>Model</th><th class="num">Plan names the hasta</th><th class="num">Episodes that read</th><th class="num">Score, read</th><th class="num">Score, no read</th><th class="num">Looks / episode</th><th class="num">Describes / episode</th></tr>' +
      models.map((d) => { const k = HB.knowledge[d.key] || {}, q = HB.quality[d.key] || {};
        return `<tr><td>${d.name}</td><td class="num">${k.named}/${k.n}</td><td class="num">${k.read_eps}</td><td class="num">${f1(k.score_read)}</td><td class="num">${f1(k.score_noread)}</td>
        <td class="num">${f1((q.n_look || 0) / d.n)}</td><td class="num">${f1((q.n_describe || 0) / d.n)}</td></tr>`; }).join('');
    if (HB.two_models.length) {
      $('lb2').innerHTML = '<tr><th>Model</th><th class="num">Score</th><th class="num">95% CI</th><th class="num">Solved</th><th class="num">$/ep</th></tr>' +
        HB.two_models.map((d) => `<tr><td><i class="sw" style="background:${fam(d.model).c}"></i>${d.name}</td><td class="num"><b>${f1(d.mean)}</b></td><td class="num">${f1(d.ci[0])}–${f1(d.ci[1])}</td><td class="num">${d.solved}/${d.n}</td><td class="num">${money(d.cost)}</td></tr>`).join('') +
        HB.two_refs.map((d) => `<tr class="ref"><td>${d.name}</td><td class="num">${f1(d.mean)}</td><td class="num">${f1(d.ci[0])}–${f1(d.ci[1])}</td><td class="num">${d.solved}/${d.n}</td><td></td></tr>`).join('');
    }
    const v = HB.scorer;
    if (v) $('scorerval').innerHTML = `<b>Is the scorer reliable?</b> When every hasta's shape is scored against every other hasta's target (at the target's orientation), ${v.diag_success}/${v.n} self-matches succeed and ${v.offdiag_false_success} of ${v.n * (v.n - 1)} cross-matches falsely succeed. ` +
      `Under Gaussian joint noise on the true target, the success rate is ${Object.entries(v.noise_success).map(([k, x]) => `${Math.round(100 * x)}% at σ = ${k.replace('deg', '°')}`).join(', ')}. The success test is therefore strict: roughly ±4° per joint.`;
  })();

  // ------------------------------------------------------------ effort table
  (function effort() {
    const byBase = {}; models.forEach((d) => (byBase[d.base] = byBase[d.base] || {})[d.effort] = d);
    const rows = Object.values(byBase).filter((g) => g.default && g.high);
    if (!rows.length) { const e = $('effortsec'); if (e) e.style.display = 'none'; return; }
    $('effort').innerHTML = '<tr><th>Model</th><th class="num">Default</th><th class="num">High</th><th class="num">Gain</th><th class="num">Hastas better at high</th><th class="num">Solved (default → high)</th><th class="num">$/ep (default → high)</th><th class="num">Calls (default → high)</th></tr>' +
      rows.map((g) => { const a = g.default, b = g.high; const common = Object.keys(a.per_target).filter((t) => t in b.per_target);
        const better = common.filter((t) => b.per_target[t] > a.per_target[t]).length;
        return `<tr><td>${a.name}</td><td class="num">${f1(a.mean)}</td><td class="num">${f1(b.mean)}</td><td class="num"><b>${(b.mean - a.mean >= 0 ? '+' : '') + f1(b.mean - a.mean)}</b></td>
        <td class="num">${better}/${common.length}</td><td class="num">${a.solved} → ${b.solved}</td><td class="num">${money(a.cost)} → ${money(b.cost)}</td><td class="num">${f1(a.calls)} → ${f1(b.calls)}</td></tr>`; }).join('');
  })();

  // ------------------------------------------------------------ gallery
  (function gallery() {
    $('gallery').innerHTML = (HB.all28 || HB.order).map((t) =>
      `<div class="gcard"><img loading="lazy" src="img/target/${t}_front.jpg"><div class="c"><div class="n">${t[0].toUpperCase() + t.slice(1)}${T[t].core8 ? ' ·<span style="font-weight:400"> core-8</span>' : ''}</div><div class="m">${T[t].meaning}</div>${T[t].desc.replace(/^[^:]*:\s*/, '')}</div></div>`).join('');
  })();

  // ------------------------------------------------------------ 3-D hand (FK + linear-blend skinning)
  function Hand(data, scene) {
    const nb = data.bodies.length;
    const tmpQ = new THREE.Quaternion(), tmpV = new THREE.Vector3(), axis = new THREE.Vector3();
    const local = data.bodies.map((b) => new THREE.Matrix4().compose(new THREE.Vector3(...b.pos), new THREE.Quaternion(b.quat[1], b.quat[2], b.quat[3], b.quat[0]), new THREE.Vector3(1, 1, 1)));
    const world = data.bodies.map(() => new THREE.Matrix4());
    const J = new THREE.Matrix4();
    function fk(q) {
      for (let i = 0; i < nb; i++) {
        const b = data.bodies[i];
        world[i].copy(b.parent < 0 ? new THREE.Matrix4() : world[b.parent]).multiply(local[i]);
        for (const j of b.joints) {
          axis.set(...j.axis);
          if (j.type === 'slide') J.makeTranslation(axis.x * q[j.q], axis.y * q[j.q], axis.z * q[j.q]);
          else J.makeRotationAxis(axis.normalize(), q[j.q]);
          world[i].multiply(J);
        }
      }
    }
    fk(new Array(data.nq).fill(0));
    const bindInv = world.map((m) => m.clone().invert());
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...data.skin_rgb), roughness: 0.62, metalness: 0.0 });
    const skins = data.skins.map((s) => {
      const g = new THREE.BufferGeometry();
      const bind = new Float32Array(s.vert);
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(s.vert), 3));
      g.setIndex(s.face);
      const mesh = new THREE.Mesh(g, mat); scene.add(mesh);
      return { s, g, bind, mesh };
    });
    const meshes = data.meshes.map((mm) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(mm.vert), 3)); g.setIndex(mm.face); g.computeVertexNormals();
      const mesh = new THREE.Mesh(g, mat); mesh.matrixAutoUpdate = false; scene.add(mesh);
      const off = new THREE.Matrix4().compose(new THREE.Vector3(...mm.pos), new THREE.Quaternion(mm.quat[1], mm.quat[2], mm.quat[3], mm.quat[0]), new THREE.Vector3(1, 1, 1));
      return { mm, mesh, off };
    });
    const M = new THREE.Matrix4(), v = new THREE.Vector3();
    function pose(q) {
      fk(q);
      for (const sk of skins) {
        const pos = sk.g.attributes.position.array; pos.fill(0);
        for (const bone of sk.s.bones) {
          M.multiplyMatrices(world[bone.body], bindInv[bone.body]);
          const e = M.elements;
          for (let k = 0; k < bone.v.length; k++) {
            const vi = bone.v[k] * 3, w = bone.w[k];
            const x = sk.bind[vi], y = sk.bind[vi + 1], z = sk.bind[vi + 2];
            pos[vi] += w * (e[0] * x + e[4] * y + e[8] * z + e[12]);
            pos[vi + 1] += w * (e[1] * x + e[5] * y + e[9] * z + e[13]);
            pos[vi + 2] += w * (e[2] * x + e[6] * y + e[10] * z + e[14]);
          }
        }
        sk.g.attributes.position.needsUpdate = true; sk.g.computeVertexNormals();
      }
      for (const m of meshes) m.mesh.matrix.multiplyMatrices(world[m.mm.body], m.off);
    }
    function dispose() { skins.forEach((s) => { scene.remove(s.mesh); s.g.dispose(); }); meshes.forEach((m) => { scene.remove(m.mesh); m.mesh.geometry.dispose(); }); }
    return { pose, dispose, data };
  }

  // ------------------------------------------------------------ replay explorer
  const stage = $('stage');
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  stage.insertBefore(renderer.domElement, stage.firstChild);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#c3c9d2');
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(0.3, -0.6, 0.7); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.3); fill.position.set(-0.5, -0.3, 0.2); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xffffff, 0.35); rim.position.set(0.1, 0.7, 0.4); scene.add(rim);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 10); camera.up.set(0, 0, 1);
  const controls = new THREE.OrbitControls(camera, renderer.domElement); controls.enableDamping = true;
  let hand = null, handKey = null, curCam = 'front';
  function setCam(name) {
    const c = hand.data.cams[name]; if (!c) return; curCam = name;
    camera.position.set(...c.pos);
    camera.quaternion.set(c.quat[1], c.quat[2], c.quat[3], c.quat[0]);
    camera.fov = c.fovy; camera.updateProjectionMatrix();
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion); camera.up.copy(up);
    const d = camera.position.length();
    controls.target.copy(camera.position).add(fwd.multiplyScalar(d)); controls.update();
    document.querySelectorAll('button.cam').forEach((b) => b.classList.toggle('on', b.dataset.cam === name));
  }
  document.querySelectorAll('button.cam').forEach((b) => (b.onclick = () => setCam(b.dataset.cam)));
  function resize() { const w = stage.clientWidth, hh = stage.clientHeight; renderer.setSize(w, hh, false); camera.aspect = w / hh; camera.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(stage);
  (function loop() { requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); })();

  const allEntries = models.concat(HB.two_models);
  const selM = $('selModel'), selT = $('selTarget');
  models.forEach((d) => selM.appendChild(h('option', { value: d.model }, d.name)));
  const targetList = HB.order.concat(HB.order_two);
  targetList.forEach((t) => selT.appendChild(h('option', { value: t }, t + (T[t].two ? ' (two hands)' : ''))));
  function entryFor(model, t) { return allEntries.find((d) => d.model === model && d.split === (T[t].two ? 'two' : 'one')); }
  function loadEp(k, cb) {
    const g = 'HB_EP_' + k; if (window[g]) return cb(window[g]);
    const s = document.createElement('script'); s.src = `data/ep/${k}.js`; s.onload = () => cb(window[g]); document.body.appendChild(s);
  }
  let ep = null, idx = 0, timer = null;
  window.openReplay = openReplay;
  function openReplay(k, t) {
    const d = allEntries.find((e) => e.key === k); if (!d) return;
    selM.value = d.model; selT.value = t;
    loadEp(k, (E) => {
      ep = E[t]; if (!ep) { $('epinfo').textContent = 'No episode recorded for this pair.'; return; }
      const two = T[t].two, hk = two ? 'two' : 'one';
      if (handKey !== hk) { if (hand) hand.dispose(); hand = Hand(two ? window.HB_HAND2 : window.HB_HAND1, scene); handKey = hk; setCam(curCam); }
      $('tgimg').src = `img/target/${t}_${['front', 'top'].includes(curCam) || two ? (curCam === 'top' ? 'top' : 'front') : curCam === 'side' ? 'side' : curCam === 'three_quarter' ? 'three_quarter' : 'front'}.jpg`;
      $('epinfo').innerHTML = `<b>${d.name}</b> on <b>${t}</b> (${T[t].meaning}) · final score <b>${f1(ep.score)}</b>${ep.solved ? ' · <span style="color:var(--ok)">solved</span>' : ''} · best ${f1(ep.best)} · ${ep.calls} calls, ${ep.looks} looks, ${ep.reads} reads · ${money(ep.cost)}` +
        (ep.plan ? `<br><span style="color:var(--ink-2)">plan: “${ep.plan.slice(0, 260).replace(/</g, '&lt;')}${ep.plan.length > 260 ? '…' : ''}”</span>` : '');
      const sl = $('slider'); sl.max = ep.steps.length - 1;
      $('steplog').innerHTML = ep.steps.map((s, i) => `<div data-i="${i}"><span class="t">${i}. ${s.t}</span> ${s.a && s.a !== '{}' ? '<span class="mono">' + s.a.replace(/</g, '&lt;').slice(0, 160) + '</span>' : ''}<br><span class="${s.ok ? '' : 'r'}">${(s.m || '').replace(/</g, '&lt;')}</span> <span style="float:right;color:var(--ink-2)">${s.s == null ? '' : f1(s.s)}</span></div>`).join('');
      $('steplog').querySelectorAll('div').forEach((dv) => (dv.onclick = () => go(+dv.dataset.i)));
      drawTrace(); go(ep.steps.length - 1);
    });
  }
  selM.onchange = selT.onchange = () => { const d = entryFor(selM.value, selT.value); if (d) openReplay(d.key, selT.value); else $('epinfo').textContent = 'No episode for this pair.'; };
  function go(i) {
    if (!ep) return; idx = Math.max(0, Math.min(ep.steps.length - 1, i));
    const st = ep.steps[idx]; hand.pose(st.q); $('slider').value = idx;
    $('ov').textContent = `call ${idx}/${ep.steps.length - 1} · ${st.t} · hidden score ${st.s == null ? '–' : f1(st.s)}`;
    $('steplog').querySelectorAll('div').forEach((dv) => dv.classList.toggle('cur', +dv.dataset.i === idx));
    const cur = $('steplog').querySelector('div.cur'); if (cur) cur.scrollIntoView({ block: 'nearest' });
    const tr = (ep.transcript || []).filter((x) => x.step <= idx).pop();
    $('thought').textContent = tr ? (tr.error ? 'API error: ' + tr.error : tr.text || '(no text this turn)') : '';
    const mk = document.getElementById('trace-cur'); if (mk) { const X = mk._X; mk.setAttribute('x1', X(idx)); mk.setAttribute('x2', X(idx)); }
  }
  $('slider').oninput = (e) => go(+e.target.value);
  $('play').onclick = () => {
    if (timer) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; }
    if (idx >= ep.steps.length - 1) go(0);
    $('play').textContent = '❚❚ pause';
    timer = setInterval(() => { if (idx >= ep.steps.length - 1) { clearInterval(timer); timer = null; $('play').textContent = '▶ play'; return; } go(idx + 1); }, Math.max(120, 14000 / ep.steps.length));
  };
  function drawTrace() {
    const W = 640, H = 120, m = { l: 30, r: 8, t: 8, b: 26 };
    const n = ep.steps.length - 1;
    const X = (i) => m.l + (n ? i / n : 0) * (W - m.l - m.r), Y = (s) => m.t + (1 - s / 100) * (H - m.t - m.b);
    const svg = sv('svg', { viewBox: `0 0 ${W} ${H}` });
    [0, 50, 100].forEach((s) => { sv('line', { x1: m.l, x2: W - m.r, y1: Y(s), y2: Y(s), stroke: GRID }, svg); sv('text', { x: m.l - 5, y: Y(s) + 4, 'text-anchor': 'end', fill: INK2, 'font-size': 10 }, svg).textContent = s; });
    const pts = ep.steps.map((s, i) => [i, s.s]).filter((p) => p[1] != null);
    let best = -1; const bp = pts.map(([i, s]) => { best = Math.max(best, s); return [i, best]; });
    sv('path', { d: bp.map(([i, s], k) => (k ? 'L' : 'M') + X(i) + ' ' + Y(s)).join(''), fill: 'none', stroke: css('--s1'), 'stroke-width': 1.5, 'stroke-dasharray': '5 3', opacity: 0.8 }, svg);
    sv('path', { d: pts.map(([i, s], k) => (k ? 'L' : 'M') + X(i) + ' ' + Y(s)).join(''), fill: 'none', stroke: css('--s1'), 'stroke-width': 2 }, svg);
    ep.steps.forEach((s, i) => {
      const c = !s.ok ? css('--bad') : s.t === 'look' ? INK3 : s.t === 'read_description' ? css('--s4') : s.t === 'plan' ? INK : ['set_fingers', 'turn_hand', 'move_hand', 'undo'].includes(s.t) ? css('--s1') : null;
      if (c) sv('rect', { x: X(i) - 2, y: H - m.b + 6, width: 4, height: 10, fill: c, rx: 1 }, svg);
    });
    const cur = sv('line', { id: 'trace-cur', y1: m.t, y2: H - m.b + 16, stroke: INK, 'stroke-width': 1 }, svg); cur._X = X;
    const hit = sv('rect', { x: m.l, y: 0, width: W - m.l - m.r, height: H, fill: 'transparent', style: 'cursor:pointer' }, svg);
    hit.addEventListener('click', (e) => { const r = svg.getBoundingClientRect(); go(Math.round((((e.clientX - r.left) / r.width) * W - m.l) / (W - m.l - m.r) * n)); });
    $('trace').innerHTML = ''; $('trace').appendChild(svg);
  }
  // open on the best single-hand episode of the top model
  const top = models[0];
  const bestT = Object.entries(top.per_target).sort((a, b) => b[1] - a[1])[0][0];
  openReplay(top.key, bestT);
})();
