/* HastaBench film: GSAP timeline for 2-D motion + deterministic three.js frame sources for the hand. */
(function () {
  const D = window.HV, H = window.HB_HAND1;
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
  const cap = (s) => s[0].toUpperCase() + s.slice(1);
  const fam = (m) => /claude/i.test(m) ? '#5b9bf0' : /gpt/i.test(m) ? '#f08a4b' : /gemini/i.test(m) ? '#46c486' : '#f0c040';

  // ---------------------------------------------------------------- build DOM
  const mosaic = $('#mosaic');
  D.all28.forEach((t) => { const i = el('img'); i.src = `assets/img/m_${t}.jpg`; mosaic.appendChild(i); });

  const board = $('#board');
  const X = (s) => (s / 100) * 1250;       // px per score point inside the track
  D.board.forEach((b, k) => {
    const r = el('div', 'row');
    r.appendChild(el('div', 'nm', b.name));
    const tr = el('div', 'track');
    const bar = el('div', 'bar'); bar.style.width = X(b.mean) + 'px'; bar.style.background = fam(b.model); bar.id = 'bar' + k;
    const val = el('div', 'val', b.mean.toFixed(1)); val.style.left = X(b.mean) + 14 + 'px'; val.id = 'val' + k;
    const sv = el('div', 'sv', b.solved ? `${b.solved} solved` : ''); sv.style.left = X(b.mean) + 86 + 'px'; sv.id = 'sv' + k;
    tr.append(bar, val, sv); r.appendChild(tr); board.appendChild(r);
  });
  const refs = [['REF_cma_score', 'blind search on the hidden score'], ['REF_mean_shape', 'average-hasta prior'], ['REF_rest', 'hands at rest']];
  refs.forEach(([k, lab], i) => {
    const l = el('div', 'refl'); l.style.left = 448 + X(D.refs[k]) + 'px'; l.style.top = '-30px'; l.style.height = (D.board.length * 54 + 40) + 'px';
    l.id = 'ref' + i; l.innerHTML = `<span style="bottom:${-40 - 26 * (i % 2)}px">${lab} · ${D.refs[k]}</span>`; board.appendChild(l);
  });

  const fails = $('#fails');
  D.fails.forEach((f, i) => {
    const c = el('div', 'fc'); c.id = 'fc' + i;
    c.innerHTML = `<div class="imgs"><img src="assets/img/${f.timg}.jpg"><img src="assets/img/${f.fimg}.jpg"></div>
      <div class="il"><span>target</span><span>attempt</span></div>
      <div class="nm">${cap(f.target)} <span class="mute" style="font-size:22px;font-style:italic">${f.meaning}</span></div>
      <div class="md">${f.model}</div><div class="sc">score ${f.score}</div><div class="nt">${f.note}</div>`;
    fails.appendChild(c);
  });

  const pairs = $('#pairs');
  D.two.forEach((p, i) => {
    const c = el('div', 'pair'); c.id = 'pr' + i;
    c.innerHTML = `<div class="imgs"><img src="assets/img/${p.timg}.jpg"><img src="assets/img/${p.fimg}.jpg"></div>
      <div class="il"><span>target</span><span>best model (${p.model})</span></div>
      <div class="nm">${cap(p.target)} <span class="mute" style="font-size:24px;font-style:italic">${p.meaning}</span></div>
      <div class="md">score ${p.score}</div>`;
    pairs.appendChild(c);
  });
  $('#cred7').textContent = `${D.n_models} frontier model configurations · ${D.n_eps} episodes · MuJoCo · WebXR hand mesh (MIT) · rendered with HyperFrames`;

  // ---------------------------------------------------------------- GSAP timeline (absolute times)
  const tl = gsap.timeline({ paused: true });
  const fadeIn = (s, t, d = 0.8, y = 30) => tl.fromTo(s, { opacity: 0, y }, { opacity: 1, y: 0, duration: d, ease: 'power2.out' }, t);
  const fadeOut = (s, t, d = 0.6) => tl.to(s, { opacity: 0, duration: d, ease: 'power1.in' }, t);
  // scene 1
  fadeIn('#l1', 0.5, 1.0); fadeIn('#l2', 2.2, 1.0); fadeOut(['#l1', '#l2', '#poseName'], 7.3);
  // scene 2
  tl.fromTo('#mosaic img', { opacity: 0, scale: 0.92 }, { opacity: 0.85, scale: 1, duration: 0.5, stagger: 0.04, ease: 'power2.out' }, 8.1);
  tl.fromTo('#veil', { opacity: 0 }, { opacity: 1, duration: 0.8 }, 9.6);
  fadeIn('#t2', 10.0, 1.0, 40); fadeIn('#sub2', 10.9, 0.9);
  fadeOut(['#mosaic', '#veil', '#t2', '#sub2'], 13.4);
  // scene 3
  fadeIn('#s3 .hdr', 14.2); fadeIn('#tgtCard', 14.6); fadeIn('#epStage', 14.8); fadeIn('#side', 15.0); fadeIn('#foot3', 16.0);
  fadeOut(['#s3 .hdr', '#tgtCard', '#epStage', '#side', '#foot3'], 27.4);
  // scene 4
  fadeIn('#s4 .hdr', 28.2);
  D.board.forEach((b, k) => {
    tl.fromTo('#bar' + k, { scaleX: 0 }, { scaleX: 1, duration: 0.9, ease: 'power3.out' }, 28.8 + k * 0.12);
    tl.fromTo(['#val' + k, '#sv' + k], { opacity: 0 }, { opacity: 1, duration: 0.4 }, 29.4 + k * 0.12);
  });
  refs.forEach((_, i) => tl.fromTo('#ref' + i, { opacity: 0 }, { opacity: 1, duration: 0.6 }, 31.0 + i * 0.4));
  fadeIn('#stat4', 32.6, 0.9);
  fadeOut(['#s4 .hdr', '#board', '#stat4'], 39.4);
  // scene 5
  fadeIn('#s5 .hdr', 40.2);
  D.fails.forEach((_, i) => fadeIn('#fc' + i, 40.8 + i * 0.7, 0.8));
  tl.to('#fails', { opacity: 0.25, duration: 0.6 }, 46.6);
  fadeIn('#thumbStat', 46.9, 0.9);
  fadeOut(['#s5 .hdr', '#fails', '#thumbStat'], 50.4);
  // scene 6
  fadeIn('#s6 .hdr', 51.2); D.two.forEach((_, i) => fadeIn('#pr' + i, 51.7 + i * 0.7, 0.8)); fadeIn('#stat6', 54.0);
  fadeOut(['#s6 .hdr', '#pairs', '#stat6'], 57.4);
  // scene 7
  fadeIn('#t7', 58.3, 1.1, 40); fadeIn('#tag7', 59.2); fadeIn('#links7', 60.0); fadeIn('#cred7', 60.8);
  fadeOut(['#t7', '#tag7', '#links7', '#cred7'], 65.2, 0.8);
  window.__timelines = window.__timelines || {};
  window.__timelines.root = tl;

  // ---------------------------------------------------------------- hand (FK + linear-blend skinning)
  function Hand(data, scene) {
    const nb = data.bodies.length, axis = new THREE.Vector3(), J = new THREE.Matrix4();
    const local = data.bodies.map((b) => new THREE.Matrix4().compose(new THREE.Vector3(...b.pos), new THREE.Quaternion(b.quat[1], b.quat[2], b.quat[3], b.quat[0]), new THREE.Vector3(1, 1, 1)));
    const world = data.bodies.map(() => new THREE.Matrix4()), I = new THREE.Matrix4();
    function fk(q) {
      for (let i = 0; i < nb; i++) {
        const b = data.bodies[i];
        world[i].copy(b.parent < 0 ? I : world[b.parent]).multiply(local[i]);
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
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(0.86, 0.66, 0.54), roughness: 0.55, metalness: 0.0 });
    const skins = data.skins.map((s) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(s.vert), 3)); g.setIndex(s.face);
      const mesh = new THREE.Mesh(g, mat); scene.add(mesh);
      return { s, g, bind: new Float32Array(s.vert) };
    });
    const meshes = data.meshes.map((mm) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(mm.vert), 3)); g.setIndex(mm.face); g.computeVertexNormals();
      const mesh = new THREE.Mesh(g, mat); mesh.matrixAutoUpdate = false; scene.add(mesh);
      return { mm, mesh, off: new THREE.Matrix4().compose(new THREE.Vector3(...mm.pos), new THREE.Quaternion(mm.quat[1], mm.quat[2], mm.quat[3], mm.quat[0]), new THREE.Vector3(1, 1, 1)) };
    });
    const M = new THREE.Matrix4();
    function pose(q) {
      fk(q);
      for (const sk of skins) {
        const pos = sk.g.attributes.position.array; pos.fill(0);
        for (const bone of sk.s.bones) {
          M.multiplyMatrices(world[bone.body], bindInv[bone.body]); const e = M.elements;
          for (let k = 0; k < bone.v.length; k++) {
            const vi = bone.v[k] * 3, w = bone.w[k], x = sk.bind[vi], y = sk.bind[vi + 1], z = sk.bind[vi + 2];
            pos[vi] += w * (e[0] * x + e[4] * y + e[8] * z + e[12]);
            pos[vi + 1] += w * (e[1] * x + e[5] * y + e[9] * z + e[13]);
            pos[vi + 2] += w * (e[2] * x + e[6] * y + e[10] * z + e[14]);
          }
        }
        sk.g.attributes.position.needsUpdate = true; sk.g.computeVertexNormals();
      }
      for (const m of meshes) m.mesh.matrix.multiplyMatrices(world[m.mm.body], m.off);
    }
    return { pose };
  }

  function stage(canvas, w, h) {
    const r = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
    r.setPixelRatio(1); r.setSize(w, h, false); r.setClearColor(0x000000, 0);
    r.outputEncoding = THREE.sRGBEncoding;
    const sc = new THREE.Scene();
    sc.add(new THREE.HemisphereLight(0xfff4e6, 0x3a3029, 0.85));
    const key = new THREE.DirectionalLight(0xfff1dd, 0.95); key.position.set(0.35, -0.6, 0.7); sc.add(key);
    const rim = new THREE.DirectionalLight(0xffc27a, 0.75); rim.position.set(-0.2, 0.8, 0.35); sc.add(rim);
    const fill = new THREE.DirectionalLight(0xbfd2ff, 0.25); fill.position.set(-0.6, -0.3, 0.1); sc.add(fill);
    const cam = new THREE.PerspectiveCamera(30, w / h, 0.01, 10); cam.up.set(0, 0, 1);
    return { r, sc, cam, hand: Hand(H, sc) };
  }
  const ease = (x) => (x <= 0 ? 0 : x >= 1 ? 1 : x * x * (3 - 2 * x));
  const lerpQ = (a, b, u) => a.map((v, i) => v + (b[i] - v) * u);
  function orbit(cam, t, dist, yaw0, amp, period, z = 0.075) {
    const yaw = yaw0 + amp * Math.sin((2 * Math.PI * t) / period);
    cam.position.set(dist * Math.sin(yaw), -dist * Math.cos(yaw), z + 0.03);
    cam.lookAt(0, 0, z);
  }

  // ---- frame source 1: hook morph through hastas
  const S1 = stage($('#cv1'), 960, 1080);
  const M = D.morph, HOLD = 1.25, MOVE = 0.55;
  const poseName = $('#poseName');
  window.__hyperframes.registerFrameSource({
    element: $('#hook3d'), ready: Promise.resolve(),
    render: (t) => {
      const seg = HOLD + MOVE; let i = Math.min(M.length - 1, Math.floor(t / seg)); const lt = t - i * seg;
      let q = M[i].q;
      if (lt > HOLD && i < M.length - 1) q = lerpQ(M[i].q, M[i + 1].q, ease((lt - HOLD) / MOVE));
      S1.hand.pose(q);
      orbit(S1.cam, t, 0.62, 0, 0.32, 9);
      S1.r.render(S1.sc, S1.cam);
      const cur = M[lt > HOLD + MOVE * 0.5 && i < M.length - 1 ? i + 1 : i];
      poseName.innerHTML = cur.name === 'rest' ? '' : `${cur.name}<small>${cur.meaning}</small>`;
      const fadeT = Math.min(1, Math.max(0, (7.3 - t) / 0.6));
      $('#cv1').style.opacity = String(Math.min(1, t / 0.6) * fadeT);
    },
  });

  // ---- frame source 2: episode replay with score + ticker
  const S2 = stage($('#cv2'), 760, 760);
  const E = D.episode, steps = E.steps, N = steps.length - 1, T0 = 1.2, TR = 10.6;   // replay window inside the 14 s scene
  const spark = $('#spark'), ticker = $('#ticker'), scoreBig = $('#scoreBig'), solved = $('#solvedTag'), lab = $('#epLab');
  const fmt = (s) => {
    let a = {}; try { a = JSON.parse(s.a || '{}'); } catch (e) {}
    const args = a.angles ? Object.entries(a.angles).slice(0, 4).map(([k, v]) => `${k}=${Math.round(v)}`).join(' ') + (Object.keys(a.angles).length > 4 ? ' …' : '')
      : a.camera ? a.camera : a.text ? '"' + a.text.slice(0, 34) + '…"' : ['roll', 'pitch', 'yaw', 'dx', 'dy', 'dz'].filter((k) => a[k]).map((k) => `${k}=${Math.round(a[k])}`).join(' ');
    return `<span class="${s.ok ? 'tool' : 'ref'}">${s.t}</span> ${args}`;
  };
  window.__hyperframes.registerFrameSource({
    element: $('#ep3d'), ready: Promise.resolve(),
    render: (t) => {
      const u = Math.max(0, Math.min(1, (t - T0) / TR)) * N; const i = Math.floor(u), f = u - i;
      const q = i >= N ? steps[N].q : lerpQ(steps[i].q, steps[i + 1].q, ease(f / 0.6));
      S2.hand.pose(q); orbit(S2.cam, t, 0.58, 0.15, 0.18, 14); S2.r.render(S2.sc, S2.cam);
      const fadeT = Math.min(1, Math.max(0, (13.4 - t) / 0.6)); $('#cv2').style.opacity = String(Math.min(1, Math.max(0, (t - 0.8) / 0.6)) * fadeT);
      const k = Math.min(N, Math.round(u));
      const sc = steps[k].s || 0; scoreBig.textContent = sc.toFixed(1);
      scoreBig.style.color = sc >= 90 ? '#46c486' : sc >= 60 ? '#f0a020' : '#f3ede4';
      solved.style.opacity = t > T0 + TR + 0.2 ? '1' : '0';
      lab.textContent = `${E.model} · call ${k} / ${N}`;
      const lines = []; for (let j = Math.max(1, k - 6); j <= k; j++) lines.push(`<div class="${j === k ? 'cur' : ''}">${fmt(steps[j])}</div>`);
      ticker.innerHTML = lines.join('');
      const W = 410, Hh = 120, pts = steps.slice(0, k + 1).map((s, j) => `${(j / N) * W},${Hh - ((s.s || 0) / 100) * Hh}`);
      spark.innerHTML = `<line x1="0" x2="${W}" y1="${Hh - 0.9 * Hh}" y2="${Hh - 0.9 * Hh}" stroke="#3a342e" stroke-dasharray="4 4"/>` +
        `<polyline points="${pts.join(' ')}" fill="none" stroke="#f0a020" stroke-width="3"/>`;
    },
  });
})();
