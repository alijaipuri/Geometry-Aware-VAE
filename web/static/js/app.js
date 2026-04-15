// ── State ──────────────────────────────────────────────────────────
const state = {
  chatHistory: [],
  pollingInterval: null,
};

// ── Navigation ─────────────────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(item => {
  item.addEventListener('click', () => {
    const tab = item.dataset.tab;
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    item.classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
    document.getElementById('pageTitle').textContent =
      item.querySelector('span:last-child').textContent;
  });
});

// ── Helpers ────────────────────────────────────────────────────────
function getModel() { return document.getElementById('globalModel').value; }
function getDataset() { return document.getElementById('globalDataset').value; }

function showStatus(id, msg, type='info') {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `status-box ${type}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

async function api(endpoint, method='GET', body=null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(endpoint, opts);
  return r.json();
}

// ── Dashboard ──────────────────────────────────────────────────────
async function loadDashboard() {
  const status = await api('/api/status');
  const badges = {
    gaussian_mnist:      ['valGM', 'badgeGM'],
    spherical_mnist:     ['valSM', 'badgeSM'],
    gaussian_fashion_mnist:  ['valGF', 'badgeGF'],
    spherical_fashion_mnist: ['valSF', 'badgeSF'],
  };
  for (const [key, [valId, badgeId]] of Object.entries(badges)) {
    const d = status[key];
    if (!d) continue;
    document.getElementById(valId).textContent =
      d.best_val != null ? d.best_val.toFixed(4) : '—';
    const badge = document.getElementById(badgeId);
    if (d.trained) {
      badge.textContent = '✓ Trained';
      badge.className = 'stat-badge trained';
    } else if (d.training) {
      badge.textContent = '⚡ Training...';
      badge.className = 'stat-badge';
      badge.style.color = '#7c6fff';
    } else {
      badge.textContent = 'Not trained';
      badge.className = 'stat-badge';
    }
  }

  // Load plots
  const model = getModel(), ds = getDataset();
  const lossEl = document.getElementById('lossPlot');
  const tsneEl = document.getElementById('tsnePlot');
  lossEl.src = `/api/plot/${model}_${ds}_loss.png?t=${Date.now()}`;
  tsneEl.src = `/api/plot/${model}_${ds}_tsne.png?t=${Date.now()}`;
  lossEl.onerror = () => { lossEl.alt = 'Train model to see loss curves'; };
  tsneEl.onerror = () => { tsneEl.alt = 'Train model to see t-SNE'; };
}

async function loadReconstructions() {
  const data = await api('/api/reconstruct', 'POST', {
    model_type: getModel(), dataset: getDataset(), n: 8
  });
  if (data.error) { console.error(data.error); return; }

  const container = document.getElementById('reconDisplay');
  container.innerHTML = '';

  const origRow = document.createElement('div');
  origRow.className = 'recon-row';
  origRow.innerHTML = '<div class="recon-label">Originals</div>';
  data.originals.forEach(b64 => {
    const img = document.createElement('img');
    img.className = 'recon-img';
    img.src = `data:image/png;base64,${b64}`;
    origRow.appendChild(img);
  });

  const reconRow = document.createElement('div');
  reconRow.className = 'recon-row';
  reconRow.innerHTML = '<div class="recon-label">Reconstructed</div>';
  data.reconstructions.forEach(b64 => {
    const img = document.createElement('img');
    img.className = 'recon-img';
    img.src = `data:image/png;base64,${b64}`;
    reconRow.appendChild(img);
  });

  container.appendChild(origRow);
  container.appendChild(reconRow);
}

// ── Training ───────────────────────────────────────────────────────
async function startTraining() {
  const model = document.getElementById('trainModel').value;
  const dataset = document.getElementById('trainDataset').value;
  const data = await api('/api/train', 'POST', { model_type: model, dataset });

  if (data.error) {
    showStatus('trainStatus', data.error, 'error');
    return;
  }
  showStatus('trainStatus', data.message, 'info');

  const prog = document.getElementById('trainProgress');
  prog.classList.remove('hidden');

  // Simulate progress polling
  let pct = 0;
  const fill = document.getElementById('progressFill');
  const txt = document.getElementById('progressText');
  const interval = setInterval(async () => {
    const st = await api('/api/status');
    const key = `${model}_${dataset}`;
    if (st[key] && st[key].training) {
      pct = Math.min(pct + Math.random() * 4, 95);
      fill.style.width = pct + '%';
      txt.textContent = `Training ${model} on ${dataset}... ${pct.toFixed(0)}%`;
    } else {
      fill.style.width = '100%';
      txt.textContent = 'Training complete!';
      showStatus('trainStatus', 'Training finished successfully!', 'success');
      clearInterval(interval);
      setTimeout(() => prog.classList.add('hidden'), 3000);
      loadDashboard();
    }
  }, 3000);
}

async function trainAllModels() {
  const datasets = ['mnist', 'fashion_mnist'];
  const models = ['gaussian', 'spherical'];
  const queueEl = document.getElementById('queueStatus');
  queueEl.innerHTML = '';

  for (const ds of datasets) {
    for (const mt of models) {
      const item = document.createElement('div');
      item.className = 'queue-item';
      item.id = `q_${mt}_${ds}`;
      item.innerHTML = `<div class="queue-dot running" id="dot_${mt}_${ds}"></div>
        <span>${mt} on ${ds}</span><span id="status_${mt}_${ds}">Queued</span>`;
      queueEl.appendChild(item);
    }
  }

  for (const ds of datasets) {
    for (const mt of models) {
      document.getElementById(`status_${mt}_${ds}`).textContent = 'Training...';
      await api('/api/train', 'POST', { model_type: mt, dataset: ds });
      await new Promise(resolve => {
        const check = setInterval(async () => {
          const st = await api('/api/status');
          if (!st[`${mt}_${ds}`]?.training) {
            clearInterval(check);
            document.getElementById(`dot_${mt}_${ds}`).className = 'queue-dot done';
            document.getElementById(`status_${mt}_${ds}`).textContent = '✓ Done';
            resolve();
          }
        }, 4000);
      });
    }
  }
}

// ── Latent Space ───────────────────────────────────────────────────
async function loadLatentSpace() {
  const data = await api('/api/latent_space', 'POST', {
    model_type: getModel(), dataset: getDataset()
  });

  if (data.error) { alert(data.error); return; }

  const colors = [
    '#7c6fff','#ff6b9d','#00d4aa','#ffd700','#ff9500',
    '#00bfff','#ff4757','#2ed573','#a29bfe','#fd9644'
  ];

  const traces = [];
  const grouped = {};
  data.points.forEach(p => {
    if (!grouped[p.label]) grouped[p.label] = { x: [], y: [] };
    grouped[p.label].x.push(p.x);
    grouped[p.label].y.push(p.y);
  });

  for (const [label, pts] of Object.entries(grouped)) {
    traces.push({
      x: pts.x, y: pts.y,
      mode: 'markers',
      type: 'scatter',
      name: `Class ${label}`,
      marker: { color: colors[parseInt(label) % colors.length], size: 5, opacity: 0.7 }
    });
  }

  Plotly.newPlot('latentPlot', traces, {
    paper_bgcolor: '#1a1a25',
    plot_bgcolor: '#16161f',
    font: { color: '#e8e8f0' },
    legend: { bgcolor: '#1a1a25', bordercolor: '#2a2a3a' },
    margin: { l: 40, r: 20, t: 20, b: 40 },
    xaxis: { gridcolor: '#2a2a3a', title: 'z₁' },
    yaxis: { gridcolor: '#2a2a3a', title: 'z₂' }
  }, { responsive: true });

  const metricDiv = document.getElementById('silhouetteDisplay');
  metricDiv.innerHTML = `
    <div class="metric-chip">Silhouette Score: <span>${data.silhouette.toFixed(4)}</span></div>
    <div class="metric-chip">Model: <span>${data.model_type}</span></div>
    <div class="metric-chip">Dataset: <span>${data.dataset}</span></div>
  `;
}

async function loadInterpolation() {
  const steps = parseInt(document.getElementById('interpSteps').value);
  const data = await api('/api/interpolate', 'POST', {
    model_type: getModel(), dataset: getDataset(), steps
  });
  if (data.error) { alert(data.error); return; }

  const container = document.getElementById('interpDisplay');
  container.innerHTML = '';
  data.images.forEach((b64, i) => {
    const img = document.createElement('img');
    img.className = 'interp-img';
    img.src = `data:image/png;base64,${b64}`;
    img.title = `Step ${i + 1}/${steps}`;
    container.appendChild(img);
  });
}

// ── Generate ───────────────────────────────────────────────────────
async function loadGenerated() {
  const n = parseInt(document.getElementById('genCount').value);
  const data = await api('/api/generate', 'POST', {
    model_type: getModel(), dataset: getDataset(), n
  });
  if (data.error) { alert(data.error); return; }

  const container = document.getElementById('genDisplay');
  container.innerHTML = '';
  data.images.forEach(b64 => {
    const img = document.createElement('img');
    img.className = 'gen-img';
    img.src = `data:image/png;base64,${b64}`;
    container.appendChild(img);
  });
}

// ── Compare ────────────────────────────────────────────────────────
async function loadComparison() {
  const ds = getDataset();
  const data = await api('/api/compare_all', 'POST', { dataset: ds });

  const wrap = document.getElementById('compareTable');
  if (!data.comparison || Object.keys(data.comparison).length === 0) {
    wrap.innerHTML = '<p style="color:var(--text2);padding:20px">Train models first to see comparison.</p>';
    return;
  }

  const allKeys = new Set();
  Object.values(data.comparison).forEach(v => Object.keys(v).forEach(k => allKeys.add(k)));
  const cols = ['Model', ...allKeys];

  let html = `<table class="compare-table"><thead><tr>`;
  cols.forEach(c => html += `<th>${c}</th>`);
  html += `</tr></thead><tbody>`;

  for (const [model, metrics] of Object.entries(data.comparison)) {
    html += `<tr><td><strong>${model}</strong></td>`;
    allKeys.forEach(k => {
      const v = metrics[k];
      html += `<td>${v !== undefined ? v : '—'}</td>`;
    });
    html += `</tr>`;
  }
  html += `</tbody></table>`;
  wrap.innerHTML = html;

  // Load side-by-side latent plots
  const gImg = document.getElementById('gaussianLatentImg');
  const sImg = document.getElementById('sphericalLatentImg');
  gImg.src = `/api/plot/gaussian_${ds}_latent.png?t=${Date.now()}`;
  sImg.src = `/api/plot/spherical_${ds}_latent.png?t=${Date.now()}`;
  gImg.onerror = () => { gImg.alt = 'Train Gaussian VAE first'; };
  sImg.onerror = () => { sImg.alt = 'Train Spherical VAE first'; };
}

// ── AI Analysis ────────────────────────────────────────────────────
async function runAIAnalysis() {
  const outputEl = document.getElementById('aiAnalysis');
  outputEl.textContent = '⟳ Analyzing your results with Llama 3.3 70B...';

  const ds = getDataset();
  const metricsData = await api('/api/metrics', 'POST', {
    model_type: getModel(), dataset: ds
  });

  const data = await api('/api/ai_analysis', 'POST', {
    metrics: metricsData,
    context: `Comparing Gaussian VAE vs Spherical VAE on dataset: ${ds}`
  });

  if (data.error) {
    outputEl.textContent = `Error: ${data.error}`;
    return;
  }
  outputEl.textContent = data.analysis;
}

async function sendChat() {
  const input = document.getElementById('chatInput');
  const msg = input.value.trim();
  if (!msg) return;

  const chatEl = document.getElementById('chatMessages');
  chatEl.innerHTML += `<div class="chat-msg user">${msg}</div>`;
  input.value = '';
  chatEl.scrollTop = chatEl.scrollHeight;

  state.chatHistory.push({ role: 'user', content: msg });

  const thinkEl = document.createElement('div');
  thinkEl.className = 'chat-msg assistant';
  thinkEl.textContent = '⟳ Thinking...';
  chatEl.appendChild(thinkEl);
  chatEl.scrollTop = chatEl.scrollHeight;

  const data = await api('/api/ai_chat', 'POST', {
    messages: state.chatHistory
  });

  thinkEl.remove();

  const reply = data.reply || data.error || 'No response.';
  state.chatHistory.push({ role: 'assistant', content: reply });

  chatEl.innerHTML += `<div class="chat-msg assistant">${reply.replace(/\n/g, '<br/>')}</div>`;
  chatEl.scrollTop = chatEl.scrollHeight;
}

// ── Init ───────────────────────────────────────────────────────────
document.getElementById('globalModel').addEventListener('change', loadDashboard);
document.getElementById('globalDataset').addEventListener('change', loadDashboard);

loadDashboard();
setInterval(loadDashboard, 15000);
