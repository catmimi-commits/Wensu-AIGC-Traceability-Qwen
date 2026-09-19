// 选项卡切换
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-page').forEach(p => {
      p.classList.remove('active');
      p.classList.add('hidden');
    });
    btn.classList.add('active');
    const target = document.getElementById('tab-' + btn.dataset.tab);
    target.classList.remove('hidden');
    target.classList.add('active');
  });
});

// 元素引用
const inputText      = document.getElementById('inputText');
const charCount      = document.getElementById('charCount');
const detectBtn      = document.getElementById('detectBtn');
const clearBtn       = document.getElementById('clearBtn');
const loading        = document.getElementById('loading');
const resultSection  = document.getElementById('resultSection');
const aiProbText     = document.getElementById('aiProbText');
const verdictBadge   = document.getElementById('verdictBadge');
const barAI          = document.getElementById('barAI');
const barHuman       = document.getElementById('barHuman');
const probAIText     = document.getElementById('probAIText');
const probHumanText  = document.getElementById('probHumanText');
const confidenceText = document.getElementById('confidenceText');
const trackingNote   = document.getElementById('trackingNote');
const trackingBars   = document.getElementById('trackingBars');
const bestModel      = document.getElementById('bestModel');
const contribBars    = document.getElementById('contribBars');
const surfaceDetail  = document.getElementById('surfaceDetail');
const toggleDetail   = document.getElementById('toggleDetail');

// 水印检测相关元素
const watermarkCard    = document.getElementById('watermarkCard');
const wmBanner         = document.getElementById('wmBanner');
const wmVerdict        = document.getElementById('wmVerdict');
const wmSub            = document.getElementById('wmSub');
const wmSourceBox      = document.getElementById('wmSourceBox');
const wmSource         = document.getElementById('wmSource');
const wmMaxZ           = document.getElementById('wmMaxZ');
const wmThreshold      = document.getElementById('wmThreshold');
const wmScoreBars      = document.getElementById('wmScoreBars');
const passiveHeader    = document.getElementById('passiveHeader');
const passiveHeaderText = document.getElementById('passiveHeaderText');
const passiveSection   = document.getElementById('passiveSection');

// Chart 实例
let gaugeChart = null, contribChart = null;

// 字数统计
inputText.addEventListener('input', () => {
  charCount.textContent = inputText.value.length + ' 字';
});

// 示例按钮
document.querySelectorAll('.sample-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const domain = btn.dataset.domain;
    try {
      const res  = await fetch('/sample/' + encodeURIComponent(domain));
      const data = await res.json();
      if (data.text) {
        inputText.value = data.text;
        charCount.textContent = data.text.length + ' 字';
      }
    } catch(e) { console.error(e); }
  });
});

// 清空
clearBtn.addEventListener('click', () => {
  inputText.value = '';
  charCount.textContent = '0 字';
  resultSection.classList.add('hidden');
});

// 展开细分
toggleDetail.addEventListener('click', () => {
  const hidden = surfaceDetail.classList.toggle('hidden');
  toggleDetail.textContent = hidden ? '▶ 点击展开12项细分' : '▼ 收起细分';
});

// 检测请求
detectBtn.addEventListener('click', async () => {
  const text = inputText.value.trim();
  if (!text) { alert('请先输入文本'); return; }

  loading.classList.remove('hidden');
  resultSection.classList.add('hidden');
  detectBtn.disabled = true;

  try {
    const res  = await fetch('/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    if (data.error) { alert('错误：' + data.error); return; }
    try {
      renderResult(data);
    } catch(e) {
      alert('渲染失败：' + e.message);
      console.error(e);
    }
  } catch(e) {
    alert('请求失败：' + e.message);
  } finally {
    loading.classList.add('hidden');
    detectBtn.disabled = false;
  }
});

// 颜色配置
const TRACK_COLORS   = ['#667eea','#f6ad55','#68d391','#fc8181','#76e4f7'];
const CONTRIB_COLORS = ['#667eea','#f6ad55','#68d391'];

// 渲染结果：先渲染第一级水印卡片，再按阶段决定是否展示第二级被动检测
function renderResult(data) {
  // 第一级：主动水印检测（始终渲染）
  renderWatermark(data.watermark);

  if (data.stage === 'watermark') {
    // 命中水印 → 直接溯源，隐藏被动检测部分
    passiveHeader.classList.add('hidden');
    passiveSection.classList.add('hidden');
    resultSection.classList.remove('hidden');
    return;
  }

  // 第二级：被动文风检测
  passiveSection.classList.remove('hidden');
  passiveHeader.classList.remove('hidden');
  const wmAvailable = data.watermark && data.watermark.available;
  passiveHeaderText.textContent = wmAvailable
    ? '未检测到明显水印，已转入被动文风检测'
    : '主动水印检测未启用，直接进行被动文风检测';

  renderPassive(data);
  resultSection.classList.remove('hidden');
}

// 第一级水印卡片渲染
function renderWatermark(wm) {
  if (!wm) { watermarkCard.classList.add('hidden'); return; }
  watermarkCard.classList.remove('hidden');
  wmThreshold.textContent = wm.threshold;

  if (!wm.available) {
    wmBanner.className = 'wm-banner wm-disabled';
    wmVerdict.textContent = '主动水印检测未启用';
    wmSub.textContent = '未安装 transformers 或缺少分词器文件，已跳过水印检测。';
    wmSourceBox.classList.add('hidden');
    wmScoreBars.innerHTML = '';
    return;
  }

  if (wm.detected) {
    wmBanner.className = 'wm-banner wm-detected';
    wmVerdict.textContent = '检测到数字水印，来源是 ' + wm.best_model;
    wmSub.textContent = '该文本由带 KGW 水印的模型生成，已直接溯源到来源模型。';
    wmSourceBox.classList.remove('hidden');
    wmSource.textContent = wm.best_model;
    wmMaxZ.textContent = Number(wm.max_z).toFixed(2);
  } else {
    wmBanner.className = 'wm-banner wm-clean';
    wmVerdict.textContent = '未检测到明显水印';
    wmSub.textContent = '最高 Z-score 低于阈值，将转入被动文风检测。';
    wmSourceBox.classList.add('hidden');
  }
  renderWmScoreBars(wm);
}

// 各候选密钥 Z-score 对比条（含阈值红线）
function renderWmScoreBars(wm) {
  wmScoreBars.innerHTML = '';
  const thr  = wm.threshold;
  const zs   = wm.scores.map(s => s.z);
  const maxZ = Math.max(thr * 1.6, ...zs, 0.1);       // 动态量程，保证阈值线可见
  const thrPct = (thr / maxZ) * 100;
  wm.scores.forEach(s => {
    const pct = Math.max(0, Math.min(100, (s.z / maxZ) * 100));   // 负 Z 截断为 0
    const hit = s.z >= thr;
    wmScoreBars.innerHTML += `
      <div class="wm-score-row ${hit ? 'hit' : ''}">
        <span class="wm-score-name">${s.model}<span class="wm-key">密钥 ${s.key}</span></span>
        <div class="wm-score-track">
          <div class="wm-score-bar" style="width:${pct}%"></div>
          <div class="wm-thr-line" style="left:${thrPct}%"></div>
        </div>
        <span class="wm-score-val">${Number(s.z).toFixed(2)}</span>
      </div>`;
  });
}

// 第二级被动检测渲染
function renderPassive(data) {
  const aiProb    = data.ai_prob;
  const humanProb = data.human_prob;
  const isAI      = data.label === 'AI';

  // 仪表盘更新
  aiProbText.textContent = aiProb.toFixed(1) + '%';
  renderGauge(aiProb);

  // 判断徽章更新
  verdictBadge.textContent = isAI ? '大概率由AI生成' : '大概率为人类撰写';
  verdictBadge.className   = 'verdict-badge ' + (isAI ? 'verdict-ai' : 'verdict-human');

  // 概率条更新
  barAI.style.width         = aiProb + '%';
  barHuman.style.width      = humanProb + '%';
  probAIText.textContent    = aiProb + '%';
  probHumanText.textContent = humanProb + '%';
  confidenceText.textContent = data.confidence + '%';

  // 溯源结果更新
  const tracking = data.tracking;
  trackingNote.textContent = isAI
    ? 'AI率较高，以下为各Qwen版本语言风格分布概率（已归一化）'
    : '当前文本人类概率更高，以下仍展示各Qwen版本风格分布供参考';

  const probs  = tracking.probs;
  const total  = Object.values(probs).reduce((a,b) => a+b, 0);
  const normed = {};
  for (const [k,v] of Object.entries(probs)) {
    normed[k] = total > 0 ? parseFloat((v/total*100).toFixed(2)) : 0;
  }
  renderTrackingBars(normed);
  bestModel.textContent = tracking.best;

  if (data.contribution) {
    renderContrib(data.contribution);
  }
  if (data.surface_detail) {
    renderSurfaceDetail(data.surface_detail);
  }

  // 重置展开状态
  surfaceDetail.classList.add('hidden');
  toggleDetail.textContent = '▶ 点击展开12项细分';
}

// 仪表盘渲染
function renderGauge(aiProb) {
  const rest  = parseFloat((100 - aiProb).toFixed(2));
  const color = aiProb >= 60 ? '#e53e3e' : aiProb >= 40 ? '#ed8936' : '#38a169';
  if (gaugeChart) gaugeChart.destroy();
  gaugeChart = new Chart(document.getElementById('gaugeChart'), {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [aiProb, rest],
        backgroundColor: [color, '#edf2f7'],
        borderWidth: 0,
        circumference: 270,
        rotation: 225
      }]
    },
    options: {
      cutout: '72%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 700 }
    }
  });
}

// 溯源进度条渲染
function renderTrackingBars(normed) {
  trackingBars.innerHTML = '';
  const sorted = Object.entries(normed).sort((a,b) => b[1]-a[1]);
  sorted.forEach(([name, prob], i) => {
    const color = TRACK_COLORS[i % TRACK_COLORS.length];
    trackingBars.innerHTML += `
      <div class="track-row">
        <span class="track-name">${name}</span>
        <div class="track-bar-bg">
          <div class="track-bar" style="width:${prob}%;background:${color}"></div>
        </div>
        <span class="track-val">${prob}%</span>
      </div>`;
  });
}

// 特征贡献度渲染
function renderContrib(contrib) {
  contribBars.innerHTML = '';
  const sorted = Object.entries(contrib).sort((a,b) => b[1]-a[1]);
  sorted.forEach(([name, val], i) => {
    const color = CONTRIB_COLORS[i % CONTRIB_COLORS.length];
    contribBars.innerHTML += `
      <div class="track-row">
        <span class="track-name">${name}</span>
        <div class="track-bar-bg">
          <div class="track-bar" style="width:${val}%;background:${color}"></div>
        </div>
        <span class="track-val">${val}%</span>
      </div>`;
  });
}

// 12项细分渲染
function renderSurfaceDetail(detail) {
  surfaceDetail.innerHTML = '';
  const sorted = Object.entries(detail).sort((a,b) => b[1]-a[1]);
  sorted.forEach(([name, val]) => {
    surfaceDetail.innerHTML += `
      <div class="surface-row">
        <span class="surface-name">${name}</span>
        <div class="surface-bar-bg">
          <div class="surface-bar" style="width:${val}%"></div>
        </div>
        <span class="surface-val">${val}%</span>
      </div>`;
  });
}
