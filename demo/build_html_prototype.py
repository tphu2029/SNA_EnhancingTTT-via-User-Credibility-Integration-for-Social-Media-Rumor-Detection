import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.inference_engine import RumorDetectionDemoEngine
from demo.preset_cases import PRESET_CASES
from demo.utils import parse_twitter_datetime

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Hệ Thống Phán Đoán Tin Đồn: UC-TTT vs. Baseline TTT (Wu et al. 2025)</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root {
      --background: #f8fafc;
      --card: #ffffff;
      --foreground: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --primary: #2563eb;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --background: #090d16;
        --card: #111827;
        --foreground: #f8fafc;
        --muted: #94a3b8;
        --border: #1f2937;
        --primary: #3b82f6;
      }
    }
    body {
      background-color: var(--background);
      color: var(--foreground);
    }
    .case-btn-active {
      border-color: #2563eb !important;
      background-color: rgba(37, 99, 235, 0.1) !important;
      color: #2563eb !important;
    }
  </style>
</head>
<body class="p-3 sm:p-6 antialiased font-sans">
  <div class="max-w-6xl mx-auto space-y-6">
    
    <!-- HEADER -->
    <header class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-3">
      <div class="flex flex-wrap gap-2 text-xs font-semibold">
        <span class="px-2.5 py-1 rounded-md bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">PROTOTYPE SYSTEM</span>
        <span class="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">PHEME BENCHMARK (5-FOLD LOEO)</span>
      </div>
      <h1 class="text-xl sm:text-2xl font-bold tracking-tight">Hệ Thống Phán Đoán Tin Đồn: UC-TTT vs. Baseline TTT</h1>
      <p class="text-xs sm:text-sm text-[var(--muted)] leading-relaxed">
        Tái hiện mô hình cơ sở <b>Temporal Tree Transformer (Wu et al., PLOS ONE 2025)</b> và đánh giá đối chứng với mô hình đề xuất <b>UC-TTT</b> (tích hợp 12 đặc trưng độ uy tín người dùng qua cơ chế cổng động Residual Gate).
      </p>
      <div class="flex flex-wrap gap-2 pt-1 text-xs">
        <span class="px-2.5 py-1 rounded bg-[var(--background)] border border-[var(--border)]">BERT-base (768-d)</span>
        <span class="px-2.5 py-1 rounded bg-[var(--background)] border border-[var(--border)]">Tree Transformer (6L/12H)</span>
        <span class="px-2.5 py-1 rounded bg-[var(--background)] border border-[var(--border)]">Temporal BiGRU (Cumulative)</span>
        <span class="px-2.5 py-1 rounded bg-[var(--background)] border border-[var(--border)]">12 User Credibility Features</span>
        <span class="px-2.5 py-1 rounded bg-[var(--background)] border border-[var(--border)]">PyTorch 2.12 + CUDA</span>
      </div>
    </header>

    <!-- CASE SELECTOR -->
    <section class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-4 shadow-sm space-y-3">
      <div class="text-xs font-bold uppercase tracking-wider text-[var(--muted)]">Chọn Ca Thực Nghiệm Đối Chiếu (Dữ liệu thực tế từ PHEME 5 Folds):</div>
      <div id="case-buttons" class="grid grid-cols-1 sm:grid-cols-5 gap-2"></div>
      <div id="case-summary-box" class="p-3 text-xs sm:text-sm rounded-lg bg-blue-500/5 border border-blue-500/20 text-[var(--foreground)]"></div>
    </section>

    <!-- IMPACT BANNER -->
    <div id="verdict-banner" class="border rounded-xl p-4 shadow-sm space-y-1">
      <h2 id="verdict-title" class="text-base sm:text-lg font-bold"></h2>
      <p id="verdict-desc" class="text-xs sm:text-sm text-[var(--muted)]"></p>
    </div>

    <!-- SECTION 1: INPUT BÀI BÁO -->
    <section class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4">
      <h3 class="text-base font-bold flex items-center gap-2">1. Đầu Vào Mô Hình (Source Tweet & Cascade — Chuẩn Bài Báo)</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- Author Profile -->
        <div class="border border-[var(--border)] rounded-lg p-4 bg-[var(--background)] space-y-2">
          <h4 class="font-bold text-xs uppercase tracking-wide text-blue-600 dark:text-blue-400">Hồ Sơ Tác Giả (Author Profile)</h4>
          <div class="flex items-center gap-2">
            <span id="author-name" class="font-bold text-sm"></span>
            <span id="author-verified" class="text-blue-500 font-bold">(Verified)</span>
          </div>
          <div id="author-handle" class="text-xs text-[var(--muted)]"></div>
          <div class="text-xs space-y-1 pt-1">
            <div class="flex justify-between"><span>Xác minh:</span><b id="author-verified-text"></b></div>
            <div class="flex justify-between"><span>Followers:</span><b id="author-followers"></b></div>
            <div class="flex justify-between"><span>Following:</span><b id="author-friends"></b></div>
            <div class="flex justify-between"><span>Tỷ lệ uy tín:</span><b id="author-rep"></b></div>
            <div class="flex justify-between"><span>Số bài đăng:</span><b id="author-statuses"></b></div>
          </div>
          <div class="pt-2 border-t border-[var(--border)]">
            <div class="flex justify-between text-xs font-semibold mb-1">
              <span>Điểm uy tín tài khoản:</span>
              <span id="author-score-val"></span>
            </div>
            <div class="w-full bg-[var(--border)] h-2 rounded-full overflow-hidden">
              <div id="author-score-bar" class="h-full bg-emerald-500 rounded-full"></div>
            </div>
          </div>
        </div>

        <!-- Source Claim -->
        <div class="md:col-span-2 border border-[var(--border)] rounded-lg p-4 bg-[var(--background)] space-y-3">
          <h4 class="font-bold text-xs uppercase tracking-wide text-blue-600 dark:text-blue-400">Nội Dung Tuyên Bố (Source Claim)</h4>
          <p id="claim-text" class="text-sm sm:text-base italic p-3 rounded-lg border-l-4 border-blue-500 bg-[var(--card)]"></p>
          <div class="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs text-[var(--muted)]">
            <div><b>Thời điểm:</b> <span id="meta-time"></span></div>
            <div><b>Sự kiện:</b> <span id="meta-event"></span></div>
            <div><b>Nhãn thực tế:</b> <span id="meta-ground-truth" class="font-bold"></span></div>
            <div><b>Tổng số nodes:</b> <span id="meta-nodes"></span></div>
            <div><b>Số phản hồi:</b> <span id="meta-reactions"></span></div>
            <div><b>Số lát cắt (Windows):</b> <span id="meta-windows"></span></div>
          </div>
        </div>
      </div>

      <!-- Expandable Cascade Details -->
      <details class="text-xs border border-[var(--border)] rounded-lg bg-[var(--background)]">
        <summary class="p-3 font-semibold cursor-pointer hover:bg-slate-500/5">Xem Chi Tiết Cây Phản Hồi Lan Truyền (Propagation Cascade Details)</summary>
        <div class="p-3 border-t border-[var(--border)] overflow-x-auto max-h-60">
          <table class="w-full text-left">
            <thead>
              <tr class="border-b border-[var(--border)] text-[var(--muted)]">
                <th class="p-1">Thời điểm</th>
                <th class="p-1">Tác giả</th>
                <th class="p-1">Xác minh</th>
                <th class="p-1">Followers</th>
                <th class="p-1">Nội dung phản hồi</th>
              </tr>
            </thead>
            <tbody id="reactions-table-body" class="divide-y divide-[var(--border)]"></tbody>
          </table>
        </div>
      </details>
    </section>

    <!-- SECTION 2: SO SÁNH ĐỐI ĐẦU -->
    <section class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4">
      <h3 class="text-base font-bold flex items-center gap-2">2. So Sánh Đối Đầu Song Song (Head-to-Head: TTT Baseline vs. UC-TTT Proposed)</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- TTT Baseline -->
        <div class="border border-[var(--border)] rounded-lg p-4 bg-[var(--background)] space-y-3">
          <div class="text-center pb-2 border-b border-[var(--border)]">
            <h4 class="font-bold text-sm text-[var(--muted)]">Baseline: TTT Gốc (Wu et al. 2025)</h4>
            <div class="text-[11px] text-[var(--muted)]">Chỉ dùng Văn bản BERT 768-d + Cây lan truyền (KHÔNG CÓ User)</div>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-xs text-[var(--muted)]">Phán đoán:</span>
            <span id="ttt-pred-label" class="px-3 py-1 rounded-full text-xs font-bold"></span>
          </div>
          <div class="space-y-2 text-xs">
            <div>
              <div class="flex justify-between mb-1"><span>Xác suất Tin đồn P(Rumour):</span><b id="ttt-p-rumour" class="text-rose-600"></b></div>
              <div class="w-full bg-[var(--border)] h-2.5 rounded-full overflow-hidden">
                <div id="ttt-bar-rumour" class="h-full bg-rose-500 rounded-full"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between mb-1"><span>Xác suất Tin xác thực P(Non-rumour):</span><b id="ttt-p-non" class="text-emerald-600"></b></div>
              <div class="w-full bg-[var(--border)] h-2.5 rounded-full overflow-hidden">
                <div id="ttt-bar-non" class="h-full bg-emerald-500 rounded-full"></div>
              </div>
            </div>
          </div>
          <div id="ttt-correct-badge" class="text-center text-xs font-bold pt-2"></div>
        </div>

        <!-- UC-TTT Proposed -->
        <div class="border border-blue-500/30 rounded-lg p-4 bg-[var(--background)] space-y-3 ring-1 ring-blue-500/20">
          <div class="text-center pb-2 border-b border-blue-500/20">
            <h4 class="font-bold text-sm text-blue-600 dark:text-blue-400">Đề Xuất: UC-TTT (Nhóm 6)</h4>
            <div class="text-[11px] text-[var(--muted)]">Tích hợp 12 đặc trưng User Credibility qua Cổng Residual Gate</div>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-xs text-[var(--muted)]">Phán đoán:</span>
            <span id="ttt-pred-label" class="px-3 py-1 rounded-full text-xs font-bold"></span>
          </div>
          <div class="space-y-2 text-xs">
            <div>
              <div class="flex justify-between mb-1"><span>Xác suất Tin đồn P(Rumour):</span><b id="ttt-p-rumour" class="text-rose-600"></b></div>
              <div class="w-full bg-[var(--border)] h-2.5 rounded-full overflow-hidden">
                <div id="ttt-bar-rumour" class="h-full bg-rose-500 rounded-full"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between mb-1"><span>Xác suất Tin xác thực P(Non-rumour):</span><b id="ttt-p-non" class="text-emerald-600"></b></div>
              <div class="w-full bg-[var(--border)] h-2.5 rounded-full overflow-hidden">
                <div id="ttt-bar-non" class="h-full bg-emerald-500 rounded-full"></div>
              </div>
            </div>
          </div>
          <div id="ttt-correct-badge" class="text-center text-xs font-bold pt-2"></div>
        </div>

        <!-- UC-TTT Proposed -->
        <div class="border border-blue-500/30 rounded-lg p-4 bg-[var(--background)] space-y-3 ring-1 ring-blue-500/20">
          <div class="text-center pb-2 border-b border-blue-500/20">
            <h4 class="font-bold text-sm text-blue-600 dark:text-blue-400">Đề Xuất: UC-TTT (Nhóm 6)</h4>
            <div class="text-[11px] text-[var(--muted)]">Tích hợp 12 đặc trưng User Credibility qua Cổng Residual Gate</div>
          </div>
          <div class="flex justify-between items-center">
            <span class="text-xs text-[var(--muted)]">Phán đoán:</span>
            <span id="uc-pred-label" class="px-3 py-1 rounded-full text-xs font-bold"></span>
          </div>
          <div class="space-y-2 text-xs">
            <div>
              <div class="flex justify-between mb-1"><span>Xác suất Tin đồn P(Rumour):</span><b id="uc-p-rumour" class="text-rose-600"></b></div>
              <div class="w-full bg-[var(--border)] h-2.5 rounded-full overflow-hidden">
                <div id="uc-bar-rumour" class="h-full bg-rose-500 rounded-full"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between mb-1"><span>Xác suất Tin xác thực P(Non-rumour):</span><b id="uc-p-non" class="text-emerald-600"></b></div>
              <div class="w-full bg-[var(--border)] h-2.5 rounded-full overflow-hidden">
                <div id="uc-bar-non" class="h-full bg-emerald-500 rounded-full"></div>
              </div>
            </div>
          </div>
          <div id="uc-correct-badge" class="text-center text-xs font-bold pt-2"></div>
        </div>
      </div>
    </section>

    <!-- SECTION 3: TRỌNG SỐ GIẢI THÍCH -->
    <section class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4">
      <h3 class="text-base font-bold flex items-center gap-2">3. Trọng Số Giải Thích Của User Credibility (Residual Gate α)</h3>
      <div class="border border-[var(--border)] rounded-lg p-4 bg-[var(--background)] space-y-3">
        <div class="text-xs text-[var(--muted)]">
          Cơ chế Residual Gate học trọng số <b>α ∈ [0, 1]</b> để quyết định tỷ lệ dung hòa giữa <b>Độ uy tín người dùng</b> và <b>Ngữ nghĩa cấu trúc văn bản lan truyền</b>:
        </div>
        
        <div class="space-y-2">
          <div class="flex justify-between text-xs font-bold">
            <span class="text-emerald-600 dark:text-emerald-400">Trọng số Người dùng: <span id="gate-user-pct"></span></span>
            <span class="text-blue-600 dark:text-blue-400">Trọng số Ngữ nghĩa Văn bản: <span id="gate-text-pct"></span></span>
          </div>
          <div class="w-full bg-blue-500 h-3 rounded-full overflow-hidden flex">
            <div id="gate-user-bar" class="h-full bg-emerald-500 transition-all duration-300"></div>
            <div id="gate-text-bar" class="h-full bg-blue-500 transition-all duration-300"></div>
          </div>
        </div>

        <div id="gate-explanation" class="text-xs p-3 rounded-lg bg-slate-500/10 border border-[var(--border)] text-[var(--foreground)]"></div>
      </div>

      <!-- 12 Features Breakdown -->
      <details class="text-xs border border-[var(--border)] rounded-lg bg-[var(--background)]">
        <summary class="p-3 font-semibold cursor-pointer hover:bg-slate-500/5">Bóc Tách 12 Đặc Trưng Độ Uy Tín Người Dùng Của Tác Giả Gốc</summary>
        <div class="p-3 border-t border-[var(--border)] overflow-x-auto">
          <table class="w-full text-left">
            <thead>
              <tr class="border-b border-[var(--border)] text-[var(--muted)]">
                <th class="p-1">Nhóm</th>
                <th class="p-1">Tên đặc trưng</th>
                <th class="p-1">Giá trị gốc</th>
                <th class="p-1">Log-transform</th>
                <th class="p-1">Tác động suy luận</th>
              </tr>
            </thead>
            <tbody id="features-table-body" class="divide-y divide-[var(--border)]"></tbody>
          </table>
        </div>
      </details>
    </section>

    <!-- SECTION 4: DIỄN BIẾN THEO THỜI GIAN -->
    <section class="bg-[var(--card)] border border-[var(--border)] rounded-xl p-5 shadow-sm space-y-4">
      <h3 class="text-base font-bold flex items-center gap-2">4. Diễn Biến Cảnh Báo Theo Thời Gian (Early Rumor Detection Timeline)</h3>
      <p class="text-xs text-[var(--muted)]">
        Mô hình chia cây lan truyền thành các cửa sổ con tích lũy (Cumulative Subtrees, mỗi lát cắt chứa tối đa 8 bài tweet phản hồi). Đường cong thể hiện xác suất tin đồn <b>P(Rumour)</b> qua từng mốc thời gian:
      </p>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <!-- SVG Curve Chart -->
        <div class="md:col-span-2 border border-[var(--border)] rounded-lg p-3 bg-[var(--background)] flex flex-col justify-between">
          <div class="flex justify-between items-center text-xs pb-2 border-b border-[var(--border)]">
            <span class="font-bold">Đường Cong Xác Suất Tin Đồn (%) Theo Thời Gian</span>
            <div class="flex gap-3 text-[11px]">
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-blue-600 inline-block"></span> UC-TTT</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded-full bg-amber-600 inline-block"></span> TTT Gốc</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-0.5 bg-red-500 inline-block"></span> Ngưỡng 50%</span>
            </div>
          </div>
          <div id="chart-container" class="w-full h-48 py-2"></div>
        </div>

        <!-- Milestones list -->
        <div class="border border-[var(--border)] rounded-lg p-3 bg-[var(--background)] space-y-2 max-h-60 overflow-y-auto text-xs">
          <div class="font-bold text-xs uppercase tracking-wide text-[var(--muted)]">Chi Tiết Từng Lát Cắt Tích Lũy:</div>
          <div id="milestones-list" class="space-y-1.5"></div>
        </div>
      </div>
    </section>

    <!-- FOOTER -->
    <footer class="text-center text-xs text-[var(--muted)] pt-4 pb-8 space-y-1">
      <div>Đồ Án Khai Phá Mạng Xã Hội (SNA) — Đề tài: Cải tiến Temporal Tree Transformer (UC-TTT)</div>
      <div>Dữ liệu đánh giá: PHEME Dataset | Bài báo tham chiếu: Wu et al. (PLOS ONE 2025)</div>
    </footer>
  </div>

  <script>
    const CASES = __CASES_JSON_PLACEHOLDER__;
    let currentIdx = 0;

    function renderButtons() {
      const container = document.getElementById("case-buttons");
      container.innerHTML = "";
      CASES.forEach((c, idx) => {
        const btn = document.createElement("button");
        btn.className = `p-2.5 rounded-lg border text-left text-xs font-semibold transition-all ${idx === currentIdx ? 'case-btn-active border-blue-600' : 'border-[var(--border)] bg-[var(--background)] hover:border-blue-400'}`;
        btn.innerHTML = `<div class="truncate font-bold">${c.title}</div><div class="text-[10px] text-[var(--muted)] truncate">${c.badge}</div>`;
        btn.onclick = () => selectCase(idx);
        container.appendChild(btn);
      });
    }

    function selectCase(idx) {
      currentIdx = idx;
      renderButtons();
      renderCase(CASES[idx]);
    }

    function renderCase(c) {
      // Summary
      document.getElementById("case-summary-box").innerHTML = `<b>Danh mục:</b> ${c.category} — ${c.summary}`;

      // Verdict Banner
      const banner = document.getElementById("verdict-banner");
      if (c.impact.status === "CORRECTED_BY_UC") {
        banner.className = "border rounded-xl p-4 shadow-sm space-y-1 bg-emerald-500/10 border-emerald-500/30 text-emerald-900 dark:text-emerald-100";
      } else {
        banner.className = "border rounded-xl p-4 shadow-sm space-y-1 bg-blue-500/10 border-blue-500/30 text-blue-900 dark:text-blue-100";
      }
      document.getElementById("verdict-title").innerText = c.impact.title;
      document.getElementById("verdict-desc").innerText = c.impact.description;

      // Section 1
      const src = c.source_tweet;
      document.getElementById("author-name").innerText = src.author_name;
      document.getElementById("author-handle").innerText = `@${src.screen_name}`;
      document.getElementById("author-verified").style.display = src.verified ? "inline" : "none";
      document.getElementById("author-verified-text").innerText = src.verified ? "ĐÃ XÁC MINH (Tick xanh)" : "Chưa xác minh";
      document.getElementById("author-followers").innerText = Number(src.followers_count).toLocaleString();
      document.getElementById("author-friends").innerText = Number(src.friends_count).toLocaleString();
      document.getElementById("author-rep").innerText = Number(src.reputation_ratio).toFixed(2);
      document.getElementById("author-statuses").innerText = Number(src.statuses_count).toLocaleString();
      document.getElementById("author-score-val").innerText = `${src.credibility_score}/100`;
      document.getElementById("author-score-bar").style.width = `${src.credibility_score}%`;

      document.getElementById("claim-text").innerText = `"${src.text}"`;
      document.getElementById("meta-time").innerText = src.created_at;
      document.getElementById("meta-event").innerText = c.event;
      document.getElementById("meta-ground-truth").innerText = c.ground_truth_name;
      document.getElementById("meta-nodes").innerText = `${c.num_nodes} bài`;
      document.getElementById("meta-reactions").innerText = `${c.num_reactions} bài`;
      document.getElementById("meta-windows").innerText = `${c.timeline.ucttt.length} lát cắt`;

      // Reactions Table
      const tbReax = document.getElementById("reactions-table-body");
      tbReax.innerHTML = "";
      (c.reactions_list || []).forEach(r => {
        const tr = document.createElement("tr");
        tr.className = "text-[11px]";
        tr.innerHTML = `<td class="p-1">${r.created_at}</td><td class="p-1 font-semibold">@${r.screen_name}</td><td class="p-1">${r.verified ? '✔️' : '❌'}</td><td class="p-1">${Number(r.followers_count).toLocaleString()}</td><td class="p-1">${r.text}</td>`;
        tbReax.appendChild(tr);
      });

      // Section 2: Head-to-Head
      const ttt = c.ttt_prediction;
      const uc = c.ucttt_prediction;

      const tttP = Math.round(ttt.prob_rumour * 100);
      document.getElementById("ttt-pred-label").innerText = ttt.label_name;
      document.getElementById("ttt-pred-label").className = `px-3 py-1 rounded-full text-xs font-bold ${ttt.label === 1 ? 'bg-rose-500/20 text-rose-600' : 'bg-emerald-500/20 text-emerald-600'}`;
      document.getElementById("ttt-p-rumour").innerText = `${(ttt.prob_rumour * 100).toFixed(1)}%`;
      document.getElementById("ttt-p-non").innerText = `${(ttt.prob_non_rumour * 100).toFixed(1)}%`;
      document.getElementById("ttt-bar-rumour").style.width = `${tttP}%`;
      document.getElementById("ttt-bar-non").style.width = `${100 - tttP}%`;
      document.getElementById("ttt-correct-badge").innerText = ttt.is_correct ? "✅ Trùng khớp Nhãn Thực tế" : "❌ Sai lệch so với Nhãn Thực tế";
      document.getElementById("ttt-correct-badge").className = `text-center text-xs font-bold pt-2 ${ttt.is_correct ? 'text-emerald-600' : 'text-rose-600'}`;

      const ucP = Math.round(uc.prob_rumour * 100);
      document.getElementById("uc-pred-label").innerText = uc.label_name;
      document.getElementById("uc-pred-label").className = `px-3 py-1 rounded-full text-xs font-bold ${uc.label === 1 ? 'bg-rose-500/20 text-rose-600' : 'bg-emerald-500/20 text-emerald-600'}`;
      document.getElementById("uc-p-rumour").innerText = `${(uc.prob_rumour * 100).toFixed(1)}%`;
      document.getElementById("uc-p-non").innerText = `${(uc.prob_non_rumour * 100).toFixed(1)}%`;
      document.getElementById("uc-bar-rumour").style.width = `${ucP}%`;
      document.getElementById("uc-bar-non").style.width = `${100 - ucP}%`;
      document.getElementById("uc-correct-badge").innerText = uc.is_correct ? "✅ Trùng khớp Nhãn Thực tế" : "❌ Sai lệch so với Nhãn Thực tế";
      document.getElementById("uc-correct-badge").className = `text-center text-xs font-bold pt-2 ${uc.is_correct ? 'text-emerald-600' : 'text-rose-600'}`;

      // Section 3: Gate Alpha
      const interp = c.interpretability;
      document.getElementById("gate-user-pct").innerText = `${interp.user_weight_pct}%`;
      document.getElementById("gate-text-pct").innerText = `${interp.text_weight_pct}%`;
      document.getElementById("gate-user-bar").style.width = `${interp.user_weight_pct}%`;
      document.getElementById("gate-text-bar").style.width = `${interp.text_weight_pct}%`;
      document.getElementById("gate-explanation").innerText = `💡 ${interp.explanation_text}`;

      // 12 Features
      const tbFeat = document.getElementById("features-table-body");
      tbFeat.innerHTML = "";
      (c.features_12 || []).forEach(f => {
        const tr = document.createElement("tr");
        tr.className = "text-[11px]";
        tr.innerHTML = `<td class="p-1 font-semibold text-[var(--muted)]">${f.group}</td><td class="p-1 font-medium">${f.name}</td><td class="p-1">${f.val_str}</td><td class="p-1 font-mono">${f.log_val}</td><td class="p-1 font-semibold">${f.impact}</td>`;
        tbFeat.appendChild(tr);
      });

      // Section 4: Timeline & SVG Chart
      renderTimeline(c.timeline);
    }

    function renderTimeline(timeline) {
      const tttList = timeline.ttt || [];
      const ucList = timeline.ucttt || [];
      const n = ucList.length;

      // Render milestones cards
      const msContainer = document.getElementById("milestones-list");
      msContainer.innerHTML = "";
      ucList.forEach((pt, i) => {
        const tttPt = tttList[i] || { prob_rumour: 0 };
        const pUc = (pt.prob_rumour * 100).toFixed(1);
        const pTtt = (tttPt.prob_rumour * 100).toFixed(1);
        const card = document.createElement("div");
        card.className = `p-2.5 rounded-lg border-l-4 bg-[var(--card)] border border-[var(--border)] ${pt.prob_rumour >= 0.5 ? 'border-l-rose-500' : 'border-l-emerald-500'}`;
        card.innerHTML = `<div class="font-bold">Mốc ${pt.elapsed_label} (${pt.cum_posts} bài):</div><div class="text-[11px] space-y-0.5 mt-0.5"><div>• UC-TTT: <b>${pUc}%</b> (${pt.prob_rumour >= 0.5 ? '🔴 Báo động' : '🟢 An toàn'})</div><div>• TTT Gốc: <b>${pTtt}%</b></div></div>`;
        msContainer.appendChild(card);
      });

      // Draw SVG chart
      const w = 500, h = 200, padX = 40, padY = 25;
      const chartW = w - padX * 2;
      const chartH = h - padY * 2;

      let ptsUc = "", ptsTtt = "";
      const stepX = n > 1 ? chartW / (n - 1) : chartW / 2;

      ucList.forEach((pt, i) => {
        const x = padX + (n > 1 ? i * stepX : chartW / 2);
        const yUc = padY + chartH * (1.0 - pt.prob_rumour);
        ptsUc += `${i === 0 ? 'M' : 'L'} ${x} ${yUc} `;
      });

      tttList.forEach((pt, i) => {
        const x = padX + (n > 1 ? i * stepX : chartW / 2);
        const yTtt = padY + chartH * (1.0 - pt.prob_rumour);
        ptsTtt += `${i === 0 ? 'M' : 'L'} ${x} ${yTtt} `;
      });

      const thresholdY = padY + chartH * 0.5;

      let circles = "";
      ucList.forEach((pt, i) => {
        const x = padX + (n > 1 ? i * stepX : chartW / 2);
        const yUc = padY + chartH * (1.0 - pt.prob_rumour);
        const yTtt = padY + chartH * (1.0 - tttList[i].prob_rumour);
        circles += `<circle cx="${x}" cy="${yUc}" r="4" fill="#2563eb" /><text x="${x}" y="${yUc - 8}" font-size="10" text-anchor="middle" fill="#2563eb" font-weight="bold">${(pt.prob_rumour * 100).toFixed(0)}%</text>`;
        circles += `<circle cx="${x}" cy="${yTtt}" r="4" fill="#ea580c" /><text x="${x}" y="${yTtt + 14}" font-size="10" text-anchor="middle" fill="#ea580c" font-weight="bold">${(tttList[i].prob_rumour * 100).toFixed(0)}%</text>`;
        circles += `<text x="${x}" y="${h - 6}" font-size="10" text-anchor="middle" fill="currentColor">${pt.elapsed_label}</text>`;
      });

      const svg = `
        <svg viewBox="0 0 ${w} ${h}" class="w-full h-full text-[var(--muted)]">
          <!-- Grid lines -->
          <line x1="${padX}" y1="${padY}" x2="${w - padX}" y2="${padY}" stroke="currentColor" stroke-opacity="0.15" />
          <line x1="${padX}" y1="${padY + chartH * 0.5}" x2="${w - padX}" y2="${padY + chartH * 0.5}" stroke="#ef4444" stroke-dasharray="4" stroke-opacity="0.6" />
          <line x1="${padX}" y1="${h - padY}" x2="${w - padX}" y2="${h - padY}" stroke="currentColor" stroke-opacity="0.2" />
          
          <!-- Y Labels -->
          <text x="${padX - 8}" y="${padY + 4}" font-size="9" text-anchor="end" fill="currentColor">100%</text>
          <text x="${padX - 8}" y="${thresholdY + 3}" font-size="9" text-anchor="end" fill="#ef4444" font-weight="bold">50%</text>
          <text x="${padX - 8}" y="${h - padY + 3}" font-size="9" text-anchor="end" fill="currentColor">0%</text>
          
          <!-- Lines -->
          <path d="${ptsTtt}" fill="none" stroke="#ea580c" stroke-width="2.5" />
          <path d="${ptsUc}" fill="none" stroke="#2563eb" stroke-width="2.5" />
          
          <!-- Dots & Labels -->
          ${circles}
        </svg>
      `;

      document.getElementById("chart-container").innerHTML = svg;
    }

    // Init
    renderButtons();
    selectCase(0);
  </script>
</body>
</html>
"""

def generate_html_prototype():
    print("Initializing demo engine for HTML prototype...")
    engine = RumorDetectionDemoEngine()
    engine.load_data()
    
    full_cases = []
    for p in PRESET_CASES:
        th = engine.threads_by_id[p["thread_id"]]
        res = engine.predict_thread(th, fold_idx=p["fold"])
        res["badge"] = p["badge"]
        res["category"] = p["category"]
        res["summary"] = p["summary"]
        res["title"] = p["title"]
        
        # Add reaction preview
        reactions_preview = []
        for r in th.get("reactions", [])[:15]:
            u = r.get("raw_user", r.get("user", {}))
            reactions_preview.append({
                "created_at": r.get("created_at", ""),
                "screen_name": u.get("screen_name", "anon"),
                "verified": bool(u.get("verified", False)),
                "followers_count": int(u.get("followers_count", 0) or 0),
                "text": r.get("text", "")
            })
        res["reactions_list"] = reactions_preview
        
        # 12 features breakdown
        src = res["source_tweet"]
        src_raw_u = th.get("source_tweet", {}).get("raw_user", th.get("source_tweet", {}).get("user", {}))
        favs = src_raw_u.get("favourites_count", 0) or 0
        lists = src_raw_u.get("listed_count", 0) or 0
        is_default_avatar = src_raw_u.get("default_profile_image", False)
        
        src_created_str = th.get("source_tweet", {}).get("created_at", "")
        u_created_str = src_raw_u.get("created_at", "")
        t_dt = parse_twitter_datetime(src_created_str)
        u_dt = parse_twitter_datetime(u_created_str)
        if t_dt and u_dt:
            age_days = max(0.0, (t_dt - u_dt).total_seconds() / 86400.0)
        else:
            age_days = 0.0
            
        desc = src_raw_u.get("description", "") or ""
        has_bio = len(desc.strip()) > 0
        has_url = bool(src_raw_u.get("url", None))
        
        reax = th.get("reactions", [])
        verified_reax_count = sum(1 for rx in reax if rx.get("raw_user", rx.get("user", {})).get("verified", False))
        verified_ratio = (verified_reax_count / len(reax)) if len(reax) > 0 else 0.0
        
        res["features_12"] = [
            {"group": "1. Mạng xã hội", "name": "Followers (log_followers)", "val_str": f"{src['followers_count']:,}", "log_val": f"{math.log1p(src['followers_count']):.2f}", "impact": "Lan truyền rộng"},
            {"group": "1. Mạng xã hội", "name": "Following (log_friends)", "val_str": f"{src['friends_count']:,}", "log_val": f"{math.log1p(src['friends_count']):.2f}", "impact": "Kết nối bạn bè"},
            {"group": "1. Mạng xã hội", "name": "Tỷ lệ uy tín (reputation_ratio)", "val_str": f"{src['reputation_ratio']:.2f}", "log_val": f"{math.log1p(src['reputation_ratio']):.2f}", "impact": "Uy tín vượt trội" if src['reputation_ratio'] > 1.0 else "Bình thường"},
            {"group": "2. Lịch sử", "name": "Tuổi thọ tài khoản (log_account_age_days)", "val_str": f"{int(age_days):,} ngày", "log_val": f"{math.log1p(age_days):.2f}", "impact": "Thâm niên cao" if age_days > 365 else "Mới tạo"},
            {"group": "2. Lịch sử", "name": "Số tweet đã đăng (log_statuses_count)", "val_str": f"{src['statuses_count']:,}", "log_val": f"{math.log1p(src['statuses_count']):.2f}", "impact": "Tần suất hoạt động"},
            {"group": "2. Lịch sử", "name": "Số lượt thích (log_favourites_count)", "val_str": f"{favs:,}", "log_val": f"{math.log1p(favs):.2f}", "impact": "Mức độ tương tác"},
            {"group": "3. Xác thực & Hồ sơ", "name": "Xác minh chính chủ (verified)", "val_str": "Có (Tick xanh)" if src['verified'] else "Không", "log_val": "1.00" if src['verified'] else "0.00", "impact": "Bảo chứng danh tính" if src['verified'] else "Chưa định danh"},
            {"group": "3. Xác thực & Hồ sơ", "name": "Có tiểu sử Bio (has_description)", "val_str": "Có" if has_bio else "Không", "log_val": "1.00" if has_bio else "0.00", "impact": "Thông tin minh bạch"},
            {"group": "3. Xác thực & Hồ sơ", "name": "Có liên kết ngoài (has_url)", "val_str": "Có" if has_url else "Không", "log_val": "1.00" if has_url else "0.00", "impact": "Dẫn nguồn xác thực"},
            {"group": "3. Xác thực & Hồ sơ", "name": "Ảnh đại diện mặc định (default_profile_image)", "val_str": "Mặc định (Bot)" if is_default_avatar else "Tùy biến", "log_val": "1.00" if is_default_avatar else "0.00", "impact": "Nghi vấn Clone" if is_default_avatar else "Đầu tư hồ sơ"},
            {"group": "4. Danh sách & Cây", "name": "Số danh sách vào (log_listed_count)", "val_str": f"{lists:,}", "log_val": f"{math.log1p(lists):.2f}", "impact": "Được cộng đồng theo dõi"},
            {"group": "4. Danh sách & Cây", "name": "Tỷ lệ người phản hồi có tick xanh (verified_reaction_ratio)", "val_str": f"{verified_ratio * 100:.1f}% ({verified_reax_count}/{len(reax)})", "log_val": f"{verified_ratio:.3f}", "impact": "Cộng đồng kiểm chứng"}
        ]
        
        full_cases.append(res)
        
    json_str = json.dumps(full_cases, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__CASES_JSON_PLACEHOLDER__", json_str)
    
    out_path = ROOT / "demo" / "demo_prototype.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> Successfully generated HTML prototype: {out_path} ({len(html):,} bytes)")

if __name__ == "__main__":
    generate_html_prototype()
