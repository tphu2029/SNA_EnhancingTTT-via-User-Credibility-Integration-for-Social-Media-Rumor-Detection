import math
import os
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo.inference_engine import RumorDetectionDemoEngine, EVENT_FOLD_MAP, LABEL_NAMES
from demo.preset_cases import PRESET_CASES
from demo.utils import parse_twitter_datetime, UserCredibilityExtractor

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ Thống Phán Đoán Tin Đồn: UC-TTT vs. TTT Gốc",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tùy biến phong cách giao diện chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        background: #f8fafc;
        padding: 20px 24px;
        border-radius: 12px;
        color: #0f172a;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
    }
    .dark .main-header {
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid #1e293b;
    }
    .main-header h1 {
        font-size: 24px;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #475569;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
        max-width: 950px;
    }
    .dark .main-header p {
        color: #94a3b8;
    }
    .badge-bar {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .badge-pill {
        background: #ffffff;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    .dark .badge-pill {
        background: #1e293b;
        color: #cbd5e1;
        border: 1px solid #334155;
    }
    .ui-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 16px;
    }
    .dark .ui-card {
        background: #0f172a;
        border: 1px solid #1e293b;
    }
    .metric-callout {
        background-color: #f8fafc;
        border-left: 3px solid #2563eb;
        padding: 12px 14px;
        border-radius: 6px;
        margin-top: 12px;
        border: 1px solid #e2e8f0;
        border-left-width: 3px;
    }
    .dark .metric-callout {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left-width: 3px;
        border-left-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    engine = RumorDetectionDemoEngine()
    engine.load_data()
    return engine

def render_header():
    st.markdown("""
    <div class="main-header">
        <div style="display: flex; gap: 8px; margin-bottom: 8px;">
            <span class="badge-pill" style="font-weight: 700; color: #2563eb;">PROTOTYPE SYSTEM</span>
            <span class="badge-pill" style="color: #059669; font-weight: 600;">PHEME BENCHMARK (5-FOLD LOEO)</span>
        </div>
        <h1>Hệ Thống Phán Đoán Tin Đồn: UC-TTT vs. TTT Gốc</h1>
        <p>Tái hiện mô hình cơ sở <b>Temporal Tree Transformer (Wu et al., PLOS ONE 2025)</b> và đánh giá đối chứng với mô hình đề xuất <b>UC-TTT</b> (tích hợp 12 đặc trưng độ uy tín người dùng qua cơ chế cổng động Residual Gate).</p>
        <div class="badge-bar">
            <span class="badge-pill">BERT-base (768-d)</span>
            <span class="badge-pill">Tree Transformer (6L/12H)</span>
            <span class="badge-pill">Temporal BiGRU (Cumulative)</span>
            <span class="badge-pill">12 User Credibility Features</span>
            <span class="badge-pill">PyTorch 2.12 + CUDA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    render_header()
    
    with st.spinner("Đang khởi tạo Inference Engine và nạp trọng số PyTorch..."):
        try:
            engine = get_engine()
        except Exception as e:
            st.error(f"Lỗi khởi tạo hệ thống: {e}")
            return

    # Sidebar Navigation
    st.sidebar.title("Bảng Điều Khiển")
    app_mode = st.sidebar.radio(
        "Chế độ hoạt động:",
        [
            "Ca thực nghiệm đối chứng (Case Studies)",
            "Khám phá toàn bộ dữ liệu PHEME",
            "Thử nghiệm tự do (Interactive Sandbox)",
        ]
    )

    selected_thread = None
    active_fold = 3

    # =========================================================================
    # CHẾ ĐỘ 1: CÁC CA THỰC NGHIỆM ĐỐI CHỨNG (5 EVENTS BENCHMARK WU ET AL. 2025)
    # =========================================================================
    if app_mode == "Ca thực nghiệm đối chứng (Case Studies)":
        st.markdown("### Chọn Ca Thực Nghiệm Đối Chiếu (5 Sự Kiện Benchmark PHEME v5)")
        st.caption("Mỗi ca đại diện chính xác cho 1 trong 5 sự kiện benchmark theo chuẩn bài báo Wu et al. (PLOS ONE 2025): CH, SS, FG, OS, GC.")
        
        if "selected_case_idx" not in st.session_state:
            st.session_state.selected_case_idx = 0

        def _on_case_click(target_idx: int):
            st.session_state.selected_case_idx = target_idx

        cols = st.columns(len(PRESET_CASES))
        case_btn_labels = [
            "1. Charlie Hebdo (CH)\n@LadySandersfarm",
            "2. Sydney Siege (SS)\n@bkjabour Guardian",
            "3. Ferguson (FG)\n@inthefade (874k)",
            "4. Ottawa Shoot. (OS)\n@BreakingNews (7M)",
            "5. Germanwings (GC)\n@CNN (16.3M Fol)",
        ]
        
        for idx, (col, lbl) in enumerate(zip(cols, case_btn_labels)):
            with col:
                is_active = (st.session_state.selected_case_idx == idx)
                st.button(
                    lbl,
                    key=f"case_btn_{idx}",
                    type="primary" if is_active else "secondary",
                    on_click=_on_case_click,
                    args=(idx,),
                    use_container_width=True,
                )
                
        preset = PRESET_CASES[st.session_state.selected_case_idx]
        thread_id = preset["thread_id"]
        selected_thread = engine.threads_by_id.get(thread_id)
        active_fold = preset["fold"]
        
        st.info(f"**{preset['title']}**\n\n**Danh mục:** {preset['category']}\n\n{preset['summary']}")

    # =========================================================================
    # CHẾ ĐỘ 2: KHÁM PHÁ TOÀN BỘ TẬP DỮ LIỆU PHEME
    # =========================================================================
    elif app_mode == "Khám phá toàn bộ dữ liệu PHEME":
        st.sidebar.subheader("Bộ lọc sự kiện:")
        events = engine.get_events()
        event_name = st.sidebar.selectbox("Sự kiện (Event):", events, index=3)
        active_fold = EVENT_FOLD_MAP[event_name]
        
        threads_in_event = engine.get_threads_for_event(event_name, min_replies=2)
        st.sidebar.write(f"Số lượng threads có phản hồi: **{len(threads_in_event)}**")
        
        filter_label = st.sidebar.selectbox("Lọc theo nhãn:", ["Tất cả", "Chỉ Tin đồn (Rumour)", "Chỉ Tin xác thực (Non-rumour)"])
        if filter_label == "Chỉ Tin đồn (Rumour)":
            threads_in_event = [th for th in threads_in_event if th["label"] == 1]
        elif filter_label == "Chỉ Tin xác thực (Non-rumour)":
            threads_in_event = [th for th in threads_in_event if th["label"] == 0]
            
        thread_options = [
            f"ID: {th['thread_id']} | @{th['source_tweet'].get('raw_user', {}).get('screen_name', 'user')} ({len(th['reactions'])} phản hồi)"
            for th in threads_in_event
        ]
        
        if threads_in_event:
            th_idx = st.sidebar.selectbox("Chọn bài đăng nguồn:", range(len(threads_in_event)), format_func=lambda i: thread_options[i])
            selected_thread = threads_in_event[th_idx]
        else:
            st.warning("Không có bài đăng nào thỏa mãn bộ lọc.")

    # =========================================================================
    # CHẾ ĐỘ 3: THỬ NGHIỆM TỰ DO (INTERACTIVE SANDBOX)
    # =========================================================================
    elif app_mode == "Thử nghiệm tự do (Interactive Sandbox)":
        st.sidebar.subheader("Thiết lập tác giả bài viết:")
        sb_author_name = st.sidebar.text_input("Tên tác giả:", value="Tin Nhanh 24/7")
        sb_screen_name = st.sidebar.text_input("Screen name (@):", value="tinnhanh_247")
        sb_verified = st.sidebar.checkbox("Tài khoản xác minh (Tick xanh ✔️)", value=False)
        sb_followers = st.sidebar.number_input("Số người theo dõi (Followers):", min_value=0, max_value=50000000, value=150)
        sb_friends = st.sidebar.number_input("Số đang theo dõi (Friends):", min_value=0, max_value=500000, value=1200)
        sb_statuses = st.sidebar.number_input("Số bài viết đã đăng:", min_value=0, max_value=500000, value=35)
        sb_age_days = st.sidebar.slider("Tuổi thọ tài khoản (ngày):", min_value=1, max_value=4000, value=15)
        sb_desc = st.sidebar.text_input("Tiểu sử (Bio):", value="Kênh cập nhật tin tức mạng xã hội")
        
        st.subheader("✍️ Nhập nội dung tin đồn và các phản hồi cộng đồng")
        col_c1, col_c2 = st.columns([3, 2])
        with col_c1:
            sb_claim_text = st.text_area(
                "Nội dung bài đăng tuyên bố (Source Claim Tweet):",
                value="CẢNH BÁO: Vừa xảy ra vụ nổ lớn tại trung tâm thương mại, hàng chục người đang tháo chạy trong hoảng loạn! Cơ quan chức năng đang phong tỏa hiện trường.",
                height=110,
            )
        with col_c2:
            st.markdown("**Các phản hồi giả định (Reactions):**")
            r1 = st.text_input("Phản hồi 1 (sau 10 phút):", value="Thật không vậy? Tôi vừa đi qua đó thấy bình thường mà?")
            r2 = st.text_input("Phản hồi 2 (sau 30 phút):", value="Tin giả đấy mọi người ơi, công an vừa đính chính là diễn tập PCCC thôi!")
            r3 = st.text_input("Phản hồi 3 (sau 60 phút):", value="Đề nghị mọi người không chia sẻ tin chưa kiểm chứng gây hoang mang.")
            
        reactions = [
            {"id": "sb_r1", "text": r1, "created_at": "Wed Oct 22 10:10:00 +0000 2026", "raw_user": {"followers_count": 500, "friends_count": 300, "verified": False}},
            {"id": "sb_r2", "text": r2, "created_at": "Wed Oct 22 10:30:00 +0000 2026", "raw_user": {"followers_count": 12000, "friends_count": 400, "verified": True}},
            {"id": "sb_r3", "text": r3, "created_at": "Wed Oct 22 11:00:00 +0000 2026", "raw_user": {"followers_count": 850, "friends_count": 600, "verified": False}},
        ]
        
        selected_thread = {
            "thread_id": "custom_sandbox_claim",
            "event": "ottawashooting",
            "label": -1,
            "source_tweet": {
                "id": "sb_source_0",
                "text": sb_claim_text,
                "created_at": "Wed Oct 22 10:00:00 +0000 2026",
                "raw_user": {
                    "name": sb_author_name,
                    "screen_name": sb_screen_name,
                    "verified": sb_verified,
                    "followers_count": sb_followers,
                    "friends_count": sb_friends,
                    "statuses_count": sb_statuses,
                    "description": sb_desc,
                    "created_at": "Thu Oct 08 10:00:00 +0000 2026",
                }
            },
            "reactions": reactions,
            "edges": [("sb_source_0", "sb_r1"), ("sb_source_0", "sb_r2"), ("sb_r2", "sb_r3")],
        }
        active_fold = 3

    if selected_thread is None:
        st.info("Vui lòng chọn hoặc nhập bài đăng để thực hiện suy luận.")
        return

    # Chạy suy luận hai mô hình song song
    with st.spinner("Đang chạy forward pass song song TTT Baseline & UC-TTT qua các lát cắt thời gian..."):
        results = engine.predict_thread(selected_thread, fold_idx=active_fold)

    st.markdown("---")

    # =========================================================================
    # HUY HIỆU ĐÁNH GIÁ TÁC ĐỘNG (VERDICT ALERT BANNER)
    # =========================================================================
    impact = results["impact"]
    if impact["status"] == "CORRECTED_BY_UC":
        st.success(f"### {impact['title']}\n{impact['description']}")
    elif impact["status"] == "BOTH_CORRECT":
        st.info(f"### {impact['title']}\n{impact['description']}")
    elif impact["status"] == "UNANNOTATED":
        st.warning(f"### {impact['title']}\n{impact['description']}")
    else:
        st.error(f"### {impact['title']}\n{impact['description']}")

    # =========================================================================
    # MỤC 1: CẤU TRÚC ĐẦU VÀO BÀI BÁO (SOURCE TWEET & CASCADE METRICS)
    # =========================================================================
    st.subheader("1. Đầu Vào Mô Hình (Source Tweet & Cascade — Chuẩn Bài Báo)")
    
    src = results["source_tweet"]
    col_auth, col_claim = st.columns([1, 2])
    
    with col_auth:
        st.markdown(f"""
        <div class="ui-card">
            <h4 style="margin: 0 0 8px 0; color: #1e3a8a;">Hồ sơ Tác giả (Author Profile)</h4>
            <p style="margin: 3px 0;"><b>Tên:</b> {src['author_name']} {'<span style="color:#3b82f6;">(Verified)</span>' if src['verified'] else ''}</p>
            <p style="margin: 3px 0;"><b>Handle:</b> @{src['screen_name']}</p>
            <p style="margin: 3px 0;"><b>Xác minh:</b> {'<span style="color: #10b981; font-weight: bold;">ĐÃ XÁC MINH (Tick xanh)</span>' if src['verified'] else '<span style="color: #64748b;">Chưa xác minh</span>'}</p>
            <p style="margin: 3px 0;"><b>Followers:</b> {src['followers_count']:,} | <b>Following:</b> {src['friends_count']:,}</p>
            <p style="margin: 3px 0;"><b>Tỷ lệ Uy tín:</b> {src['reputation_ratio']:.2f}</p>
            <p style="margin: 3px 0;"><b>Số bài đăng:</b> {src['statuses_count']:,}</p>
            <div style="margin-top: 10px;">
                <span style="font-size: 13px; font-weight: 600;">Điểm Uy tín Tài khoản:</span>
                <div style="background-color: #e2e8f0; border-radius: 8px; height: 14px; width: 100%; margin-top: 4px;">
                    <div style="background-color: {'#10b981' if src['credibility_score'] > 65 else ('#f59e0b' if src['credibility_score'] > 40 else '#ef4444')}; width: {src['credibility_score']}%; height: 100%; border-radius: 8px;"></div>
                </div>
                <small style="color: #64748b;">Chỉ số: {src['credibility_score']}/100</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_claim:
        st.markdown(f"""
        <div class="ui-card">
            <h4 style="margin: 0 0 8px 0; color: #1e3a8a;">Nội dung Bài đăng Tuyên bố (Source Claim)</h4>
            <p style="font-size: 15.5px; font-style: italic; color: #1e293b; background: #f8fafc; padding: 14px; border-radius: 10px; border-left: 4px solid #3b82f6;">
                "{src['text']}"
            </p>
            <div style="display: flex; gap: 20px; font-size: 13px; color: #64748b; margin-top: 10px;">
                <span><b>Thời điểm:</b> {src['created_at']}</span>
                <span><b>Sự kiện:</b> {results['event']}</span>
                <span><b>Nhãn thực tế:</b> <b style="color: {'#dc2626' if results['ground_truth'] == 1 else '#16a34a'};">{results['ground_truth_name']}</b></span>
            </div>
            <div style="display: flex; gap: 20px; font-size: 13px; color: #334155; margin-top: 8px; background: #f1f5f9; padding: 10px 14px; border-radius: 8px;">
                <span><b>Tổng số nodes:</b> {results['num_nodes']}</span>
                <span><b>Số phản hồi:</b> {results['num_reactions']}</span>
                <span><b>Số lát cắt thời gian (Windows):</b> {len(results['timeline']['ucttt'])}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Xem cây phản hồi phân nhánh
    with st.expander("Xem chi tiết Cây phản hồi lan truyền (Propagation Cascade Details)"):
        reactions = selected_thread.get("reactions", [])
        if reactions:
            reax_data = []
            for r in reactions[:15]:
                u = r.get("raw_user", r.get("user", {}))
                reax_data.append({
                    "Thời điểm": r.get("created_at", "N/A"),
                    "Tác giả": f"@{u.get('screen_name', 'anon')}",
                    "Xác minh": "Có" if u.get("verified") else "Không",
                    "Followers": f"{u.get('followers_count', 0):,}",
                    "Nội dung phản hồi": r.get("text", "")
                })
            st.dataframe(pd.DataFrame(reax_data), use_container_width=True, hide_index=True)
            if len(reactions) > 15:
                st.caption(f"Đang hiển thị 15 phản hồi tiêu biểu trên tổng số {len(reactions)} bài đăng.")
        else:
            st.info("Luồng hội thoại này không có phản hồi kèm theo.")

    # =========================================================================
    # MỤC 2: SO SÁNH ĐỐI ĐẦU VỚI TTT GỐC (HEAD-TO-HEAD COMPARISON)
    # =========================================================================
    st.subheader("2. So Sánh Đối Đầu Song Song (Head-to-Head: TTT Baseline vs. UC-TTT Proposed)")

    c_ttt, c_uc = st.columns(2)
    ttt = results["ttt_prediction"]
    uc = results["ucttt_prediction"]
    
    with c_ttt:
        st.markdown("""
        <div style="text-align: center; background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #475569; font-size: 17px;">Baseline: TTT Gốc (Wu et al. 2025)</h3>
            <small style="color: #64748b;">Chỉ dùng Văn bản BERT 768-d + Cây lan truyền + BiGRU (KHÔNG CÓ User Credibility)</small>
        </div>
        """, unsafe_allow_html=True)
        
        is_rumour_ttt = ttt["label"] == 1
        label_color_ttt = "#dc2626" if is_rumour_ttt else "#16a34a"
        
        st.markdown(f"""
        <div class="ui-card" style="border-top: 4px solid {label_color_ttt};">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 14px; color: #64748b;">Phán đoán của mô hình:</span>
                <span style="background: {label_color_ttt}20; color: {label_color_ttt}; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 15px;">
                    {ttt['label_name']}
                </span>
            </div>
            <div style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span>Xác suất Tin đồn P(Rumour):</span>
                    <b style="color: #dc2626;">{ttt['prob_rumour'] * 100:.2f}%</b>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 6px; height: 12px; width: 100%; margin-top: 4px;">
                    <div style="background-color: #dc2626; width: {ttt['prob_rumour'] * 100}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            <div style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span>Xác suất Tin xác thực P(Non-rumour):</span>
                    <b style="color: #16a34a;">{ttt['prob_non_rumour'] * 100:.2f}%</b>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 6px; height: 12px; width: 100%; margin-top: 4px;">
                    <div style="background-color: #16a34a; width: {ttt['prob_non_rumour'] * 100}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 14px; font-weight: 600; font-size: 14px;">
                {'[ĐÚNG] Trùng khớp Nhãn Thực tế' if ttt['is_correct'] else ('[SAI] Sai lệch so với Nhãn Thực tế' if ttt['is_correct'] is False else 'Chưa có nhãn đối chứng')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_uc:
        st.markdown("""
        <div style="text-align: center; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
            <h3 style="margin: 0; color: #1d4ed8; font-size: 17px;">Đề xuất: UC-TTT (Nhóm 6)</h3>
            <small style="color: #2563eb;">Tích hợp thêm 12 đặc trưng Độ uy tín Người dùng với Cổng Residual Gated Attention</small>
        </div>
        """, unsafe_allow_html=True)
        
        is_rumour_uc = uc["label"] == 1
        label_color_uc = "#dc2626" if is_rumour_uc else "#16a34a"
        
        st.markdown(f"""
        <div class="ui-card" style="border-top: 4px solid {label_color_uc}; border-color: #93c5fd;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 14px; color: #64748b;">Phán đoán của mô hình:</span>
                <span style="background: {label_color_uc}20; color: {label_color_uc}; padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 15px;">
                    {uc['label_name']}
                </span>
            </div>
            <div style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span>Xác suất Tin đồn P(Rumour):</span>
                    <b style="color: #dc2626;">{uc['prob_rumour'] * 100:.2f}%</b>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 6px; height: 12px; width: 100%; margin-top: 4px;">
                    <div style="background-color: #dc2626; width: {uc['prob_rumour'] * 100}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            <div style="margin: 12px 0;">
                <div style="display: flex; justify-content: space-between; font-size: 13px;">
                    <span>Xác suất Tin xác thực P(Non-rumour):</span>
                    <b style="color: #16a34a;">{uc['prob_non_rumour'] * 100:.2f}%</b>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 6px; height: 12px; width: 100%; margin-top: 4px;">
                    <div style="background-color: #16a34a; width: {uc['prob_non_rumour'] * 100}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            <div style="text-align: center; margin-top: 14px; font-weight: 600; font-size: 14px;">
                {'[ĐÚNG] Trùng khớp Nhãn Thực tế' if uc['is_correct'] else ('[SAI] Sai lệch so với Nhãn Thực tế' if uc['is_correct'] is False else 'Chưa có nhãn đối chứng')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # MỤC 3: TRỌNG SỐ GIẢI THÍCH CỦA USER CREDIBILITY (GATE ALPHA & INTERPRET)
    # =========================================================================
    st.subheader("3. Trọng Số Giải Thích của User Credibility (Gate Alpha & Interpretability)")
    
    interp = results["interpretability"]
    col_w1, col_w2 = st.columns([1, 1])
    
    with col_w1:
        st.markdown(f"""
        <div class="ui-card">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">Tỷ trọng Đóng góp trong Cổng Dung hợp (Residual Gate)</h4>
            <div style="margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px;">
                    <span><b>Độ uy tín Tài khoản (User Credibility):</b></span>
                    <b style="color: #2563eb;">{interp['user_weight_pct']:.1f}%</b>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 6px; height: 14px; width: 100%;">
                    <div style="background-color: #2563eb; width: {interp['user_weight_pct']}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            <div style="margin-top: 15px;">
                <div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 4px;">
                    <span><b>Ngữ nghĩa Văn bản Tweet (Text Semantics):</b></span>
                    <b style="color: #64748b;">{interp['text_weight_pct']:.1f}%</b>
                </div>
                <div style="background-color: #e2e8f0; border-radius: 6px; height: 14px; width: 100%;">
                    <div style="background-color: #64748b; width: {interp['text_weight_pct']}%; height: 100%; border-radius: 6px;"></div>
                </div>
            </div>
            <div class="metric-callout">
                <p style="margin: 0; font-size: 13.5px; color: #334155;">
                    <b>Giải thích cơ chế:</b> {interp['explanation_text']}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_w2:
        st.markdown("""
        <div class="ui-card">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">Bóc tách 12 Đặc trưng Uy tín Người dùng (Features Breakdown)</h4>
        """, unsafe_allow_html=True)
        
        src_raw_u = selected_thread.get("source_tweet", {}).get("raw_user", selected_thread.get("source_tweet", {}).get("user", {}))
        favs = src_raw_u.get("favourites_count", 0) or 0
        lists = src_raw_u.get("listed_count", 0) or 0
        is_default_avatar = src_raw_u.get("default_profile_image", False)
        
        src_created_str = selected_thread.get("source_tweet", {}).get("created_at", "")
        u_created_str = src_raw_u.get("created_at", "")
        t_dt = parse_twitter_datetime(src_created_str)
        u_dt = parse_twitter_datetime(u_created_str)
        age_days = (t_dt - u_dt).total_seconds() / 86400.0 if (t_dt and u_dt and t_dt >= u_dt) else 365.0
        
        st_rate = src["statuses_count"] / (age_days + 1.0)
        fav_rate = favs / (src["statuses_count"] + 1.0)

        feat_df = pd.DataFrame([
            {"Nhóm": "1. Xác minh", "Tên đặc trưng": "Tick xanh xác minh (is_verified)", "Giá trị": "CÓ (1.0)" if src["verified"] else "KHÔNG (0.0)", "Log-Transform": f"{1.0 if src['verified'] else 0.0:.1f}", "Tác động": "Tích cực (Chính danh)" if src["verified"] else "Trung tính"},
            {"Nhóm": "1. Xác minh", "Tên đặc trưng": "Avatar mặc định chưa đổi (has_default_avatar)", "Giá trị": "Mặc định" if is_default_avatar else "Đã đổi avatar", "Log-Transform": f"{1.0 if is_default_avatar else 0.0:.1f}", "Tác động": "Nguy cơ bot" if is_default_avatar else "Tài khoản thực"},
            {"Nhóm": "1. Xác minh", "Tên đặc trưng": "Mô tả tiểu sử Bio (has_description)", "Giá trị": "Có tiểu sử" if src["description"] else "Để trống", "Log-Transform": f"{1.0 if src['description'] else 0.0:.1f}", "Tác động": "Tích cực" if src["description"] else "Cảnh báo"},
            {"Nhóm": "2. Lịch sử", "Tên đặc trưng": "Tuổi thọ tài khoản (log_account_age_days)", "Giá trị": f"{int(age_days):,} ngày", "Log-Transform": f"{math.log1p(max(0.0, float(age_days))):.2f}", "Tác động": "Thâm niên cao" if age_days > 365 else "Mới tạo"},
            {"Nhóm": "2. Quy mô", "Tên đặc trưng": "Số người theo dõi (log_followers)", "Giá trị": f"{src['followers_count']:,}", "Log-Transform": f"{math.log1p(max(0.0, float(src['followers_count']))):.2f}", "Tác động": "Ảnh hưởng lớn" if src['followers_count'] > 50000 else "Bình thường"},
            {"Nhóm": "2. Quy mô", "Tên đặc trưng": "Số tài khoản theo dõi (log_friends)", "Giá trị": f"{src['friends_count']:,}", "Log-Transform": f"{math.log1p(max(0.0, float(src['friends_count']))):.2f}", "Tác động": "Bình thường"},
            {"Nhóm": "2. Quy mô", "Tên đặc trưng": "Tổng số bài đăng (log_statuses)", "Giá trị": f"{src['statuses_count']:,}", "Log-Transform": f"{math.log1p(max(0.0, float(src['statuses_count']))):.2f}", "Tác động": "Hoạt động đều"},
            {"Nhóm": "3. Tương tác", "Tên đặc trưng": "Số bài yêu thích đã bấm (log_favourites)", "Giá trị": f"{favs:,}", "Log-Transform": f"{math.log1p(max(0.0, float(favs))):.2f}", "Tác động": "Người dùng thực" if favs > 20 else "Ít tương tác"},
            {"Nhóm": "3. Tương tác", "Tên đặc trưng": "Số danh sách công khai (log_listed)", "Giá trị": f"{lists:,}", "Log-Transform": f"{math.log1p(max(0.0, float(lists))):.2f}", "Tác động": "Được công nhận" if lists > 10 else "Bình thường"},
            {"Nhóm": "4. Tỷ lệ", "Tên đặc trưng": "Tỷ lệ uy tín (log_reputation_ratio)", "Giá trị": f"{src['reputation_ratio']:.2f}", "Log-Transform": f"{math.log1p(max(0.0, float(src['reputation_ratio']))):.2f}", "Tác động": "Đáng tin cậy" if src['reputation_ratio'] > 5.0 else "Rủi ro"},
            {"Nhóm": "4. Tỷ lệ", "Tên đặc trưng": "Tần suất đăng bài (log_status_rate)", "Giá trị": f"{st_rate:.2f} bài/ngày", "Log-Transform": f"{math.log1p(max(0.0, float(st_rate))):.2f}", "Tác động": "Tự nhiên" if st_rate < 50 else "Nghi vấn spam"},
            {"Nhóm": "4. Tỷ lệ", "Tên đặc trưng": "Tỷ lệ thích / bài (log_favourite_rate)", "Giá trị": f"{fav_rate:.2f} likes/post", "Log-Transform": f"{math.log1p(max(0.0, float(fav_rate))):.2f}", "Tác động": "Bình thường"},
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # =========================================================================
    # MỤC 4: DIỄN BIẾN CẢNH BÁO THEO THỜI GIAN (EARLY DETECTION TIMELINE)
    # =========================================================================
    st.subheader("4. Diễn Biến Cảnh Báo Theo Thời Gian (Early Rumor Detection Curve)")
    st.markdown("""
    Do mô hình cấu trúc cây theo các lát cắt thời gian tích lũy (*Cumulative Subtrees* theo Wu et al. 2025), 
    đồ thị dưới đây thể hiện **Xác suất Tin đồn P(Rumour) biến thiên theo thời gian phản hồi tích lũy** từ khi tweet gốc xuất hiện cho đến khi luồng kết thúc.
    """)
    
    tl_ttt = results["timeline"]["ttt"]
    tl_uc = results["timeline"]["ucttt"]
    
    rows = []
    for i in range(len(tl_uc)):
        t_pt = tl_uc[i]
        t_ttt = tl_ttt[i]
        label = t_pt["elapsed_label"]
        rows.append({"Mốc": label, "Mô hình": "UC-TTT Proposed", "Xác suất Tin đồn (%)": t_pt["prob_rumour"] * 100.0, "Bài đăng": t_pt["cum_posts"]})
        rows.append({"Mốc": label, "Mô hình": "TTT Baseline", "Xác suất Tin đồn (%)": t_ttt["prob_rumour"] * 100.0, "Bài đăng": t_pt["cum_posts"]})
        
    df_long = pd.DataFrame(rows)
    
    c_chart, c_stages = st.columns([2, 1])
    
    with c_chart:
        color_scale = alt.Scale(domain=["UC-TTT Proposed", "TTT Baseline"], range=["#2563eb", "#ea580c"])
        line_chart = alt.Chart(df_long).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X("Mốc:N", sort=None, title="Thời điểm trôi qua (Cumulative Subtrees)"),
            y=alt.Y("Xác suất Tin đồn (%):Q", scale=alt.Scale(domain=[0, 100]), title="Xác suất Tin đồn P(Rumour) %"),
            color=alt.Color("Mô hình:N", scale=color_scale),
            tooltip=["Mốc", "Mô hình", alt.Tooltip("Xác suất Tin đồn (%):Q", format=".1f"), "Bài đăng"]
        ).properties(height=320)
        
        rule = alt.Chart(pd.DataFrame([{"threshold": 50}])).mark_rule(color="#ef4444", strokeDash=[6, 4], strokeWidth=2).encode(
            y="threshold:Q"
        )
        rule_text = alt.Chart(pd.DataFrame([{"y": 53, "text": "Ngưỡng Cảnh báo Tin đồn (50%)"}])).mark_text(
            align="right", color="#ef4444", fontSize=12, fontWeight="bold"
        ).encode(
            x=alt.value(580),
            y="y:Q",
            text="text:N"
        )
        
        final_chart = line_chart + rule + rule_text
        st.altair_chart(final_chart, use_container_width=True)
        
    with c_stages:
        st.markdown("#### Các Mốc Thời Gian Trọng Yếu:")
        for i in range(len(tl_uc)):
            p_uc = tl_uc[i]["prob_rumour"] * 100.0
            p_ttt = tl_ttt[i]["prob_rumour"] * 100.0
            time_lbl = tl_uc[i]["elapsed_label"]
            n_posts = tl_uc[i]["cum_posts"]
            alarm_tag = "[BÁO ĐỘNG] Tin đồn" if p_uc >= 50 else "[AN TOÀN] Chưa phát hiện"
            
            st.markdown(f"""
            <div style="background: #f8fafc; border-left: 3px solid {'#dc2626' if p_uc >= 50 else '#16a34a'}; padding: 8px 12px; margin-bottom: 8px; border-radius: 6px; font-size: 13px;">
                <b>Mốc {time_lbl}</b> ({n_posts} bài đăng):<br/>
                • UC-TTT: <b>{p_uc:.1f}%</b> ({alarm_tag})<br/>
                • TTT Gốc: <b>{p_ttt:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 13px;">
        Đề tài: <b>Detecting the Spread of Misinformation in Social Network</b> | Môn học: Mạng Xã Hội | GVHD: TS. Trần Hưng Nghiệp<br/>
        Nhóm 6: Trần Thiên Phú, Mạc Nguyễn Gia Huy, Võ Tấn Đạt, Nguyễn Thị Thanh Mai, Bùi Phạm Bích Phương.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
