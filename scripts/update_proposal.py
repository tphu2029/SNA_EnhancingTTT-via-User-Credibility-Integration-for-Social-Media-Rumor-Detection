import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os

def style_table(table, col_widths, align_center_cols=[]):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = table._tbl.tblPr
    borders_xml = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        f'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="333333"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="DCDCDC"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders_xml)

    for row_idx, row in enumerate(table.rows):
        is_header = (row_idx == 0)
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        for col_idx, cell in enumerate(row.cells):
            cell.width = Inches(col_widths[col_idx])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            tcPr = cell._tc.get_or_add_tcPr()
            
            tcMar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'<w:top w:w="70" w:type="dxa"/>'
                f'<w:bottom w:w="70" w:type="dxa"/>'
                f'<w:left w:w="100" w:type="dxa"/>'
                f'<w:right w:w="100" w:type="dxa"/>'
                f'</w:tcMar>'
            )
            tcPr.append(tcMar)

            if is_header:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="243342"/>')
                tcPr.append(shd)
            elif row_idx % 2 == 1:
                shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F9FAFB"/>')
                tcPr.append(shd)

            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = 1.05
                if col_idx in align_center_cols:
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name = "Arial"
                    if is_header:
                        r.bold = True
                        r.font.color.rgb = RGBColor(255, 255, 255)
                        r.font.size = Pt(8.0)
                    else:
                        r.font.size = Pt(8.0)

def add_p(doc, text="", bold_prefix=None, space_before=1.0, space_after=2.0, font_size=9.5, italic=False):
    p = doc.add_paragraph(style='normal')
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(font_size)
    if text:
        r_txt = p.add_run(text)
        r_txt.font.name = "Arial"
        r_txt.font.size = Pt(font_size)
        r_txt.italic = italic
    return p

def add_heading_item(doc, text, level=1):
    p = doc.add_paragraph(style='normal')
    p.paragraph_format.space_before = Pt(4.0 if level == 1 else 2.5)
    p.paragraph_format.space_after = Pt(1.5)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(text)
    run.font.name = "Arial"
    run.bold = True
    if level == 1:
        run.font.size = Pt(10.5)
        run.font.color.rgb = RGBColor(18, 42, 78)
    elif level == 2:
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(35, 65, 100)
    return p

def main():
    doc_path = os.path.abspath("[Main Template] Mini Proposal - THN.docx")
    doc = docx.Document(doc_path)

    # Locate '3. Experimental plan (proposal version 2)'
    exp_plan_idx = None
    for i, p in enumerate(doc.paragraphs):
        if "3. Experimental plan (proposal version 2)" in p.text:
            exp_plan_idx = i
            break
            
    if exp_plan_idx is None:
        raise ValueError("Could not find '3. Experimental plan' heading!")

    # Remove forced page break before Appendix 3 if any
    p_before = doc.paragraphs[exp_plan_idx - 1]
    for r in p_before.runs:
        if 'type="page"' in r._r.xml:
            for child in list(r._r):
                if child.tag.endswith('br') and child.get(qn('w:type')) == 'page':
                    r._r.remove(child)

    # Remove everything after P103
    paras_to_remove = doc.paragraphs[exp_plan_idx + 1:]
    for p in paras_to_remove:
        p_elm = p._p
        p_elm.getparent().remove(p_elm)

    # Introductory sentence
    add_p(doc, 
          "Kế hoạch thực nghiệm kiểm chứng 3 câu hỏi nghiên cứu (RQ1–RQ3) và các giả thuyết (H1–H3) theo giao thức Leave-One-Event-Out (LOEO), đảm bảo không rò rỉ dữ liệu và có tính tái lập 100%.", 
          space_before=1.5, space_after=2.5)

    # ==========================================
    # 1. Dataset:
    # ==========================================
    add_heading_item(doc, "1. Dataset:", level=1)
    
    add_heading_item(doc, "Detailed stats table.", level=2)
    add_p(doc, "Dữ liệu benchmark PHEME gồm 6,425 luồng hội thoại (2,402 Rumours / 4,023 Non-rumours) và 105,354 phản hồi trên 9 sự kiện Twitter/X:")
    
    headers = ["Sự kiện (Event)", "Threads", "Rumours", "Non-Rum.", "Tỷ lệ Rum.", "Reactions", "Avg React.", "Avg Depth", "Verified %"]
    data = [
        ["charliehebdo", "2,079", "458", "1,621", "22.0%", "36,189", "17.4", "3.69", "54.7%"],
        ["ferguson", "1,143", "284", "859", "24.8%", "23,032", "20.2", "3.59", "36.8%"],
        ["sydneysiege", "1,221", "522", "699", "42.8%", "22,775", "18.7", "3.90", "63.0%"],
        ["ottawashooting", "890", "470", "420", "52.8%", "11,394", "12.8", "2.74", "67.8%"],
        ["germanwings-crash", "469", "238", "231", "50.7%", "4,020", "8.6", "2.16", "63.1%"],
        ["prince-toronto", "233", "229", "4", "98.3%", "669", "2.9", "1.23", "34.3%"],
        ["putinmissing", "238", "126", "112", "52.9%", "597", "2.5", "1.07", "7.1%"],
        ["gurlitt", "138", "61", "77", "44.2%", "41", "0.3", "0.18", "16.7%"],
        ["ebola-essien", "14", "14", "0", "100.0%", "212", "15.1", "3.07", "21.4%"],
        ["PHEME v5 (5 sự kiện)", "5,802", "1,972", "3,830", "33.99%", "97,410", "16.8", "3.22", "55.7%"],
        ["Full PHEME (9 sự kiện)", "6,425", "2,402", "4,023", "37.38%", "105,354", "16.4", "3.01", "52.2%"],
    ]
    widths = [1.35, 0.55, 0.55, 0.60, 0.65, 0.65, 0.60, 0.60, 0.65]
    t1 = doc.add_table(rows=len(data) + 1, cols=len(headers))
    for c_i, h in enumerate(headers):
        t1.rows[0].cells[c_i].paragraphs[0].text = h
    for r_i, row in enumerate(data):
        for c_i, val in enumerate(row):
            t1.rows[r_i + 1].cells[c_i].paragraphs[0].text = val
    style_table(t1, widths, align_center_cols=[1, 2, 3, 4, 5, 6, 7, 8])
    
    add_heading_item(doc, "Train / validation / test statistics.", level=2)
    add_p(doc, "Giao thức Leave-One-Event-Out (LOEO): Mỗi fold giữ 1 sự kiện làm Test (D_test). Các sự kiện còn lại chia phân tầng (Stratified): 85% Train (D_train) và 15% Validation (D_val) phục vụ Early Stopping.")
    
    headers_loeo = ["Fold", "Sự kiện Test (D_test)", "Số mẫu Test (R / NR)", "Số mẫu Train", "Số mẫu Val"]
    data_loeo = [
        ["Fold 0", "Charlie Hebdo", "2,079 (458 / 1,621)", "3,165", "558"],
        ["Fold 1", "Sydney Siege", "1,221 (522 / 699)", "3,894", "687"],
        ["Fold 2", "Ferguson", "1,143 (284 / 859)", "3,960", "699"],
        ["Fold 3", "Ottawa Shooting", "890 (470 / 420)", "4,175", "737"],
        ["Fold 4", "Germanwings Crash", "469 (238 / 231)", "4,533", "800"],
    ]
    t_loeo = doc.add_table(rows=len(data_loeo) + 1, cols=len(headers_loeo))
    for c_i, h in enumerate(headers_loeo):
        t_loeo.rows[0].cells[c_i].paragraphs[0].text = h
    for r_i, row in enumerate(data_loeo):
        for c_i, val in enumerate(row):
            t_loeo.rows[r_i + 1].cells[c_i].paragraphs[0].text = val
    style_table(t_loeo, [0.75, 1.8, 1.8, 1.3, 1.15], align_center_cols=[0, 2, 3, 4])

    add_heading_item(doc, "Detailed data pipeline.", level=2)
    add_p(doc, "Pipeline gồm 5 bước: (1) Parser & Graph Builder: Đọc structure.json, dựng cây phân cấp T=(V, E) và trích nhãn; (2) Offline Text Embedding: Mã hóa văn bản tweet bằng all-MiniLM-L6-v2 (d_text=384); (3) User Credibility Extraction: Trích xuất 12 đặc trưng uy tín gồm Profile (verified, avatar, bio, age), Counts (followers, friends, statuses, favourites, listed qua log1p) và Ratios (reputation, status rate, favourite rate); (4) Temporal Window Slicing: Phân đoạn thời gian phát tán thành K=4 lát cắt; (5) Dynamic Batch Collation: Đệm zero-padding theo N_max, sinh Key Padding Mask và Depth Encoding.")

    # ==========================================
    # 2. Evaluation Protocol:
    # ==========================================
    add_heading_item(doc, "2. Evaluation Protocol:", level=1)
    
    add_heading_item(doc, "Detailed evaluation pipeline, evaluation data.", level=2)
    add_p(doc, "Đánh giá chéo LOEO trên 2 bộ dữ liệu: PHEME v5 (5 sự kiện chuẩn, đối chiếu Wu et al. 2025) và Full PHEME (toàn bộ 9 sự kiện). Huấn luyện tối đa 35 epochs với EarlyStopping(patience=8) theo Validation Macro-F1. Hàm mất mát sử dụng Cross-Entropy có trọng số cân bằng lớp nghịch đảo tần suất w_c = N_total / (2 * N_c).")

    add_heading_item(doc, "Evaluation metrics.", level=2)
    add_p(doc, "Hệ thống đo lường: (1) Pooled Metrics: Pooled Accuracy, Pooled Macro-F1, Precision, Recall tính trên ma trận nhầm lẫn gộp toàn bộ các test fold (metric chính); (2) Mean LOEO Metrics: Trung bình và độ lệch chuẩn mu +- sigma Macro-F1 qua các fold đo lường độ ổn định chuyển giao; (3) Kiểm định thống kê: Paired t-test và Wilcoxon signed-rank test với ngưỡng ý nghĩa p < 0.05.")

    # ==========================================
    # 3. Baseline:
    # ==========================================
    add_heading_item(doc, "3. Baseline:", level=1)
    
    add_p(doc, "TF-IDF (2,000 từ vựng), hồi quy Logistic L2. Biến thể B1a (Source-only) đạt Pooled Acc 71.77%, Pooled F1 67.56% (Mean F1 56.17%); B1b (Source+Reactions) đạt Pooled Acc 70.60%, Pooled F1 65.67% (Mean F1 52.85%).", bold_prefix="Baseline 1 — TF-IDF + Logistic Regression (Content-Only): ")
    
    add_p(doc, "8 đặc trưng topo hình thái cây (Size, Depth, Degree, Virality, Velocity), Random Forest 100 cây. Đạt Pooled Acc 56.62%, Pooled F1 55.62% (Mean F1 46.12%).", bold_prefix="Baseline 2 — Propagation Topological Features + Random Forest (Structure-Only): ")
    
    add_p(doc, "Tái hiện mô hình học sâu SOTA theo đúng cấu trúc bài báo gốc: Biểu diễn văn bản chiếu lên không gian ẩn d_model = 768, Sinusoidal Depth Encoding, Tree Transformer Multi-head Attention gồm 6 layers và 12 heads (feedforward dim 2048), chia K = 8 lát cắt thời gian qua Bidirectional Temporal GRU (hidden dim 128) và phân loại bằng tầng FC ẩn 600 (GELU activation). Huấn luyện bằng AdamW (lr = 4.2e-5, weight decay 1e-4, dropout 0.2). Đây là mốc đối chứng chính thức (Benchmarking Target) nhằm kiểm chứng giả thuyết H1, với mục tiêu tái hiện bám sát kết quả công bố trong bài báo gốc trên PHEME v5 (Accuracy 75.84% và Macro-F1 71.98%).", bold_prefix="Baseline 3 — Original Temporal Tree Transformer (Wu et al. PLOS ONE 2025): ")

    # ==========================================
    # 4. Proposed Method:
    # ==========================================
    add_heading_item(doc, "4. Proposed Method:", level=1)
    
    add_heading_item(doc, "Main idea, description.", level=2)
    add_p(doc, "Mô hình UC-TTT (User-Credibility-Aware Temporal Tree Transformer) tích hợp độ tin cậy của tác giả vào biểu diễn từng nút trong cây lan truyền nhằm giải quyết thiếu sót của TTT gốc. Kiến trúc gồm 4 khối:")
    add_p(doc, "1. User Credibility Encoder (UCE): MLP 2 tầng (Linear 12 -> 128 -> d_model với GELU, Dropout 0.2) mã hóa vector uy tín 12 chiều thành biểu diễn ẩn e_i in R^{d_model} (với d_model = 768).")
    add_p(doc, "2. Gated Attention Fusion: Cổng động dung hợp giữa biểu diễn nội dung + độ sâu (h_i) và vector uy tín (e_i): g_i = sigmoid(W_g [h_i || e_i] + b_g), z_i = g_i * h_i + (1 - g_i) * e_i, giúp tự động điều tiết mức độ tin tưởng bài đăng dựa trên độ uy tín của người dùng.")
    add_p(doc, "3. Tree Transformer & Temporal BiGRU: Các lớp Multi-head Self-Attention (6 layers, 12 heads) mô hình hóa quan hệ tranh luận giữa các nút; tiếp theo chia K = 8 lát cắt thời gian qua Mean-pooling và BiGRU để trích xuất vector động lực học thời gian h_temporal.")
    add_p(doc, "4. Joint Classification Head: Ghép nối vector nút gốc h_root và vector thời gian h_temporal qua tầng phân loại 2 lớp (FC 600) dự đoán nhãn Rumour vs. Non-rumour.")

    # ==========================================
    # 5. Ablation:
    # ==========================================
    add_heading_item(doc, "5. Ablation:", level=1)
    
    add_p(doc, "Đánh giá đóng góp từng nhóm đặc trưng trên 5 folds PHEME v5: Full 12 đặc trưng (chuẩn); loại bỏ từng nhóm (w/o Profile, w/o Counts, w/o Ratios); kích hoạt đơn lẻ (Profile-only, Counts-only, Ratios-only); và Source User Only vs. Cascade Users để đo Delta Macro-F1 và xác định nhóm đặc trưng then chốt (kiểm chứng H2/RQ2).", bold_prefix="Ablation 1 (Feature-level Ablations): ")
    
    add_p(doc, "Đánh giá vai trò kiến trúc: (1) w/o Temporal GRU (cây tĩnh); (2) w/o Tree Transformer (chuỗi tuần tự); (3) So sánh 4 chiến lược fusion (Gated, Residual Gate, Concat, Additive).", bold_prefix="Ablation 2 (Model & Architecture-level Ablations): ")

    # ==========================================
    # 6. Experiment specifications:
    # ==========================================
    add_heading_item(doc, "6. Experiment specifications:", level=1)
    
    add_heading_item(doc, "Experiment 1:", level=2)
    add_p(doc, "Kế hoạch đánh giá hiệu năng tổng thể và kiểm chứng Giả thuyết H1 (User credibility giúp cải thiện Macro-F1 so với TTT gốc, trả lời RQ1). Ma trận đối chuẩn và mục tiêu kỳ vọng:")
    
    headers_exp1 = ["Mô hình Đối chứng", "Không gian Đặc trưng", "Giao thức", "Pooled Acc (Kỳ vọng)", "Pooled F1 (Kỳ vọng)", "Mục đích đối chuẩn"]
    data_exp1 = [
        ["B1a: TF-IDF + LR (Source)", "Text only", "9-Fold LOEO", "Baseline cơ sở", "Baseline cơ sở", "Đo lường giới hạn từ vựng tĩnh"],
        ["B1b: TF-IDF + LR (Full)", "Text only", "9-Fold LOEO", "Baseline cơ sở", "Baseline cơ sở", "Đo lường từ vựng mở rộng"],
        ["B2: Prop. Features + RF", "Graph topology", "9-Fold LOEO", "Baseline cơ sở", "Baseline cơ sở", "Đo lường giới hạn cấu trúc đồ thị"],
        ["B3: Original TTT (Wu et al. 2025)", "Text + Depth + Time", "5-Fold LOEO", "75.84% (Công bố)", "71.98% (Công bố)", "Mốc SOTA chuẩn của bài báo gốc"],
        ["UC-TTT (Proposed - Gated)", "Text + Depth + Time + User", "5-Fold LOEO", "> 75.84% (Mục tiêu)", "> 72.00% (Mục tiêu)", "Khẳng định đóng góp của User Credibility"],
        ["UC-TTT (Proposed - Gated)", "Text + Depth + Time + User", "9-Fold LOEO", "Đánh giá mở rộng", "Đánh giá mở rộng", "Kiểm định độ bền vững trên 9 sự kiện"],
    ]
    t_exp1 = doc.add_table(rows=len(data_exp1) + 1, cols=len(headers_exp1))
    for c_i, h in enumerate(headers_exp1):
        t_exp1.rows[0].cells[c_i].paragraphs[0].text = h
    for r_i, row in enumerate(data_exp1):
        for c_i, val in enumerate(row):
            t_exp1.rows[r_i + 1].cells[c_i].paragraphs[0].text = val
    style_table(t_exp1, [1.8, 1.35, 1.0, 1.15, 1.15, 1.55], align_center_cols=[2, 3, 4])

    add_p(doc, "Mục tiêu kiểm chứng H1: Đề tài kỳ vọng mô hình UC-TTT vượt mốc chuẩn của bài báo gốc Wu et al. 2025 (Acc > 75.84% và Macro-F1 > 72.00%), chứng minh thông tin độ tin cậy người dùng cung cấp tín hiệu bổ sung hữu hiệu cho bài toán phát hiện tin đồn.")

    add_heading_item(doc, "Experiment 2:", level=2)
    add_p(doc, "Kế hoạch bóc tách đặc trưng (H2/RQ2) và kiểm chứng tính không đồng nhất giữa các sự kiện (H3/RQ3). Số liệu công bố của bài báo gốc Wu et al. 2025 (Table 3) làm mốc đối chứng:")
    
    headers_events = ["Sự kiện Test (LOEO)", "Bài báo Wu et al. 2025 (Acc / F1)", "Kỳ vọng UC-TTT", "Phân tích giả thuyết H3"]
    data_events = [
        ["Ottawa Shooting (OS)", "70.25% / 69.54%", "Cải thiện F1", "Sự kiện khủng bố; tỷ lệ verified cao (67.8%)"],
        ["Charlie Hebdo (CH)", "84.15% / 78.25%", "Cải thiện F1", "Nhiều cơ quan báo chí tham gia đính chính"],
        ["Sydney Siege (SS)", "74.83% / 74.33%", "Tăng mạnh F1", "Tin giật gân; kỳ vọng lọc tốt tài khoản kích động"],
        ["Ferguson (FG)", "77.52% / 65.33%", "Tăng mạnh F1 (Trọng tâm)", "Xung đột xã hội; nhiều tài khoản nặc danh/bot"],
        ["Germanwings Crash (GC)", "72.46% / 72.45%", "Cải thiện ổn định", "Tai nạn hàng không; ít yếu tố thao túng dư luận"],
    ]
    t_ev = doc.add_table(rows=len(data_events) + 1, cols=len(headers_events))
    for c_i, h in enumerate(headers_events):
        t_ev.rows[0].cells[c_i].paragraphs[0].text = h
    for r_i, row in enumerate(data_events):
        for c_i, val in enumerate(row):
            t_ev.rows[r_i + 1].cells[c_i].paragraphs[0].text = val
    style_table(t_ev, [1.6, 1.7, 1.3, 2.4], align_center_cols=[1, 2])

    add_p(doc, "Cơ sở kiểm chứng H3: Mức độ đóng góp của User Credibility được kỳ vọng sẽ không đồng đều giữa các sự kiện. Cụ thể, các sự kiện nhạy cảm chính trị/xã hội có nhiều bot tham gia tung tin thất thiệt (như Ferguson) sẽ có mức cải thiện F1 bứt phá rõ rệt hơn so với các sự kiện tai nạn thuần túy.")
    
    add_p(doc, "Cấu hình siêu tham số đồng bộ bài báo gốc Wu et al. (PLOS ONE 2025): Biểu diễn ẩn d_model = 768, số attention heads nhead = 12, số lớp Tree Transformer = 6, số lát cắt thời gian K = 8, Temporal BiGRU hidden = 128, FC classifier hidden = 600, Dropout = 0.2, Optimizer AdamW (lr = 4.2e-5, weight_decay = 1e-4), Batch size = 32 luồng cây, EarlyStopping patience = 8 theo Validation Macro-F1, Seed = 42 cố định trên GPU CUDA.")

    try:
        doc.save(doc_path)
        print(f"Successfully saved updated compact proposal to: {doc_path}")
    except PermissionError:
        compact_path = doc_path.replace(".docx", "_compact.docx")
        doc.save(compact_path)
        print(f"File was locked by Word. Successfully saved to: {compact_path}")

if __name__ == "__main__":
    main()
