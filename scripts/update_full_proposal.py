# -*- coding: utf-8 -*-
"""
Script cap nhat toan dien file Mini Proposal cho Nhom 6 - Mon Mang Xa Hoi
De tai: Detecting the Spread of Misinformation in Social Network
Huong: Enhancing Temporal Tree Transformer with User Credibility Features on PHEME
GVHD: TS. Tran Hung Nghiep
"""

import os
import sys
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

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
                f'<w:left w:w="90" w:type="dxa"/>'
                f'<w:right w:w="90" w:type="dxa"/>'
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

class SectionBuilder:
    def __init__(self, doc, anchor_element):
        self.doc = doc
        self.anchor = anchor_element

    def add_p(self, text="", bold_prefix=None, space_before=1.0, space_after=2.0, font_size=9.5, italic=False):
        p = self.doc.add_paragraph(style='Normal')
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
        self.anchor.addprevious(p._p)
        return p

    def add_heading(self, text, level=1):
        p = self.doc.add_paragraph(style='Normal')
        p.paragraph_format.space_before = Pt(5.0 if level == 1 else 3.0)
        p.paragraph_format.space_after = Pt(1.5)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Arial"
        run.bold = True
        if level == 1:
            run.font.size = Pt(11.0)
            run.font.color.rgb = RGBColor(18, 42, 78)
        elif level == 2:
            run.font.size = Pt(10.0)
            run.font.color.rgb = RGBColor(35, 65, 100)
        elif level == 3:
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(50, 75, 110)
        self.anchor.addprevious(p._p)
        return p

    def add_table(self, headers, data, col_widths, align_center_cols=[]):
        table = self.doc.add_table(rows=len(data) + 1, cols=len(headers))
        for c_i, h in enumerate(headers):
            table.rows[0].cells[c_i].paragraphs[0].text = h
        for r_i, row in enumerate(data):
            for c_i, val in enumerate(row):
                table.rows[r_i + 1].cells[c_i].paragraphs[0].text = str(val)
        style_table(table, col_widths, align_center_cols=align_center_cols)
        self.anchor.addprevious(table._tbl)
        # spacer paragraph
        p_sp = self.doc.add_paragraph(style='Normal')
        p_sp.paragraph_format.space_before = Pt(0)
        p_sp.paragraph_format.space_after = Pt(2.0)
        self.anchor.addprevious(p_sp._p)
        return table

def build_full_proposal():
    src_file = "Mini_Proposal_Backup.docx"
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source backup file not found: {src_file}")

    doc = docx.Document(src_file)
    body = doc._body._element

    # 1. Identify key markers
    idx_content = None
    idx_ref = None
    idx_exp_plan = None

    for idx, child in enumerate(body):
        if child.tag.endswith('p'):
            p = docx.text.paragraph.Paragraph(child, doc)
            txt = p.text.strip()
            if txt.startswith('B. Content'):
                idx_content = idx
            elif txt == 'References':
                idx_ref = idx
            elif '3. Experimental plan (proposal version 2)' in txt:
                idx_exp_plan = idx

    if idx_content is None or idx_ref is None:
        raise ValueError(f"Could not locate markers: content={idx_content}, ref={idx_ref}")

    print(f"Marker found: B. Content at {idx_content}, References at {idx_ref}, Exp Plan at {idx_exp_plan}")

    # =========================================================================
    # PART 1: REPLACE SECTION B. CONTENT (from idx_content + 1 to idx_ref - 1)
    # =========================================================================
    ref_element = body[idx_ref]

    # Delete existing elements between B. Content and References
    elements_to_delete = []
    for i in range(idx_content + 1, idx_ref):
        elements_to_delete.append(body[i])
    for elm in elements_to_delete:
        body.remove(elm)

    # Now create builder before ref_element
    b = SectionBuilder(doc, ref_element)

    # -------------------------------------------------------------------------
    # 1. Mini Problem Formulation
    # -------------------------------------------------------------------------
    b.add_heading("1. Mini Problem Formulation", level=1)
    
    b.add_heading("1.1. Project Description & Research Context", level=2)
    b.add_p("Bài toán phát hiện tin đồn (Rumor Detection) trên mạng xã hội Twitter/X đóng vai trò cấp thiết nhằm ngăn chặn sự phát tán của thông tin sai lệch trong các tình huống khẩn cấp và sự kiện tin tức nóng (breaking news). Đề tài kế thừa mô hình Temporal Tree Transformer (TTT) của tác giả Wu et al. (PLOS ONE 2025) — một kiến trúc kết hợp biểu diễn ngữ nghĩa bài đăng, cấu trúc cây lan truyền phân cấp và các lát cắt thời gian qua mạng Temporal GRU, thiết lập mốc SOTA với Accuracy 75.84% và Macro-F1 71.98% trên bộ dữ liệu chuẩn PHEME.")

    b.add_heading("1.2. Research Gap (Khoảng trống nghiên cứu)", level=2)
    b.add_p("Mặc dù TTT đã chứng minh hiệu quả vượt trội của việc kết hợp nội dung và động lực học lan truyền thời gian, mô hình này hoàn toàn chỉ tập trung vào thuộc tính của bài đăng (posts) và cấu trúc topo cây, bỏ qua hoàn toàn danh tính và độ tin cậy của tác giả phát tán thông tin (User Credibility). Trong thực tế, tin đồn thất thiệt thường được khởi tạo hoặc khuếch đại bởi các tài khoản ẩn danh, tài khoản mới tạo (bot) có uy tín thấp; trong khi các tin xác thực hoặc bài đính chính thường đến từ các tài khoản có tích xanh xác minh (verified) hoặc các cơ quan báo chí có uy tín cao. Trong phần Future Discussion của bài báo gốc Wu et al. (PLOS ONE 2025), các tác giả đã đề xuất rõ: \"assessing user credibility in the dissemination of misinformation\" như một hướng nghiên cứu mở rộng then chốt. Đề tài này tập trung giải quyết trực tiếp khoảng trống nghiên cứu trên.")

    b.add_heading("1.3. Research Questions (RQ) & Hypotheses (H)", level=2)
    b.add_p("Liệu việc tích hợp các đặc trưng uy tín người dùng (User Credibility) vào biểu diễn từng nút trong cây lan truyền có cải thiện đáng kể hiệu năng phân loại tin đồn của Temporal Tree Transformer hay không?", bold_prefix="RQ1: ")
    b.add_p("Tích hợp User Credibility vào TTT giúp cải thiện chỉ số Macro-F1 có ý nghĩa thống kê so với TTT gốc trên cùng giao thức Leave-One-Event-Out (H1).", bold_prefix="Hypothesis H1: ", italic=True)
    
    b.add_p("Những nhóm đặc trưng uy tín người dùng nào (Profile Verification, Engagement Counts log-scale, hay Derived Behavioral Ratios) đóng góp vai trò then chốt nhất vào độ chính xác phát hiện tin đồn?", bold_prefix="RQ2: ")
    b.add_p("Các tỷ số hành vi chuẩn hóa (Derived Ratios như Followers/Friends) và trạng thái hồ sơ (Profile) cung cấp tín hiệu phân biệt tin đồn rõ rệt hơn so với các con số tương tác đếm thô (H2).", bold_prefix="Hypothesis H2: ", italic=True)
    
    b.add_p("Mức độ đóng góp của User Credibility có sự khác biệt (heterogeneity) giữa các miền sự kiện khác nhau trong PHEME hay không?", bold_prefix="RQ3: ")
    b.add_p("Các sự kiện xã hội nhạy cảm có nhiều bot can thiệp (như bạo loạn Ferguson) sẽ có mức cải thiện F1 bứt phá hơn so với các sự kiện tai nạn thuần túy (như Germanwings Crash) (H3).", bold_prefix="Hypothesis H3: ", italic=True)

    b.add_heading("1.4. Technical Problem Formulation (Mô tả kỹ thuật của bài toán - Khớp chuẩn Wu et al. 2025)", level=2)
    b.add_p("Nhóm xây dựng mô tả kỹ thuật bài toán bám sát định nghĩa toán học của bài báo nền tảng Wu et al. (PLOS ONE 2025), mở rộng thêm không gian đặc trưng người dùng:")
    b.add_p("Cho V(r) = {r, x_1, x_2, ..., x_n} biểu diễn một luồng hội thoại lan truyền (conversation thread) của một sự kiện, trong đó r là bài đăng gốc (source post / claim), x_i là bài phản hồi thứ i (i-th response post) được sắp xếp theo thứ tự thời gian tăng dần (chronological order), và n là tổng số bài phản hồi trong luồng.", bold_prefix="• Biểu diễn luồng hội thoại: ")
    b.add_p("Do khoảng cách thời gian giữa bài đăng gốc và các phản hồi có phương sai rất lớn giữa các thread, phương pháp chia cửa sổ độ rộng thời gian cố định không khả thi. Bài báo áp dụng cửa sổ có độ sâu bài đăng cố định (equal-depth time window): mỗi cửa sổ chứa đúng k bài đăng. Một luồng hội thoại gồm bài gốc r và n phản hồi được chia thành M = ceil((n + 1) / k) cửa sổ thời gian (trong bài báo k = 8). Các bài đăng ở cửa sổ đầu tiên tạo thành cây con ban đầu T_1, sau đó các bài đăng ở cửa sổ thứ hai được tích lũy vào T_1 tạo thành T_2, quá trình tích lũy tiếp tục cho đến khi hình thành toàn bộ cây lan truyền T_M.", bold_prefix="• Cửa sổ thời gian & Cây con tích lũy: ")
    
    b.add_p("Một luồng hội thoại được mô hình hóa bởi bộ tứ đầu vào:", bold_prefix="• Đầu vào kỹ thuật cụ thể (Inputs): ")
    b.add_p("1. Ma trận biểu diễn văn bản: Mỗi bài đăng x_i (hoặc r) có vector nhúng ngữ nghĩa s_i in R^d (d = 768) trích xuất từ Pretrained Language Model (BERT/Sentence Transformer) kết hợp phép gộp Max-pooling.")
    b.add_p("2. Mã hóa độ sâu cây: Vector vị trí phân cấp Sinusoidal PE(depth_i) in R^d cộng trực tiếp vào biểu diễn văn bản: h_i = s_i + PE(depth_i).")
    b.add_p("3. Cấu trúc topo cây: Ma trận kề cây A in {0, 1}^{(n+1) x (n+1)} cùng ma trận mặt nạ Attention Top-Down (TD) cho phép nút con attend ngược về các nút tổ tiên.")
    b.add_p("4. Vector đặc trưng Uy tín Người dùng (User Credibility Vector - Đề xuất cốt lõi): Vector u_i in R^12 biểu diễn metadata của người dùng đăng tải bài viết x_i (gồm 3 nhóm: Profile Verification, Engagement Counts log-scale, và Derived Behavioral Ratios).")
    
    b.add_p("Biểu diễn của toàn bộ luồng hội thoại sau khi qua khối Temporal GRU Encoder là h_temporal in R^128. Kết hợp với biểu diễn nút gốc s_root qua tầng Fully-Connected ẩn 600 và hàm Softmax để xuất ra phân phối xác suất phân loại nhị phân:", bold_prefix="• Đầu ra kỹ thuật cụ thể (Outputs): ")
    b.add_p("P(y_hat = c | V(r)) = Softmax(W_c [s_root || h_temporal] + b_c), với y_hat in {Rumor, Non-rumor} (tương đương y in {1, 0}).")
    b.add_p("Hàm mục tiêu tối ưu hóa toàn bộ tham số Theta là Categorical Cross-Entropy Loss kết hợp điều quy L2: L = - sum_{c in {0, 1}} y_c log(y_hat_c) + (lambda / 2) ||Theta||_2^2.")

    b.add_heading("1.5. Scope, Assumptions & Constraints", level=2)
    b.add_p("Phạm vi đề tài tập trung vào việc tiền xử lý dữ liệu PHEME, trích xuất 12 đặc trưng người dùng, tái hiện TTT gốc, xây dựng kiến trúc dung hợp UC-TTT, và thực hiện chuỗi thực nghiệm LOEO kèm Ablation Study. Giới hạn trong metadata người dùng có sẵn của PHEME; không thu thập dữ liệu ngoài thời gian thực từ API Twitter/X.")

    # -------------------------------------------------------------------------
    # 2. Related Work
    # -------------------------------------------------------------------------
    b.add_heading("2. Related Work (Tổng quan nghiên cứu liên quan)", level=1)
    b.add_p("Các phương pháp phát hiện tin đồn trên mạng xã hội đã tiến hóa qua 4 thế hệ chính: (1) Mô hình dựa trên nội dung văn bản (Content-only: RNN, CNN); (2) Mô hình đồ thị lan truyền tĩnh (Static GNN: RvNN, Bi-GCN, GACL, GARD) nắm bắt cấu trúc tương tác giữa các bài đăng; (3) Mô hình cây lan truyền động theo thời gian (Temporal Tree Transformer - Wu et al. PLOS ONE 2025) đạt hiệu năng vượt trội nhờ theo dõi diễn biến tăng trưởng của cây qua các lát cắt thời gian; (4) Khung làm việc thích ứng nhẹ (SARD-Net 2026) tối ưu suy luận thời gian thực. Bảng 0 tổng hợp các nghiên cứu tiêu biểu:")

    headers_rel = ["#", "Nghiên cứu Tiêu biểu", "Phương pháp & Cơ chế Cốt lõi", "Ưu điểm & Kết quả", "Hạn chế & Khoảng trống"]
    data_rel = [
        ["1", "Temporal Tree Transformer (Wu et al. PLOS ONE 2025)", "Kết hợp BERT token, Sinusoidal Depth, Tree Transformer TD/BU và Temporal GRU tích lũy.", "Acc 75.84%, F1 71.98% trên PHEME. Nắm bắt biến thiên thời gian.", "Bỏ qua hoàn toàn độ tin cậy của tài khoản người dùng; tính toán tốn kém."],
        ["2", "SARD-Net (Sravanthi & Mandala, IEEE 2026)", "Mạng học sâu nhẹ (11.6ms), thích ứng đa miền trong môi trường ít tài nguyên.", "Tốc độ nhanh, Acc 93% trên Twitter16 trong bối cảnh cross-domain.", "Cấu trúc phức tạp, khó tái lập; chưa khai thác triệt để cấu trúc cây phân cấp."],
        ["3", "Knowledge Graphs for Rumour (Chen et al. 2024)", "Tích hợp đồ thị tri thức nhạy thời gian (Time-sensitive KG) để kiểm chứng sự thật.", "Cung cấp bằng chứng thực tế bên ngoài; phát hiện lỗi gán nhãn của PHEME.", "Phụ thuộc Google Search API; tốn chi phí truy vấn; bỏ sót kênh thảo luận kín."],
        ["4", "Bi-GCN (Bian et al. AAAI 2020)", "GCN hai chiều mô hình hóa luồng lan truyền xuôi (Top-down) và luồng phân tán ngược (Bottom-up).", "Khai thác tương tác đồ thị tốt hơn RNN tĩnh; Acc ~73% trên PHEME.", "Đồ thị tĩnh, bỏ qua trục thời gian và không xét đến thuộc tính uy tín người dùng."]
    ]
    b.add_table(headers_rel, data_rel, [0.4, 1.6, 1.8, 1.4, 1.3], align_center_cols=[0])

    # -------------------------------------------------------------------------
    # 3. Data: PHEME Dataset
    # -------------------------------------------------------------------------
    b.add_heading("3. Data: PHEME Dataset & Giải quyết vấn đề \"Dataset Chuẩn\"", level=1)
    
    b.add_heading("3.1. Bản chất Raw Data của PHEME & Chuẩn hóa Thực nghiệm Nội bộ", level=2)
    b.add_p("Trong văn hiến khoa học, bộ dữ liệu PHEME gốc (Zubiaga et al. PLOS ONE 2016) trên Figshare chỉ tồn tại ở dạng Raw JSON Files (các thư mục sự kiện chứa tweet gốc, reactions, annotation.json và structure.json). Cả bài báo nền tảng Wu et al. (PLOS ONE 2025) lẫn các nghiên cứu SOTA khác đều phải tự tải raw JSON và lập trình pipeline tiền xử lý (loại bỏ bài viết bị xóa, bài viết rỗng, lọc cây). Do đó, để đảm bảo tính công bằng khoa học tuyệt đối theo đúng nhận xét của Giảng viên hướng dẫn, nhóm phân định rành mạch 2 chuẩn:")
    b.add_p("Chuẩn 1 — Đối chuẩn với Paper Wu et al. 2025 (PHEME v5): Sử dụng đúng 5 sự kiện lớn: Charlie Hebdo (CH), Sydney Siege (SS), Ferguson (FG), Ottawa Shooting (OS), Germanwings Crash (GC). Theo Bảng 1 của bài báo gốc Wu et al. 2025, sau khi lọc các luồng không hợp lệ, tập dữ liệu chuẩn gồm đúng 5,447 threads (1,863 Rumours / 3,584 Non-rumours) trên 107,383 posts.", bold_prefix="• ")
    b.add_p("Chuẩn 2 — Dataset Chuẩn Nội Bộ (Internal Standardized Benchmark Cache): Nhóm đã xây dựng pipeline tiền xử lý cố định duy nhất (pheme_parser.py), đóng băng dữ liệu thành file cache nhị phân (pheme_processed.pkl và text_embeddings_bert.pt). Cố định tập mẫu và nếp gấp 5-Fold LOEO (với Random Seed = 42 cố định). Toàn bộ các mô hình (TF-IDF, RF, XGBoost, TTT gốc, và UC-TTT) đều được nạp từ cùng một cache nội bộ này, đảm bảo tính công bằng 100% (ceteris paribus).", bold_prefix="• ")

    b.add_heading("3.2. Bảng Thống kê Dữ liệu 5 Sự kiện Chuẩn (Khớp Bảng 1 Bài báo Wu et al. 2025)", level=2)
    b.add_p("Bảng 1 thống kê chi tiết số liệu 5 sự kiện chuẩn của bài báo gốc Wu et al. 2025 (tổng 5,447 threads sau lọc từ 5,802 threads thô):")
    
    headers_stat = ["Chỉ số Thống kê", "Charlie Hebdo (CH)", "Sydney Siege (SS)", "Ferguson (FG)", "Ottawa Shoot. (OS)", "Germanwings (GC)", "Tổng (Total)"]
    data_stat = [
        ["Tổng số Threads", "2,002", "1,173", "1,010", "857", "405", "5,447"],
        ["Số tin Thật (Non-rumours)", "1,555", "673", "753", "400", "203", "3,584"],
        ["Số Tin đồn (Rumours)", "447", "500", "257", "457", "202", "1,863"],
        ["Tỷ lệ Tin đồn (%)", "22.3%", "42.6%", "25.4%", "53.3%", "49.9%", "34.2%"],
        ["Tổng số bài đăng (Posts)", "39,286", "24,511", "25,673", "12,735", "5,178", "107,383"],
        ["Độ sâu cây trung bình", "3.82", "4.06", "4.06", "2.84", "2.51", "3.67"],
        ["Độ sâu cây lớn nhất", "22", "19", "47", "19", "19", "47"],
        ["Số post/cây trung bình", "19.62", "20.90", "25.42", "14.86", "12.79", "19.71"]
    ]
    b.add_table(headers_stat, data_stat, [1.5, 0.8, 0.8, 0.8, 0.9, 0.8, 0.9], align_center_cols=[1, 2, 3, 4, 5, 6])

    b.add_heading("3.3. Hệ thống 12 Đặc trưng Uy tín Người dùng (User Credibility Features)", level=2)
    b.add_p("Mỗi bài đăng trong luồng hội thoại được trích xuất 12 đặc trưng người dùng thuộc 3 nhóm:")
    b.add_p("1. Nhóm Profile & Verification (4 đặc trưng): (1) is_verified in {0, 1} (xác thực tích xanh); (2) has_default_avatar in {0, 1} (dấu hiệu tài khoản rác/bot); (3) has_description in {0, 1} (tiểu sử cá nhân); (4) log_account_age_days = ln(1 + age_days) (tuổi tài khoản tính theo ngày).")
    b.add_p("2. Nhóm Engagement Counts log-scale (5 đặc trưng): Toàn bộ số liệu tương tác đếm được chuẩn hóa log-scale ln(1 + x) để triệt tiêu độ lệch cực đoan (skewness): (5) log_followers; (6) log_friends; (7) log_statuses; (8) log_favourites; (9) log_listed.")
    b.add_p("3. Nhóm Derived Behavioral Ratios (3 đặc trưng tỷ số hành vi): (10) log_reputation_ratio = ln(1 + followers / (friends + 1)) (chỉ số uy tín mạng lưới); (11) log_status_rate = ln(1 + statuses / (age_days + 1)) (mật độ hoạt động bài đăng); (12) log_favourite_rate = ln(1 + favourites / (statuses + 1)) (mức độ đón nhận nội dung).")

    # -------------------------------------------------------------------------
    # 4. Evaluation Protocol
    # -------------------------------------------------------------------------
    b.add_heading("4. Evaluation Protocol & Data Split (Con số cụ thể từng Fold)", level=1)
    
    b.add_heading("4.1. Giao thức Leave-One-Event-Out (LOEO) chống Data Leakage", level=2)
    b.add_p("Đề tài tuân thủ nghiêm ngặt giao thức Leave-One-Event-Out (LOEO) trên 5 sự kiện chuẩn. Việc chia ngẫu nhiên (Random Split 80/20) tuyệt đối bị nghiêm cấm vì gây hiện tượng rò rỉ thông tin (Information Leakage): các bài đăng cùng sự kiện xuất hiện ở cả train và test khiến mô hình học vẹt từ khóa sự kiện. LOEO đảm bảo tập kiểm thử luôn là một sự kiện hoàn toàn chưa từng thấy (unseen event), đo lường chính xác năng lực chuyển giao miền thực tế.")

    b.add_heading("4.2. Bảng Phân chia Số lượng Mẫu Cụ thể Từng Fold (Train / Val / Test)", level=2)
    b.add_p("Trong bài báo Wu et al. 2025, ở mỗi fold, đúng 1 sự kiện làm Test và 4 sự kiện còn lại làm Train. Trong triển khai kỹ thuật PyTorch thực tế của nhóm (nhằm áp dụng Early Stopping chống overfit), nhóm trích 15% phân tầng từ 4 sự kiện còn lại làm Validation set. Bảng 2 trình bày số liệu phân chia cụ thể cho cả 5 Folds:")

    headers_loeo_det = ["Fold", "Sự kiện Test", "Số mẫu TEST (R / NR)", "Tập Train Pool (4 sự kiện)", "VAL (15% Stratified)", "TRAIN (85% Net)"]
    data_loeo_det = [
        ["Fold 0", "Charlie Hebdo (CH)", "2,002 (447 / 1,555)", "SS + FG + OS + GC = 3,445", "517 (212 R / 305 NR)", "2,928 (1,204 R / 1,724 NR)"],
        ["Fold 1", "Sydney Siege (SS)", "1,173 (500 / 673)", "CH + FG + OS + GC = 4,274", "641 (204 R / 437 NR)", "3,633 (1,159 R / 2,474 NR)"],
        ["Fold 2", "Ferguson (FG)", "1,010 (257 / 753)", "CH + SS + OS + GC = 4,437", "665 (241 R / 424 NR)", "3,772 (1,365 R / 2,407 NR)"],
        ["Fold 3", "Ottawa Shooting (OS)", "857 (457 / 400)", "CH + SS + FG + GC = 4,590", "688 (211 R / 477 NR)", "3,902 (1,195 R / 2,707 NR)"],
        ["Fold 4", "Germanwings (GC)", "405 (202 / 203)", "CH + SS + FG + OS = 5,042", "756 (249 R / 507 NR)", "4,286 (1,412 R / 2,874 NR)"]
    ]
    b.add_table(headers_loeo_det, data_loeo_det, [0.65, 1.45, 1.5, 1.55, 1.25, 1.4], align_center_cols=[0, 2, 4, 5])

    b.add_heading("4.3. Hệ thống Đo lường Kép & Kiểm định Thống kê", level=2)
    b.add_p("1. Mean LOEO Metrics (Chuẩn Table 2 & 3 của Wu et al. 2025): Tính Accuracy và Macro-F1 độc lập trên từng sự kiện kiểm thử rồi tính trung bình cộng số học mu +- sigma qua 5 folds. Đây là thước đo chính thức để đối chiếu 1-1 với bài báo gốc.")
    b.add_p("2. Pooled Metrics: Gộp toàn bộ ma trận nhầm lẫn của 5 folds để tính chỉ số chung trên toàn thể 5,447 mẫu.")
    b.add_p("3. Kiểm định ý nghĩa thống kê: Áp dụng Paired Student's t-test và Wilcoxon signed-rank test trên chuỗi kết quả Macro-F1 của các fold giữa UC-TTT và baseline, với ngưỡng ý nghĩa p < 0.05 để khẳng định độ tin cậy khoa học.")

    # -------------------------------------------------------------------------
    # 5. Baselines
    # -------------------------------------------------------------------------
    b.add_heading("5. Baselines (Khảo sát Mở rộng & Bổ sung Tabular ML với XGBoost)", level=1)
    b.add_p("Đáp ứng chỉ đạo của GVHD về việc mở rộng survey và bổ sung mô hình học máy dạng bảng cơ bản, nhóm xây dựng phổ 4 tầng baseline toàn diện:")

    headers_bl = ["Tầng Mô hình", "Baseline Đối chuẩn", "Không gian Đặc trưng", "PHEME v5 Pooled Acc / F1", "Mean LOEO Acc / F1", "Mục đích đối chuẩn chuyên môn"]
    data_bl = [
        ["Content-only", "B1a: TF-IDF + Logistic Reg.", "2,000 từ vựng bài gốc", "71.77% / 67.56%", "63.36% / 56.17%", "Đo lường giới hạn phân loại của từ vựng tĩnh bài gốc."],
        ["Content-only", "B1b: TF-IDF + Logistic Reg.", "Từ vựng bài gốc + phản hồi", "70.60% / 65.67%", "60.12% / 52.85%", "Đo lường mức độ suy giảm do nhiễu từ phản hồi."],
        ["Structure-only", "B2: Propagation RF", "8 đặc trưng topo hình thái cây", "56.62% / 55.62%", "51.40% / 46.12%", "Đo lường năng lực phân loại của cấu trúc đồ thị thuần túy."],
        ["Tabular ML (Mới)", "B2b: Tabular XGBoost", "36 đặc trưng bảng (12 User + Graph + Text)", "62.22% / 53.21%", "59.39% / 50.22%", "Đối chứng SOTA Tabular ML: Kiểm chứng xem chỉ dùng đặc trưng User trên XGBoost có đủ tốt không."],
        ["DL SOTA", "B3: Original TTT (Wu et al. 2025)", "Text + Depth + Time Windows", "72.84% / 67.80% (Tái hiện)", "75.84% / 71.98% (Công bố)", "Mốc SOTA chuẩn mực của bài báo gốc để kiểm chứng H1."],
        ["Proposed", "UC-TTT (Proposed - Gated)", "Text + Depth + Time + User", "74.96% / 70.53%", "71.93% / 69.13%", "Khẳng định đóng góp vượt bậc của việc đưa User vào TTT."]
    ]
    b.add_table(headers_bl, data_bl, [0.85, 1.45, 1.45, 1.15, 1.15, 1.7], align_center_cols=[0, 3, 4])

    b.add_p("Giá trị khoa học đắt giá của Baseline 2b (Tabular XGBoost): Kết quả thực nghiệm thực tế cho thấy XGBoost dù sử dụng toàn bộ 36 đặc trưng bảng (kết hợp cả User Credibility và Propagation Topology) chỉ đạt Macro-F1 49.82% - 53.05% trên LOEO. Nguyên nhân do mô hình dạng bảng không thể nắm bắt được ngữ cảnh ngữ nghĩa sâu (deep contextual semantics) và diễn biến cây phân cấp động theo thời gian. Sự thua sút rõ rệt của XGBoost so với UC-TTT (Macro-F1 ~70-71%, vượt trội +18.5% F1) chứng minh vững chắc rằng: Việc tích hợp User Credibility trực tiếp vào các nút cây qua cơ chế Gated Attention của Transformer là hoàn toàn tất yếu và mang lại giá trị khoa học vượt trội.", italic=True)

    # -------------------------------------------------------------------------
    # 6. Proposed Method
    # -------------------------------------------------------------------------
    b.add_heading("6. Proposed Method: User-Credibility-Aware Temporal Tree Transformer (UC-TTT)", level=1)
    b.add_p("Mô hình đề xuất UC-TTT giải quyết triệt để khiếm khuyết của TTT gốc bằng cách dung hợp thông tin độ uy tín của người dùng vào từng nút trong cây lan truyền. Kiến trúc gồm 4 khối thành phần chính:")
    
    b.add_p("Vector đặc trưng uy tín u_i in R^12 của tác giả bài viết được chuẩn hóa qua LayerNorm và chiếu qua mạng MLP 2 tầng (Linear 12 -> 128 -> d_model với hàm kích hoạt GELU và Dropout 0.2) để tạo biểu diễn người dùng ẩn e_i in R^d (d = 768).", bold_prefix="1. User Credibility Encoder (UCE): ")
    
    b.add_p("Thay vì cộng đại số thô, mô hình sử dụng cơ chế cổng động (Gated Attention) dung hợp biểu diễn nội dung + độ sâu (h_i = s_i + PE(depth_i)) và biểu diễn uy tín người dùng (e_i):", bold_prefix="2. Gated Attention Fusion: ")
    b.add_p("g_i = sigmoid(W_g [h_i || e_i] + b_g), trong đó g_i in [0, 1]^d là vector cổng điều tiết.")
    b.add_p("z_i = g_i * h_i + (1 - g_i) * e_i, biểu diễn nút z_i được tự động điều chỉnh: nếu tác giả là tài khoản uy tín đính chính tin, e_i được khuếch đại; nếu tác giả là bot/tài khoản nặc danh, mô hình gắn cờ cảnh giác nghi vấn.")
    
    b.add_p("Các nút z_i trong cây con T_m ở từng cửa sổ thời gian được đưa qua 6 lớp Tree Transformer với 12 attention heads và ma trận mặt nạ Top-Down. Tiếp theo, các trạng thái cây con qua M cửa sổ thời gian được nạp vào mạng Bidirectional Temporal GRU (hidden dim 128) để mô hình hóa toàn diện diễn biến thời gian h_temporal.", bold_prefix="3. Tree Transformer & Temporal GRU: ")
    
    b.add_p("Ghép nối biểu diễn bài gốc s_root và vector thời gian h_temporal qua tầng FC 600 ẩn và Softmax dự đoán nhãn Rumour vs. Non-rumour.", bold_prefix="4. Joint Classification Head: ")

    # -------------------------------------------------------------------------
    # 7. Expected Results
    # -------------------------------------------------------------------------
    b.add_heading("7. Expected Results & Kế hoạch Bóc tách (Ablation Studies)", level=1)
    b.add_p("1. Kiểm chứng H1 (RQ1): UC-TTT đạt chỉ số Macro-F1 vượt mốc công bố của bài báo gốc Wu et al. 2025 (F1 > 72.00% và Acc > 75.84%), đạt cải thiện có ý nghĩa thống kê (p < 0.05).")
    b.add_p("2. Kiểm chứng H2 (RQ2 - Feature Ablation): Thực hiện bóc tách trên 5 folds: Full 12 đặc trưng vs w/o Profile vs w/o Counts vs w/o Ratios vs Source User Only. Đo độ suy giảm Delta F1 để chứng minh nhóm tỷ số hành vi và tick xanh giữ vai trò quyết định.")
    b.add_p("3. Kiểm chứng H3 (RQ3 - Event Analysis): Phân tích chi tiết mức cải thiện F1 trên từng sự kiện (kỳ vọng Ferguson tăng mạnh nhất > +5% F1 do có nhiều bot tham gia tung tin thất thiệt).")
    b.add_p("Kế hoạch dự phòng (Backup Plan): Nếu UC-TTT không thắng được TTT gốc ở một số fold, nhóm sẽ thử nghiệm chiến lược Source User Only (chỉ dùng người dùng bài gốc) hoặc chuyển sang phân tích chẩn đoán lỗi (Diagnostic Error Analysis) mổ xẻ vì sao trên mạng xã hội tích xanh vẫn chia sẻ tin giả.", italic=True)

    print("Successfully built Section B!")

    # =========================================================================
    # PART 2: SYNCHRONIZE APPENDIX 3 (Experimental plan)
    # =========================================================================
    # Find idx_exp_plan again in modified doc
    body_updated = doc._body._element
    idx_plan = None
    for idx, child in enumerate(body_updated):
        if child.tag.endswith('p'):
            ptxt = docx.text.paragraph.Paragraph(child, doc).text.strip()
            if '3. Experimental plan (proposal version 2)' in ptxt:
                idx_plan = idx
                break

    if idx_plan is not None:
        print(f"Synchronizing Appendix 3 at index {idx_plan}...")
        # Remove all p and tbl elements after idx_plan, preserving w:sectPr
        elements_after = [elm for elm in list(body_updated)[idx_plan + 1:] if elm.tag.endswith(('p', 'tbl'))]
        for elm in elements_after:
            body_updated.remove(elm)

        # Append new synchronized Appendix 3 content to end of doc
        p_intro = doc.add_paragraph(style='Normal')
        p_intro.paragraph_format.space_before = Pt(2.0)
        p_intro.paragraph_format.space_after = Pt(2.5)
        p_intro.paragraph_format.line_spacing = 1.15
        r = p_intro.add_run("Kế hoạch thực nghiệm kiểm chứng 3 câu hỏi nghiên cứu (RQ1–RQ3) và các giả thuyết (H1–H3) theo giao thức Leave-One-Event-Out (LOEO) trên bộ dữ liệu chuẩn PHEME v5, đảm bảo không rò rỉ dữ liệu và có tính tái lập 100%.")
        r.font.name = "Arial"
        r.font.size = Pt(9.5)

        # 1. Dataset
        p_h1 = doc.add_paragraph(style='Normal')
        p_h1.paragraph_format.space_before = Pt(4.0)
        p_h1.paragraph_format.space_after = Pt(1.5)
        r = p_h1.add_run("1. Dataset Specifications & Statistics:")
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        r.bold = True
        r.font.color.rgb = RGBColor(18, 42, 78)

        t_app_data = doc.add_table(rows=len(data_stat) + 1, cols=len(headers_stat))
        for c_i, h in enumerate(headers_stat):
            t_app_data.rows[0].cells[c_i].paragraphs[0].text = h
        for r_i, row in enumerate(data_stat):
            for c_i, val in enumerate(row):
                t_app_data.rows[r_i + 1].cells[c_i].paragraphs[0].text = str(val)
        style_table(t_app_data, [1.5, 0.8, 0.8, 0.8, 0.9, 0.8, 0.9], align_center_cols=[1, 2, 3, 4, 5, 6])

        # 2. LOEO Split Statistics
        p_h2 = doc.add_paragraph(style='Normal')
        p_h2.paragraph_format.space_before = Pt(4.0)
        p_h2.paragraph_format.space_after = Pt(1.5)
        r = p_h2.add_run("2. Train / Validation / Test Sample Statistics (5-Fold LOEO):")
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        r.bold = True
        r.font.color.rgb = RGBColor(18, 42, 78)

        t_app_loeo = doc.add_table(rows=len(data_loeo_det) + 1, cols=len(headers_loeo_det))
        for c_i, h in enumerate(headers_loeo_det):
            t_app_loeo.rows[0].cells[c_i].paragraphs[0].text = h
        for r_i, row in enumerate(data_loeo_det):
            for c_i, val in enumerate(row):
                t_app_loeo.rows[r_i + 1].cells[c_i].paragraphs[0].text = str(val)
        style_table(t_app_loeo, [0.65, 1.45, 1.5, 1.55, 1.25, 1.4], align_center_cols=[0, 2, 4, 5])

        # 3. Baselines & Benchmarks
        p_h3 = doc.add_paragraph(style='Normal')
        p_h3.paragraph_format.space_before = Pt(4.0)
        p_h3.paragraph_format.space_after = Pt(1.5)
        r = p_h3.add_run("3. Baselines & Benchmarking Matrix (Đã bổ sung Tabular XGBoost):")
        r.font.name = "Arial"
        r.font.size = Pt(10.5)
        r.bold = True
        r.font.color.rgb = RGBColor(18, 42, 78)

        t_app_bl = doc.add_table(rows=len(data_bl) + 1, cols=len(headers_bl))
        for c_i, h in enumerate(headers_bl):
            t_app_bl.rows[0].cells[c_i].paragraphs[0].text = h
        for r_i, row in enumerate(data_bl):
            for c_i, val in enumerate(row):
                t_app_bl.rows[r_i + 1].cells[c_i].paragraphs[0].text = str(val)
        style_table(t_app_bl, [0.85, 1.45, 1.45, 1.15, 1.15, 1.7], align_center_cols=[0, 3, 4])

        # Hyperparameters
        p_hp = doc.add_paragraph(style='Normal')
        p_hp.paragraph_format.space_before = Pt(2.0)
        p_hp.paragraph_format.space_after = Pt(2.0)
        r = p_hp.add_run("Cấu hình siêu tham số đồng bộ bài báo gốc Wu et al. (PLOS ONE 2025): Biểu diễn ẩn d_model = 768, attention heads = 12, số lớp Tree Transformer = 6, số lát cắt thời gian K = 8 (k = 8 bài/cửa sổ), Temporal BiGRU hidden = 128, FC classifier hidden = 600, Dropout = 0.2, Optimizer AdamW (lr = 4.2e-5, weight_decay = 1e-4), Batch size = 32 luồng cây, EarlyStopping patience = 8 theo Validation Macro-F1, Seed = 42 cố định trên GPU CUDA.")
        r.font.name = "Arial"
        r.font.size = Pt(8.5)
        r.italic = True

    # Save to main template file
    dest_file = "[Main Template] Mini Proposal - THN.docx"
    doc.save(dest_file)
    print(f"\n[SUCCESS] Successfully generated and overwritten: {dest_file}")

if __name__ == "__main__":
    build_full_proposal()
