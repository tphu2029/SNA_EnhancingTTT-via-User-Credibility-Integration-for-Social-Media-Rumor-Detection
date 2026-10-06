# Script tao 8 GitHub Issues bang GitHub CLI (gh)
# Yeu cau da cai dat gh: winget install --id GitHub.cli -e
# Va da dang nhap: gh auth login

Write-Host "Dang tao 8 Issues len GitHub..." -ForegroundColor Cyan

# Issue 1
gh issue create `
  --title "[Data] [Giai đoạn 1] Xây dựng script trích xuất Offline Text Embeddings cho dữ liệu PHEME" `
  --body-file "scripts/issues/issue_1.md" `
  --label "giai-doan-1,data-pipeline,enhancement,priority: high"

# Issue 2
gh issue create `
  --title "[Data] [Giai đoạn 1] Lập trình PHEMETreeDataset và collate_fn gom batch cây lan truyền động" `
  --body-file "scripts/issues/issue_2.md" `
  --label "giai-doan-1,data-pipeline,model-infra,priority: high"

# Issue 3
gh issue create `
  --title "[Model] [Giai đoạn 2] Huấn luyện 9-Fold LOEO tái hiện Baseline 3 — Original Temporal Tree Transformer" `
  --body-file "scripts/issues/issue_3.md" `
  --label "giai-doan-2,deep-learning,baseline,priority: high"

# Issue 4
gh issue create `
  --title "[Model] [Giai đoạn 3] Huấn luyện mô hình đề xuất UC-TTT trên 9 Folds LOEO và thử nghiệm cơ chế Fusion (RQ1)" `
  --body-file "scripts/issues/issue_4.md" `
  --label "giai-doan-3,core-feature,deep-learning,priority: critical"

# Issue 5
gh issue create `
  --title "[Experiment] [Giai đoạn 4] Thực hiện chuỗi thực nghiệm Ablation Study bóc tách đặc trưng và mô hình (RQ2)" `
  --body-file "scripts/issues/issue_5.md" `
  --label "giai-doan-4,ablation,experiments,priority: high"

# Issue 6
gh issue create `
  --title "[Analysis] [Giai đoạn 3 & 5] Phân tích thống kê EDA, kiểm định p-value và phân tích 9 sự kiện (RQ3)" `
  --body-file "scripts/issues/issue_6.md" `
  --label "giai-doan-3,giai-doan-5,analysis,statistics,priority: high"

# Issue 7
gh issue create `
  --title "[Analysis] [Giai đoạn 5] Phân tích chi tiết các ca dự đoán sai (Error Analysis) và tuyển chọn Case Studies" `
  --body-file "scripts/issues/issue_7.md" `
  --label "giai-doan-5,analysis,case-study,priority: medium"

# Issue 8
gh issue create `
  --title "[Documentation] [Giai đoạn 6] Biên soạn Báo cáo cuối kỳ chuẩn IMRaD, Slide thuyết trình & Đóng gói Repo" `
  --body-file "scripts/issues/issue_8.md" `
  --label "giai-doan-6,documentation,report,presentation,priority: high"

Write-Host "Hoan thanh tao 8 issues!" -ForegroundColor Green
