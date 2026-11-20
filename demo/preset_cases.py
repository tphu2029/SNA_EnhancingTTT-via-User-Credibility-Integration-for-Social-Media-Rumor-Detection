"""
Danh sách 5 Ca Thực Nghiệm Tiêu Biểu (PHEME Benchmark 5 Events)
Tương ứng chính xác 1-1 với 5 sự kiện chuẩn của bài báo Wu et al. (PLOS ONE 2025):
  1. Charlie Hebdo (CH) - Fold 0
  2. Sydney Siege (SS) - Fold 4
  3. Ferguson (FG) - Fold 1
  4. Ottawa Shooting (OS) - Fold 3
  5. Germanwings Crash (GC) - Fold 2
"""

PRESET_CASES = [
    {
        "id": "case_charlie_hebdo",
        "title": "Charlie Hebdo (CH): Vụ tấn công tòa soạn @LadySandersfarm",
        "category": "Đảo chiều phán đoán (TTT: 76.9% -> UC: 20.8%)",
        "thread_id": "553569333150896129",
        "event": "charliehebdo",
        "fold": 0,
        "badge": "Fold 0 (CH): TTT đoán 76.9% -> UC sửa 20.8%",
        "summary": "Bài đăng khẳng định vụ tấn công khủng bố tòa soạn Charlie Hebdo là có thật ('Wake up, it\\'s real'). TTT gốc bị từ ngữ kích động đánh lừa phán đoán là TIN ĐỒN với độ tự tin 76.9%. UC-TTT phân tích tín hiệu người dùng và mạng lưới phản hồi đã hạ xác suất xuống 20.8%, sửa đúng thành TIN XÁC THỰC.",
    },
    {
        "id": "case_sydney_siege",
        "title": "Sydney Siege (SS): Nhà báo Guardian @bkjabour",
        "category": "Xác minh nguồn tin hiện trường (TTT: 56.5% -> UC: 35.9%)",
        "thread_id": "544384467809230848",
        "event": "sydneysiege",
        "fold": 4,
        "badge": "Fold 4 (SS): TTT đoán 56.5% -> UC sửa 35.9%",
        "summary": "Phóng viên hiện trường báo cáo thông tin từ Cảnh sát Liên bang Úc (AFP) về việc kẻ bắt giữ con tin đang theo dõi mạng xã hội. TTT gốc dự đoán sai lệch là TIN ĐỒN (56.5%). UC-TTT nhận diện tác giả là phóng viên Guardian tác nghiệp hiện trường, điều chỉnh xác suất về 35.9% và phán đoán đúng TIN XÁC THỰC.",
    },
    {
        "id": "case_ferguson",
        "title": "Ferguson (FG): Bắt đúng tin đồn kích động @inthefade",
        "category": "Khắc phục bỏ sót tin đồn (TTT: 47.6% -> UC: 72.0%)",
        "thread_id": "498254929942028288",
        "event": "ferguson",
        "fold": 1,
        "badge": "Fold 1 (FG): TTT bỏ sót 47.6% -> UC bắt đúng 72.0%",
        "summary": "Bài đăng lan truyền tin đồn sai sự thật rằng thiếu niên bị cảnh sát bắn 10 phát chỉ vì trộm kẹo. TTT gốc bị đánh lừa bởi cấu trúc câu hỏi han và dự đoán là Tin xác thực (47.6% tin đồn, tức bỏ sót tin đồn). UC-TTT nhận diện đặc trưng tài khoản và phản hồi hoài nghi của cộng đồng, nâng xác suất tin đồn lên 72.0% và bắt chính xác TIN ĐỒN.",
    },
    {
        "id": "case_ottawa_shooting",
        "title": "Ottawa Shooting (OS): Hãng tin @BreakingNews (7.0M Followers)",
        "category": "Triệt tiêu báo động giả (TTT: 70.4% -> UC: 35.9%)",
        "thread_id": "524931083797004288",
        "event": "ottawashooting",
        "fold": 3,
        "badge": "Fold 3 (OS): TTT đoán 70.4% -> UC sửa 35.9%",
        "summary": "Thông tin khẩn xác nhận Thủ tướng Canada Stephen Harper đã an toàn rời khỏi tòa nhà Nghị viện sau vụ xả súng. TTT gốc bị ngữ cảnh xả súng bạo lực đánh lừa dự đoán là TIN ĐỒN (70.4%). UC-TTT nhận diện hãng tin @BreakingNews (7 triệu followers, tick xanh) đã sửa đúng thành TIN XÁC THỰC (35.9%).",
    },
    {
        "id": "case_germanwings",
        "title": "Germanwings (GC): Hãng tin @CNN (16.3M Followers)",
        "category": "Triệt tiêu báo động giả thảm kịch (TTT: 57.4% -> UC: 8.8%)",
        "thread_id": "581247332282306561",
        "event": "germanwings-crash",
        "fold": 2,
        "badge": "Fold 2 (GC): TTT đoán 57.4% -> UC sửa 8.8%",
        "summary": "Trong thảm họa máy bay rơi Germanwings, tweet hỏi đáp trực tiếp của @CNN bị TTT gốc báo động giả là TIN ĐỒN (57.4%) do chứa từ khóa tai nạn máy bay rơi. UC-TTT kết hợp độ uy tín tối thượng của @CNN (16.3M followers, tick xanh) và phản hồi thực tế đã hạ xác suất tin đồn xuống 8.8%, phục hồi đúng nhãn TIN XÁC THỰC.",
    },
]
