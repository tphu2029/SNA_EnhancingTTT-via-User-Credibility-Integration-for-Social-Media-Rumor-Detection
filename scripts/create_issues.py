import os
import re
import sys
import json
import urllib.request
import urllib.error

REPO_OWNER = "tphu2029"
REPO_NAME = "SNA_EnhancingTTT-via-User-Credibility-Integration-for-Social-Media-Rumor-Detection"
BASE_API_URL = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"

def send_github_request(token, endpoint, payload=None, method="GET"):
    url = f"{BASE_API_URL}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
        "User-Agent": "Antigravity-Issue-Updater"
    }

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8")
        return None, f"HTTP {e.code}: {err}"
    except Exception as e:
        return None, str(e)

def get_or_create_milestones(token, milestone_titles):
    print("\n⏳ Đang đồng bộ các Milestones trên GitHub...")
    existing, err = send_github_request(token, "milestones", method="GET")
    milestone_map = {}

    if existing and isinstance(existing, list):
        for ms in existing:
            milestone_map[ms["title"]] = ms["number"]

    for title in milestone_titles:
        if title not in milestone_map:
            print(f"  ➕ Tạo mới Milestone: '{title}'")
            res, err = send_github_request(token, "milestones", payload={"title": title}, method="POST")
            if res and "number" in res:
                milestone_map[title] = res["number"]
            else:
                print(f"  ⚠️ Không thể tạo milestone '{title}': {err}")

    print(f"✅ Đã có {len(milestone_map)} Milestones trên GitHub.")
    return milestone_map

def parse_issues_from_todo(todo_path="TODO.md"):
    if not os.path.exists(todo_path):
        print(f"Error: {todo_path} not found.")
        sys.exit(1)

    with open(todo_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Match each issue block between '### Issue #X:' and next '### Issue #' or '## 4. QUY CHUẨN'
    pattern = r'### Issue #(\d+):[^\n]*\n(.*?\n)(?=(?:### Issue #\d+:|## 4\. QUY CHUẨN))'
    matches = re.findall(pattern, content, re.DOTALL)
    issues = []

    for num, raw_block in matches:
        clean_text = raw_block.strip()
        if clean_text.startswith("```markdown"):
            clean_text = clean_text[len("```markdown"):].strip()
        clean_text = re.sub(r'```\s*$', '', clean_text).strip()

        # Extract title
        title_m = re.search(r'## Tiêu đề Issue:\s*\n([^\n]+)', clean_text)
        title = title_m.group(1).strip() if title_m else f"Issue #{num}"

        # Extract labels
        labels = []
        labels_m = re.search(r'## Nhãn \(Labels\):\s*\n([^\n]+)', clean_text)
        if labels_m:
            labels = re.findall(r'`([^`]+)`', labels_m.group(1))

        # Extract milestone name
        ms_m = re.search(r'## Milestone:\s*\n([^\n]+)', clean_text)
        milestone_title = ms_m.group(1).strip() if ms_m else None

        # Extract assignee & reviewer text
        assignee_m = re.search(r'## Người phụ trách \(Assignee\):\s*\n([^\n]+)', clean_text)
        assignee = assignee_m.group(1).strip() if assignee_m else ''

        reviewer_m = re.search(r'## Người phối hợp[^\n]*:\s*\n([^\n]+)', clean_text)
        reviewer = reviewer_m.group(1).strip() if reviewer_m else ''

        # Extract description part (Everything from ### 📖 DESCRIPTION onwards)
        desc_start = clean_text.find('### 📖 DESCRIPTION')
        if desc_start != -1:
            desc_part = clean_text[desc_start:].strip()
        else:
            desc_part = clean_text.strip()

        # Build clean body: Remove redundant title/labels/milestones, keep clean info header + description
        header_lines = []
        if assignee:
            header_lines.append(f"> 👤 **Người phụ trách:** {assignee}")
        if reviewer:
            header_lines.append(f"> 👥 **Người phối hợp:** {reviewer}")

        header = "\n".join(header_lines) + "\n\n---\n\n" if header_lines else ""
        clean_body = header + desc_part

        issues.append({
            "number": int(num),
            "title": title,
            "labels": labels,
            "milestone_title": milestone_title,
            "body": clean_body
        })

    return issues

def main():
    print("=" * 65)
    print("  ĐỒNG BỘ NỘI DUNG VÀ LIÊN KẾT LABELS/MILESTONE CHO 8 ISSUES")
    print(f"  Repo: {REPO_OWNER}/{REPO_NAME}")
    print("=" * 65)

    token = os.environ.get("GITHUB_TOKEN")
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]

    if not token:
        token = input("\nNhập GitHub Token của bạn (ghp_...): ").strip()

    if not token:
        print("Lỗi: Chưa nhập token. Dừng chương trình.")
        sys.exit(1)

    issues = parse_issues_from_todo("TODO.md")
    print(f"\nĐã đọc {len(issues)} issues từ TODO.md.")

    # 1. Thu thập và tự động tạo Milestones trên GitHub
    milestone_titles = set(iss["milestone_title"] for iss in issues if iss["milestone_title"])
    milestone_map = get_or_create_milestones(token, milestone_titles)

    print("\nChọn thao tác:")
    print("  1. CẬP NHẬT đè nội dung sạch & liên kết Milestone vào 8 Issues đã có (Khuyên dùng)")
    print("  2. TẠO MỚI các Issues")
    choice = input("Lựa chọn (1 hoặc 2, mặc định là 1): ").strip()

    method = "POST" if choice == "2" else "PATCH"
    print(f"\n--- ĐANG TIẾN HÀNH ({'TẠO MỚI' if method == 'POST' else 'CẬP NHẬT'}) ---")

    for iss in issues:
        num = iss["number"]
        ms_id = milestone_map.get(iss["milestone_title"])

        payload = {
            "title": iss["title"],
            "body": iss["body"],
            "labels": iss["labels"]
        }
        if ms_id:
            payload["milestone"] = ms_id

        endpoint = "issues" if method == "POST" else f"issues/{num}"
        print(f"Đang đồng bộ Issue #{num}: {iss['title'][:50]}... (Milestone: {iss['milestone_title']})")

        res, err = send_github_request(token, endpoint, payload=payload, method=method)
        if res:
            print(f"  ✅ Thành công: {res.get('html_url')}")
        else:
            print(f"  ❌ Lỗi: {err}")

    print("\n🎉 Hoàn thành! Mời bạn mở lại GitHub xem kết quả:")
    print(f"https://github.com/{REPO_OWNER}/{REPO_NAME}/issues")

if __name__ == "__main__":
    main()
