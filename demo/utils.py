import math
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import torch

def parse_twitter_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parse Twitter created_at timestamp string into datetime object."""
    if not date_str:
        return None
    for fmt in [
        "%a %b %d %H:%M:%S +0000 %Y",
        "%a %b %d %H:%M:%S %z %Y",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None

class UserCredibilityExtractor:
    """
    Trích xuất vector 12 đặc trưng độ uy tín người dùng (User Credibility Vector - u_i in R^12)
    chuẩn hóa theo đề tài nhóm:
      - 3 đặc trưng Profile: is_verified, has_default_avatar, has_description
      - 5 đặc trưng Quy mô (log1p): followers, friends, statuses, favourites, listed
      - 4 đặc trưng Tỷ lệ hành vi (log1p): reputation_ratio, status_rate, favourite_rate, account_age_days
    """
    @staticmethod
    def extract_from_raw_user(raw_user: Dict[str, Any], tweet_created_at: Optional[str] = None) -> np.ndarray:
        if not raw_user:
            return np.zeros(12, dtype=np.float32)

        # 1. Profile Verification & Authenticity (3 features)
        is_verified = 1.0 if bool(raw_user.get("verified", False)) else 0.0
        has_default_avatar = 1.0 if bool(raw_user.get("default_profile_image", False)) else 0.0
        desc = raw_user.get("description", "")
        has_description = 1.0 if (desc and len(str(desc).strip()) > 0) else 0.0

        # 2. Activity & Engagement Counts (5 features)
        followers = max(0.0, float(raw_user.get("followers_count", 0) or 0))
        friends = max(0.0, float(raw_user.get("friends_count", 0) or 0))
        statuses = max(0.0, float(raw_user.get("statuses_count", 0) or 0))
        favourites = max(0.0, float(raw_user.get("favourites_count", 0) or 0))
        listed = max(0.0, float(raw_user.get("listed_count", 0) or 0))

        log_followers = float(math.log1p(followers))
        log_friends = float(math.log1p(friends))
        log_statuses = float(math.log1p(statuses))
        log_favourites = float(math.log1p(favourites))
        log_listed = float(math.log1p(listed))

        # 3. Behavioral Ratios & Age (4 features)
        t_dt = parse_twitter_datetime(tweet_created_at) if tweet_created_at else None
        u_dt = parse_twitter_datetime(raw_user.get("created_at"))
        
        if t_dt and u_dt and t_dt >= u_dt:
            age_days = max(1.0, (t_dt - u_dt).total_seconds() / 86400.0)
        else:
            age_days = 365.0

        reputation_ratio = followers / (friends + 1.0)
        status_rate = statuses / (age_days + 1.0)
        favourite_rate = favourites / (statuses + 1.0)

        log_reputation = float(math.log1p(reputation_ratio))
        log_status_rate = float(math.log1p(status_rate))
        log_favourite_rate = float(math.log1p(favourite_rate))
        log_age_days = float(math.log1p(age_days))

        feats = [
            is_verified,
            has_default_avatar,
            has_description,
            log_followers,
            log_friends,
            log_statuses,
            log_favourites,
            log_listed,
            log_reputation,
            log_status_rate,
            log_favourite_rate,
            log_age_days,
        ]
        return np.asarray(feats, dtype=np.float32)

    @staticmethod
    def compute_credibility_score(raw_user: Dict[str, Any]) -> int:
        """Tính điểm uy tín tổng quan (0-100) cho hiển thị giao diện trực quan."""
        if not raw_user:
            return 30
        score = 40.0
        if raw_user.get("verified", False):
            score += 35.0
        if raw_user.get("default_profile_image", False):
            score -= 20.0
        desc = raw_user.get("description", "")
        if desc and len(str(desc).strip()) > 0:
            score += 5.0
        
        followers = max(0, int(raw_user.get("followers_count", 0) or 0))
        friends = max(0, int(raw_user.get("friends_count", 0) or 0))
        if followers > 500000:
            score += 15.0
        elif followers > 50000:
            score += 10.0
        elif followers > 5000:
            score += 5.0
        
        rep = followers / (friends + 1.0)
        if rep > 10.0:
            score += 10.0
        elif rep > 2.0:
            score += 5.0
        elif rep < 0.2:
            score -= 10.0
            
        return max(5, min(100, int(round(score))))

class TextEmbeddingExtractor:
    """Singleton trích xuất vector BERT-base 768 chiều cho bài đăng mới."""
    _tokenizer = None
    _model = None

    @classmethod
    def get_embedding(cls, text: str, device="cpu") -> torch.Tensor:
        if cls._tokenizer is None or cls._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                model_name = "bert-base-uncased"
                cls._tokenizer = AutoTokenizer.from_pretrained(model_name)
                cls._model = AutoModel.from_pretrained(model_name).to(device)
                cls._model.eval()
            except Exception:
                # Fallback: Random deterministic embedding if transformers not available
                torch.manual_seed(abs(hash(text)) % (2**31))
                return torch.randn(768).float()

        inputs = cls._tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        with torch.no_grad():
            outputs = cls._model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu() # [CLS] token
        return emb.float()
