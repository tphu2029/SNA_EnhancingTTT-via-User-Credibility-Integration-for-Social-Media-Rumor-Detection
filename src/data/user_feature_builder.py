import math
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"

def parse_twitter_datetime(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, TWITTER_DATE_FORMAT)
    except Exception:
        return None

class UserCredibilityExtractor:
    """
    Trích xuất các nhóm đặc trưng User Credibility từ tweet user metadata trong PHEME.
    Bao gồm 3 nhóm:
      1. Profile & Verification (verified, default avatar, description, account age)
      2. Engagement Counts (followers, friends, statuses, favourites, listed - log transformed)
      3. Derived Behavioral Ratios (reputation ratio, activity intensity, engagement ratio)
    """

    FEATURE_NAMES = [
        # Nhóm 1: Profile & Verification
        "is_verified",
        "has_default_avatar",
        "has_description",
        "log_account_age_days",
        
        # Nhóm 2: Engagement Counts (Log-scaled)
        "log_followers",
        "log_friends",
        "log_statuses",
        "log_favourites",
        "log_listed",
        
        # Nhóm 3: Derived Behavioral Ratios
        "log_reputation_ratio",  # log(followers / (friends + 1))
        "log_status_rate",        # log(statuses / (age_days + 1))
        "log_favourite_rate",     # log(favourites / (statuses + 1))
    ]

    GROUP_INDICES = {
        "profile": [0, 1, 2, 3],
        "counts": [4, 5, 6, 7, 8],
        "ratios": [9, 10, 11],
    }

    def __init__(self):
        self.num_features = len(self.FEATURE_NAMES)

    def extract_from_tweet(self, tweet_dict: Dict[str, Any]) -> np.ndarray:
        """
        Trích xuất vector credibility từ một đối tượng tweet JSON.
        """
        user = tweet_dict.get("user")
        if not user or not isinstance(user, dict):
            # Missing user metadata fallback
            return np.zeros(self.num_features, dtype=np.float32)

        tweet_dt = parse_twitter_datetime(tweet_dict.get("created_at"))
        user_dt = parse_twitter_datetime(user.get("created_at"))
        
        # 1. Profile & Verification
        is_verified = 1.0 if user.get("verified", False) else 0.0
        has_default_avatar = 1.0 if user.get("default_profile_image", False) else 0.0
        desc = user.get("description")
        has_description = 1.0 if (desc is not None and len(str(desc).strip()) > 0) else 0.0
        
        if tweet_dt and user_dt and tweet_dt >= user_dt:
            age_days = max(0.0, (tweet_dt - user_dt).total_seconds() / 86400.0)
        else:
            age_days = 0.0
        log_account_age_days = math.log1p(age_days)

        # 2. Counts
        followers = max(0.0, float(user.get("followers_count", 0) or 0))
        friends = max(0.0, float(user.get("friends_count", 0) or 0))
        statuses = max(0.0, float(user.get("statuses_count", 0) or 0))
        favourites = max(0.0, float(user.get("favourites_count", 0) or 0))
        listed = max(0.0, float(user.get("listed_count", 0) or 0))

        log_followers = math.log1p(followers)
        log_friends = math.log1p(friends)
        log_statuses = math.log1p(statuses)
        log_favourites = math.log1p(favourites)
        log_listed = math.log1p(listed)

        # 3. Derived Ratios
        reputation_ratio = followers / (friends + 1.0)
        log_reputation_ratio = math.log1p(max(0.0, reputation_ratio))

        status_rate = statuses / (age_days + 1.0)
        log_status_rate = math.log1p(max(0.0, status_rate))

        favourite_rate = favourites / (statuses + 1.0)
        log_favourite_rate = math.log1p(max(0.0, favourite_rate))

        features = [
            is_verified,
            has_default_avatar,
            has_description,
            log_account_age_days,
            log_followers,
            log_friends,
            log_statuses,
            log_favourites,
            log_listed,
            log_reputation_ratio,
            log_status_rate,
            log_favourite_rate,
        ]

        return np.array(features, dtype=np.float32)

    def select_feature_group(self, feature_vector: np.ndarray, mode: str = "all") -> np.ndarray:
        """
        Lọc các nhóm feature phục vụ cho Ablation Studies.
        mode:
          - 'all': đầy đủ 12 features
          - 'no_profile': loại bỏ nhóm 1
          - 'no_counts': loại bỏ nhóm 2
          - 'no_ratios': loại bỏ nhóm 3
          - 'profile_only': chỉ nhóm 1
          - 'counts_only': chỉ nhóm 2
          - 'ratios_only': chỉ nhóm 3
        """
        if mode == "all":
            return feature_vector
        elif mode == "no_profile":
            idx = self.GROUP_INDICES["counts"] + self.GROUP_INDICES["ratios"]
            return feature_vector[..., idx]
        elif mode == "no_counts":
            idx = self.GROUP_INDICES["profile"] + self.GROUP_INDICES["ratios"]
            return feature_vector[..., idx]
        elif mode == "no_ratios":
            idx = self.GROUP_INDICES["profile"] + self.GROUP_INDICES["counts"]
            return feature_vector[..., idx]
        elif mode == "profile_only":
            return feature_vector[..., self.GROUP_INDICES["profile"]]
        elif mode == "counts_only":
            return feature_vector[..., self.GROUP_INDICES["counts"]]
        elif mode == "ratios_only":
            return feature_vector[..., self.GROUP_INDICES["ratios"]]
        else:
            raise ValueError(f"Unknown mode: {mode}")
