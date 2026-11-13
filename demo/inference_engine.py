import math
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import build_tree, paper_time_windows, _parse_time
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer
from demo.utils import UserCredibilityExtractor, TextEmbeddingExtractor, parse_twitter_datetime

EVENT_FOLD_MAP = {
    "charliehebdo": 0,
    "ferguson": 1,
    "germanwings-crash": 2,
    "ottawashooting": 3,
    "sydneysiege": 4,
}

LABEL_NAMES = {
    0: "Non-rumour (Tin xác thực)",
    1: "Rumour (Tin đồn)",
}

class RumorDetectionDemoEngine:
    def __init__(self, device: Optional[str] = None):
        self.root_dir = ROOT
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.user_extractor = UserCredibilityExtractor()
        
        self.threads: List[Dict[str, Any]] = []
        self.threads_by_id: Dict[str, Dict[str, Any]] = {}
        self.text_embeddings: Optional[Dict[str, torch.Tensor]] = None
        
        # Model cache: fold_idx -> (model_ttt, model_uc)
        self.model_cache: Dict[int, Tuple[UserCredibilityTemporalTreeTransformer, UserCredibilityTemporalTreeTransformer]] = {}

    def load_data(self):
        """Tải dữ liệu PHEME đã tiền xử lý và precomputed BERT embeddings."""
        pkl_path = self.root_dir / "data" / "processed" / "pheme_processed.pkl"
        emb_path = self.root_dir / "data" / "processed" / "text_embeddings_bert.pt"
        
        if not pkl_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {pkl_path}")
        if not emb_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file embeddings: {emb_path}")
            
        with open(pkl_path, "rb") as f:
            self.threads = pickle.load(f)
            
        self.threads_by_id = {str(th["thread_id"]): th for th in self.threads}
        self.text_embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)

    def load_models_for_fold(self, fold_idx: int = 3) -> Tuple[UserCredibilityTemporalTreeTransformer, UserCredibilityTemporalTreeTransformer]:
        """Tải cặp checkpoint: TTT Baseline và UC-TTT Proposed cho fold chỉ định."""
        if fold_idx in self.model_cache:
            return self.model_cache[fold_idx]
            
        ttt_ckpt_path = self.root_dir / "results" / "checkpoints" / "ttt_baseline" / f"ttt_baseline_fold_{fold_idx}.pt"
        uc_ckpt_path = self.root_dir / "results" / "checkpoints" / "ucttt" / f"ucttt_residual_gate_mode5_fold_{fold_idx}.pt"
        
        if not ttt_ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint TTT Baseline: {ttt_ckpt_path}")
        if not uc_ckpt_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint UC-TTT Proposed: {uc_ckpt_path}")
            
        # 1. Khởi tạo & nạp trọng số TTT Baseline (use_user_credibility=False)
        model_ttt = UserCredibilityTemporalTreeTransformer(
            text_input_dim=768,
            d_model=768,
            nhead=12,
            num_tree_layers=6,
            fc_hidden_dim=600,
            use_user_credibility=False,
            window_aggregation="cumulative",
        ).to(self.device)
        
        ckpt_ttt = torch.load(ttt_ckpt_path, map_location=self.device, weights_only=False)
        state_ttt = ckpt_ttt["model_state_dict"] if "model_state_dict" in ckpt_ttt else ckpt_ttt
        model_ttt.load_state_dict(state_ttt)
        model_ttt.eval()
        
        # 2. Khởi tạo & nạp trọng số UC-TTT Proposed (use_user_credibility=True, residual_gate)
        model_uc = UserCredibilityTemporalTreeTransformer(
            text_input_dim=768,
            d_model=768,
            nhead=12,
            num_tree_layers=6,
            fc_hidden_dim=600,
            use_user_credibility=True,
            fusion_type="residual_gate",
            window_aggregation="cumulative",
        ).to(self.device)
        
        ckpt_uc = torch.load(uc_ckpt_path, map_location=self.device, weights_only=False)
        state_uc = ckpt_uc["model_state_dict"] if "model_state_dict" in ckpt_uc else ckpt_uc
        model_uc.load_state_dict(state_uc)
        model_uc.eval()
        
        self.model_cache[fold_idx] = (model_ttt, model_uc)
        return model_ttt, model_uc

    def get_events(self) -> List[str]:
        return list(EVENT_FOLD_MAP.keys())

    def get_threads_for_event(self, event_name: str, min_replies: int = 1) -> List[Dict[str, Any]]:
        return [
            th for th in self.threads
            if th["event"] == event_name and len(th.get("reactions", [])) >= min_replies
        ]

    def _prepare_thread_tensors(self, thread: Dict[str, Any]) -> Dict[str, Any]:
        """Chuẩn hóa dữ liệu đầu vào của luồng hội thoại theo đúng quy cách bài báo Wu et al. 2025."""
        nodes = [thread["source_tweet"]] + list(thread.get("reactions", []))
        n = len(nodes)
        
        # 1. Text Embeddings (d = 768)
        text_tensors = []
        for x in nodes:
            x_id = str(x.get("id"))
            if self.text_embeddings and x_id in self.text_embeddings:
                emb = self.text_embeddings[x_id]
            else:
                emb = TextEmbeddingExtractor.get_embedding(x.get("text", ""), device=self.device)
            text_tensors.append(emb)
        text = torch.stack(text_tensors).float() # [N, 768]
        
        # 2. User Features (12 đặc trưng)
        user_feats = []
        for x in nodes:
            if "credibility_features" in x:
                feat = np.asarray(x["credibility_features"], dtype=np.float32)
            elif "raw_user" in x:
                feat = self.user_extractor.extract_from_raw_user(x["raw_user"], x.get("created_at"))
            elif "user" in x:
                feat = self.user_extractor.extract_from_raw_user(x["user"], x.get("created_at"))
            else:
                feat = np.zeros(12, dtype=np.float32)
            user_feats.append(feat)
        user = torch.from_numpy(np.stack(user_feats)).float() # [N, 12]
        
        # 3. Cấu trúc cây & Lát cắt thời gian tích lũy
        parent, depth, reach = build_tree(nodes, thread.get("edges", []))
        times = np.array([_parse_time(x.get("created_at")) for x in nodes])
        win = paper_time_windows(times, window_size=8)
        num_win = int(win.max()) + 1 if len(win) > 0 else 1
        
        related = reach | reach.T
        same_win = win[:, None] == win[None, :]
        adj = related | same_win
        
        return {
            "text_features": text.unsqueeze(0).to(self.device),               # [1, N, 768]
            "user_features": user.unsqueeze(0).to(self.device),               # [1, N, 12]
            "depths": torch.from_numpy(depth).unsqueeze(0).to(self.device),   # [1, N]
            "time_window_assignments": torch.from_numpy(win).unsqueeze(0).to(self.device), # [1, N]
            "num_windows": torch.tensor([num_win], dtype=torch.long, device=self.device),
            "adj_mask": torch.from_numpy(adj).unsqueeze(0).to(self.device).bool(), # [1, N, N]
            "node_padding_mask": torch.zeros((1, n), dtype=torch.bool, device=self.device),
            "timestamps": times,
            "nodes": nodes,
            "window_assignments_raw": win,
            "total_windows": num_win,
        }

    def _compute_timeline_progression(
        self,
        model: UserCredibilityTemporalTreeTransformer,
        h_tree: torch.Tensor,
        tw: torch.Tensor,
        num_windows: torch.Tensor,
        timestamps: np.ndarray,
        use_uc: bool = False,
        root_user_emb: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Tính toán xác suất phán đoán qua từng lát cắt thời gian tích lũy w = 0, 1, ..., W-1.
        Phản ánh năng lực Phát hiện sớm (Early Rumor Detection) của mô hình.
        """
        total_w = int(num_windows[0].item())
        root_repr = h_tree[:, 0, :]
        root_time = timestamps[0] if len(timestamps) > 0 and timestamps[0] > 0 else 0
        
        timeline_points = []
        all_window_embeddings = []
        
        for w in range(total_w):
            mask_w = (tw <= w).unsqueeze(-1).float()
            denom = mask_w.sum(dim=1).clamp(min=1e-6)
            pooled_w = (h_tree * mask_w).sum(dim=1) / denom
            all_window_embeddings.append(pooled_w)
            
        seq_full = torch.stack(all_window_embeddings, dim=1) # [1, total_w, d_model]
        
        for w in range(total_w):
            current_seq = seq_full[:, :w+1, :]
            cur_len = torch.tensor([w + 1], device=self.device, dtype=torch.long)
            h_temporal = model.temporal_gru(current_seq, lengths=cur_len)
            
            feats = [root_repr, h_temporal]
            if use_uc and model.use_root_user_shortcut and root_user_emb is not None:
                feats.append(root_user_emb)
                
            logits = model.classifier(torch.cat(feats, dim=-1))
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
            
            nodes_in_w = np.where(tw[0].cpu().numpy() <= w)[0]
            if len(nodes_in_w) > 0 and root_time > 0:
                max_t = np.max(timestamps[nodes_in_w])
                elapsed_minutes = max(0.0, (max_t - root_time) / 60.0)
            else:
                elapsed_minutes = w * 15.0
                
            timeline_points.append({
                "window_idx": w,
                "elapsed_minutes": float(elapsed_minutes),
                "elapsed_label": f"{int(elapsed_minutes)}m" if elapsed_minutes < 60 else f"{elapsed_minutes/60:.1f}h",
                "cum_posts": len(nodes_in_w),
                "prob_non_rumour": float(probs[0]),
                "prob_rumour": float(probs[1]),
                "pred_label": int(np.argmax(probs)),
            })
            
        return timeline_points

    def predict_thread(self, thread: Dict[str, Any], fold_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        Thực hiện đánh giá toàn diện một thread theo chuẩn bài báo & output mở rộng:
          1. Thông tin đầu vào chuẩn bài báo (Source Claim, Cây Cascade, Embeddings)
          2. Dự đoán TTT Baseline (Wu et al. 2025)
          3. Dự đoán UC-TTT Proposed (Nhóm 6)
          4. Trọng số giải thích của User Credibility (Gate Alpha)
          5. Diễn biến cảnh báo theo thời gian (Early Rumor Detection Curve)
          6. So sánh đối đầu song song và phân tích sửa lỗi
        """
        event = thread.get("event", "ottawashooting")
        if fold_idx is None:
            fold_idx = EVENT_FOLD_MAP.get(event, 3)
            
        model_ttt, model_uc = self.load_models_for_fold(fold_idx)
        inputs = self._prepare_thread_tensors(thread)
        
        texts = inputs["text_features"]
        depths = inputs["depths"]
        users = inputs["user_features"]
        adj = inputs["adj_mask"]
        tw = inputs["time_window_assignments"]
        nw = inputs["num_windows"]
        timestamps = inputs["timestamps"]
        
        with torch.no_grad():
            # ==========================================
            # 1. Forward Pass: TTT Baseline
            # ==========================================
            h_text_ttt = model_ttt.text_proj(texts) + model_ttt.depth_enc(depths)
            tree_attn_ttt = torch.zeros(adj.shape, dtype=texts.dtype, device=self.device).masked_fill(~adj, float("-inf")).repeat_interleave(model_ttt.nhead, dim=0)
            h_tree_ttt = model_ttt.tree_transformer(h_text_ttt, attn_mask=tree_attn_ttt)
            
            ttt_timeline = self._compute_timeline_progression(
                model_ttt, h_tree_ttt, tw, nw, timestamps, use_uc=False
            )
            final_ttt = ttt_timeline[-1]
            pred_ttt = final_ttt["pred_label"]
            prob_ttt = [final_ttt["prob_non_rumour"], final_ttt["prob_rumour"]]
            
            # ==========================================
            # 2. Forward Pass: UC-TTT Proposed
            # ==========================================
            h_text_uc = model_uc.text_proj(texts) + model_uc.depth_enc(depths)
            u_all = model_uc.user_encoder.encode_user(users)
            root_user_emb = u_all[:, 0, :]
            
            # Trích xuất Gate Alpha (Hệ số điều biến đóng góp User vs Text)
            combined_all = torch.cat([h_text_uc, u_all], dim=-1)
            alpha_all = model_uc.user_encoder.gate(combined_all) # [1, N, d_model]
            
            alpha_root_vec = alpha_all[0, 0, :].cpu().numpy()
            gate_root_mean = float(alpha_root_vec.mean())
            gate_overall_mean = float(alpha_all.mean().item())
            
            h_node_fused = h_text_uc + alpha_all * u_all
            tree_attn_uc = torch.zeros(adj.shape, dtype=texts.dtype, device=self.device).masked_fill(~adj, float("-inf")).repeat_interleave(model_uc.nhead, dim=0)
            h_tree_uc = model_uc.tree_transformer(h_node_fused, attn_mask=tree_attn_uc)
            
            uc_timeline = self._compute_timeline_progression(
                model_uc, h_tree_uc, tw, nw, timestamps, use_uc=True, root_user_emb=root_user_emb
            )
            final_uc = uc_timeline[-1]
            pred_uc = final_uc["pred_label"]
            prob_uc = [final_uc["prob_non_rumour"], final_uc["prob_rumour"]]

        # Nhãn thực tế (Ground Truth)
        gt_label = int(thread.get("label", -1))
        
        # Phân tích tác động đối đầu giữa 2 mô hình
        if gt_label != -1:
            if pred_ttt != gt_label and pred_uc == gt_label:
                impact_status = "CORRECTED_BY_UC"
                if gt_label == 0:
                    impact_title = "🎯 SỬA ĐÚNG NGOẠN MỤC: TRÁNH BÁO ĐỘNG GIẢ (False Rumour Alarm Prevented)"
                    impact_desc = "TTT gốc bị đánh lừa bởi từ ngữ căng thẳng giật gân nên dự đoán là TIN ĐỒN. Nhưng mô hình UC-TTT nhờ nhận diện tác giả là phóng viên/tài khoản uy tín cao nên đã khôi phục đúng phán đoán TIN XÁC THỰC."
                else:
                    impact_title = "🎯 SỬA ĐÚNG NGOẠN MỤC: BÓC MẼ TIN ĐỒN NÚP BÓNG (Stealth Rumour Unmasked)"
                    impact_desc = "Tin đồn sử dụng văn phong trang trọng hành chính đánh lừa TTT gốc, nhưng UC-TTT kết hợp phản hồi hoài nghi của người dùng bên dưới cây đã bóc mẽ chính xác TIN ĐỒN."
            elif pred_ttt == gt_label and pred_uc == gt_label:
                impact_status = "BOTH_CORRECT"
                impact_title = "✅ CẢ HAI MÔ HÌNH ĐỀU PHÁN ĐOÁN CHÍNH XÁC"
                impact_desc = "Cả TTT gốc và UC-TTT đều nhận định trùng khớp nhãn thực tế của luồng thông tin."
            elif pred_ttt == gt_label and pred_uc != gt_label:
                impact_status = "TTT_BETTER"
                impact_title = "⚠️ TTT GỐC CHÍNH XÁC HƠN Ở TRƯỜNG HỢP NÀY"
                impact_desc = "Tín hiệu người dùng ở luồng này có độ nhiễu cao, TTT gốc nắm bắt ngữ nghĩa tốt hơn."
            else:
                impact_status = "BOTH_WRONG"
                impact_title = "❌ CẢ HAI MÔ HÌNH ĐỀU CHƯA PHÁN ĐOÁN ĐÚNG"
                impact_desc = "Ca lan truyền phức tạp, tín hiệu ngữ nghĩa lẫn uy tín người dùng đều chứa nhiều nhiễu."
        else:
            impact_status = "UNANNOTATED"
            impact_title = "ℹ️ DỮ LIỆU TỰ DO (INTERACTIVE SANDBOX)"
            impact_desc = "Đánh giá thời gian thực trên dữ liệu mô phỏng do người dùng tùy biến."

        src_raw_u = thread["source_tweet"].get("raw_user", thread["source_tweet"].get("user", {}))
        cred_score = self.user_extractor.compute_credibility_score(src_raw_u)
        
        user_weight_pct = round(gate_root_mean * 100.0, 1)
        text_weight_pct = round((1.0 - gate_root_mean) * 100.0, 1)
        
        if src_raw_u.get("verified", False) or cred_score >= 70:
            exp_text = f"Mô hình phân bổ {user_weight_pct}% trọng số vào độ uy tín tài khoản và {text_weight_pct}% vào ngữ nghĩa văn bản. Tài khoản có tick xanh và chỉ số tương tác cao giúp củng cố độ tin cậy của thông tin."
        else:
            exp_text = f"Mô hình phân bổ {user_weight_pct}% trọng số vào độ uy tín tài khoản và {text_weight_pct}% vào ngữ nghĩa văn bản. Tài khoản chưa xác minh nên mô hình thận trọng dựa chủ yếu vào ngữ nghĩa và phản hồi."

        return {
            "thread_id": str(thread.get("thread_id", "")),
            "event": event,
            "ground_truth": gt_label,
            "ground_truth_name": LABEL_NAMES.get(gt_label, "Chưa gán nhãn"),
            "source_tweet": {
                "id": str(thread["source_tweet"].get("id", "")),
                "author_name": src_raw_u.get("name", "Unknown"),
                "screen_name": src_raw_u.get("screen_name", "user"),
                "verified": bool(src_raw_u.get("verified", False)),
                "followers_count": int(src_raw_u.get("followers_count", 0) or 0),
                "friends_count": int(src_raw_u.get("friends_count", 0) or 0),
                "reputation_ratio": float(src_raw_u.get("followers_count", 0) or 0) / (float(src_raw_u.get("friends_count", 0) or 0) + 1.0),
                "statuses_count": int(src_raw_u.get("statuses_count", 0) or 0),
                "credibility_score": cred_score,
                "text": thread["source_tweet"].get("text", ""),
                "created_at": thread["source_tweet"].get("created_at", ""),
                "description": src_raw_u.get("description", ""),
            },
            "num_nodes": len(inputs["nodes"]),
            "num_reactions": len(thread.get("reactions", [])),
            "ttt_prediction": {
                "label": pred_ttt,
                "label_name": LABEL_NAMES.get(pred_ttt, "Unknown"),
                "prob_non_rumour": float(prob_ttt[0]),
                "prob_rumour": float(prob_ttt[1]),
                "is_correct": (pred_ttt == gt_label) if gt_label != -1 else None,
            },
            "ucttt_prediction": {
                "label": pred_uc,
                "label_name": LABEL_NAMES.get(pred_uc, "Unknown"),
                "prob_non_rumour": float(prob_uc[0]),
                "prob_rumour": float(prob_uc[1]),
                "is_correct": (pred_uc == gt_label) if gt_label != -1 else None,
            },
            "interpretability": {
                "gate_root_alpha": gate_root_mean,
                "gate_overall_alpha": gate_overall_mean,
                "user_weight_pct": user_weight_pct,
                "text_weight_pct": text_weight_pct,
                "explanation_text": exp_text,
            },
            "timeline": {
                "ttt": ttt_timeline,
                "ucttt": uc_timeline,
            },
            "impact": {
                "status": impact_status,
                "title": impact_title,
                "description": impact_desc,
            }
        }
