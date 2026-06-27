"""
Stroke Segmenter — 叠写笔画切分引擎 (纯 DP, 不依赖 MLP 模型).

核心思路:
  用单字分类模型对所有可能的笔画分组打分,
  通过经典 DP 找最优切分路径, 无需额外训练切分模型.

架构:
  ┌─────────────────────────────────────────────────┐
  │  用户连续书写 (叠写)                             │
  │       ↓                                         │
  │  枚举所有可能的笔画分组 (j, i)                    │
  │       ↓                                         │
  │  每组用单字模型打分 → seg_score                  │
  │       ↓                                         │
  │  DP: dp[i] = max(dp[j] + seg_score + 惩罚)       │
  │       ↓                                         │
  │  回溯最优切分路径 + 组合替代字 → 候选文本          │
  └─────────────────────────────────────────────────┘
"""

import math
from dataclasses import dataclass, field
from itertools import product


@dataclass
class SegmentedChar:
    """一个分割出的字."""
    strokes: list[list[tuple[int, int]]]
    text: str = ""
    confidence: float = 0.0
    candidates: list[tuple[str, float]] = field(default_factory=list)


class SegmentEngine:
    """叠写实时分割引擎 (纯 DP, 不需要训练)."""

    def __init__(
        self,
        single_char_model,
        chars: list[str],
        max_strokes_per_char: int = 25,
        beam_width: int = 5,
    ):
        self.single_char_model = single_char_model
        self.chars = chars
        self.max_strokes_per_char = max_strokes_per_char
        self.beam_width = beam_width

        # 当前缓冲笔画 + 时间戳
        self.buffer_strokes: list[list[tuple[int, int]]] = []
        self.buffer_times: list[float] = []

        # 已确认输出的字符
        self.confirmed_text: str = ""

        # 最新分割结果
        self.latest_chars: list[SegmentedChar] = []

    def add_stroke(
        self, stroke: list[tuple[int, int]], timestamp: float | None = None
    ) -> list[SegmentedChar]:
        """添加一笔, 返回当前分割出的字符列表."""
        import time
        if timestamp is None:
            timestamp = time.time()
        self.buffer_strokes.append(stroke)
        self.buffer_times.append(timestamp)
        self._segment()
        return self.latest_chars

    def _segment(self):
        """对缓冲笔画执行分割 (经典 DP)."""
        strokes = self.buffer_strokes
        n = len(strokes)
        if n == 0:
            self.latest_chars = []
            return

        # ── DP 表 ──
        dp: list[tuple[float, int]] = [(0.0, 0)]
        seg_scores: dict[tuple[int, int], float] = {}

        for i in range(1, n + 1):
            best_score = float('-inf')
            best_j = 0
            for j in range(max(0, i - self.max_strokes_per_char), i):
                key = (j, i)
                if key not in seg_scores:
                    seg_scores[key] = self._score_segment(strokes[j:i])
                score = seg_scores[key]
                char_penalty = 0.0 if j == 0 else -0.3
                total = dp[j][0] + score + char_penalty
                if total > best_score:
                    best_score = total
                    best_j = j
            dp.append((best_score, best_j))

        # ── 回溯最优路径 ──
        cuts = []
        pos = n
        while pos > 0:
            cuts.append(pos)
            pos = dp[pos][1]
        cuts.append(0)
        cuts.reverse()

        # ── 收集备选路径 ──
        beam_paths: list[tuple[float, list[int]]] = [(dp[n][0], list(cuts))]
        for perturb_idx in range(1, len(cuts) - 1):
            alt_cuts = list(cuts)
            alt_cuts.pop(perturb_idx)
            score = 0.0
            for k in range(len(alt_cuts) - 1):
                j, i = alt_cuts[k], alt_cuts[k + 1]
                key = (j, i)
                if key in seg_scores:
                    score += seg_scores[key]
                    if k > 0:
                        score -= 0.3
            beam_paths.append((score, alt_cuts))

        beam_paths.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        unique_beams = []
        for score, path in beam_paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique_beams.append((score, path))
                if len(unique_beams) >= self.beam_width * 2:
                    break
        self._beam_candidates = unique_beams

        # ── 提取最优切分结果 ──
        best_cuts = cuts
        chars = []
        for k in range(len(best_cuts) - 1):
            seg_strokes = strokes[best_cuts[k]:best_cuts[k + 1]]
            if not seg_strokes:
                continue
            result = self._classify_segment(seg_strokes)
            chars.append(result)
        self.latest_chars = chars

    def get_candidates(self, top_k: int = 10) -> list[tuple[str, float]]:
        """返回候选文本序列."""
        if not hasattr(self, '_beam_candidates') or not self._beam_candidates:
            return [(self.get_text(), 0.0)]

        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        strokes = self.buffer_strokes

        path_candidates: list[tuple[float, list[list[tuple[str, float]]]]] = []
        for log_prob, cuts in self._beam_candidates:
            segments = []
            for k in range(len(cuts) - 1):
                seg_strokes = strokes[cuts[k]:cuts[k + 1]]
                if not seg_strokes:
                    continue
                result = self._classify_segment(seg_strokes)
                segments.append(result.candidates if result.candidates else [])
            path_candidates.append((log_prob, segments))

        for log_prob, segments in path_candidates:
            texts = [seg[0][0] for seg in segments if seg]
            full_text = "".join(texts)
            prob = math.exp(log_prob) if texts else 0.0
            if full_text and full_text not in seen:
                seen.add(full_text)
                results.append((full_text, prob))

        if path_candidates:
            _, best_segments = path_candidates[0]
            alt_lists = [seg[:3] if seg else [("?", 0.0)] for seg in best_segments]
            max_combo = min(len(alt_lists), 2)
            if max_combo > 0:
                for combo in product(*alt_lists[:max_combo]):
                    texts = [ch for ch, _ in combo]
                    for seg in alt_lists[max_combo:]:
                        if seg:
                            texts.append(seg[0][0])
                    full_text = "".join(texts)
                    if full_text not in seen:
                        seen.add(full_text)
                        prob = 1.0
                        for _, p in combo:
                            prob *= p
                        for seg in alt_lists[max_combo:]:
                            if seg:
                                prob *= seg[0][1]
                        results.append((full_text, prob))

        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]
        if not results:
            results.append((self.get_text(), 0.0))
        return results

    def _score_segment(self, segment_strokes) -> float:
        """对一组笔画打分, 返回 log 概率."""
        if not segment_strokes:
            return -10.0
        from .demo import predict as single_predict
        results = single_predict(
            self.single_char_model, self.chars,
            segment_strokes, top_k=1,
        )
        if results:
            return math.log(max(results[0][1], 1e-10))
        return -5.0

    def _classify_segment(self, segment_strokes) -> SegmentedChar:
        """对一组笔画执行单字识别."""
        from .demo import predict as single_predict
        results = single_predict(
            self.single_char_model, self.chars,
            segment_strokes, top_k=10,
        )
        if results:
            return SegmentedChar(
                strokes=segment_strokes,
                text=results[0][0],
                confidence=results[0][1],
                candidates=results,
            )
        return SegmentedChar(strokes=segment_strokes, text="?", confidence=0.0)

    def get_text(self) -> str:
        """获取当前识别的文本序列."""
        if self.confirmed_text:
            return self.confirmed_text + "".join(ch.text for ch in self.latest_chars)
        return "".join(ch.text for ch in self.latest_chars)

    def confirm(self, candidate_idx: int = 0) -> str:
        """确认当前结果, 追加到已确认文本, 清空缓冲."""
        full_text = "".join(
            ch.candidates[candidate_idx][0] if ch.candidates else ch.text
            for ch in self.latest_chars
        ) if self.latest_chars else ""
        self.confirmed_text += full_text
        self.buffer_strokes = []
        self.buffer_times = []
        self.latest_chars = []
        return self.confirmed_text

    def clear(self):
        """清空所有状态."""
        self.buffer_strokes = []
        self.buffer_times = []
        self.latest_chars = []
        self.confirmed_text = ""
