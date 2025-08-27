import numpy as np
from datetime import timedelta

def _timedelta_seconds(td):
    if isinstance(td, np.timedelta64):
        return int(td / np.timedelta64(1, 's'))
    elif isinstance(td, timedelta):
        return int(td.total_seconds())
    else:
        return int(td)
from typing import List, Optional, Tuple
import numpy as np

class TemporalSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _filter_window(products, alert_date: np.datetime64, pre: int, post: int):
        start = alert_date - np.timedelta64(pre,'D')
        end   = alert_date + np.timedelta64(post,'D')
        return [p for p in products if start <= p.date <= end]

    def select_single(self, products, alert_date: np.datetime64):
        if not products: return None
        deltas = [(p, _timedelta_seconds(p.date - alert_date)) for p in products]
        post = [d for d in deltas if d[1] >= 0]
        return (min(post, key=lambda x: x[1])[0] if post else min(deltas, key=lambda x: abs(x[1]))[0])

    def best_pair(self, s2_list, s1_list, alert_date: np.datetime64, max_days: int):
        if not s2_list or not s1_list:
            return self.select_single(s2_list, alert_date), self.select_single(s1_list, alert_date)
        s2w = self._filter_window(s2_list, alert_date, self.cfg.pre_days, self.cfg.post_days)
        s1w = self._filter_window(s1_list, alert_date, self.cfg.pre_days, self.cfg.post_days)
        if not s2w or not s1w:
            return self.select_single(s2_list, alert_date), self.select_single(s1_list, alert_date)
        pairs = []
        for s2 in s2w:
            s1 = min(s1w, key=lambda p: abs(_timedelta_seconds(p.date - s2.date)))
            dt_days = abs((s1.date - s2.date) / np.timedelta64(1, 'D')) if isinstance(s1.date, np.datetime64) and isinstance(s2.date, np.datetime64) else abs((s1.date - s2.date).days)
            is_post = int(_timedelta_seconds(s2.date - alert_date) >= 0) + int(_timedelta_seconds(s1.date - alert_date) >= 0)
            pairs.append((dt_days, -is_post, s2, s1))
        pairs.sort(key=lambda t: (t[0], t[1]))
        best = pairs[0]
        if best[0] <= max_days: return best[2], best[3]
        return self.select_single(s2_list, alert_date), self.select_single(s1_list, alert_date)

    def all_pairs(self, s2_list, s1_list, alert_date: np.datetime64, max_days: int):
        """Return a list of (s2,s1) pairs for every S2 in window, matched to nearest S1
        within max_days (duplicates of S1 allowed)."""
        out = []
        if not s2_list or not s1_list: return out
        s2w = self._filter_window(s2_list, alert_date, self.cfg.pre_days, self.cfg.post_days)
        s1w = self._filter_window(s1_list, alert_date, self.cfg.pre_days, self.cfg.post_days)
        for s2 in s2w:
            if not s1w:
                continue
            s1 = min(s1w, key=lambda p: abs(_timedelta_seconds(p.date - s2.date)))
            dt_days = abs((s1.date - s2.date) / np.timedelta64(1, 'D')) if isinstance(s1.date, np.datetime64) and isinstance(s2.date, np.datetime64) else abs((s1.date - s2.date).days)
            if dt_days <= max_days:
                out.append((s2, s1))
        return out
    
    # def all_pairs(self, s2_list, s1_list, alert_date: np.datetime64, max_days: int):
    #     out = []
    #     if not s2_list or not s1_list:
    #         return out
    #     s2w = self._filter_window(s2_list, alert_date, self.cfg.pre_days, self.cfg.post_days)
    #     s1w = self._filter_window(s1_list, alert_date, self.cfg.pre_days, self.cfg.post_days)
    #     for s2 in s2w:
    #         s1 = min(s1w, key=lambda p: abs(int((p.date - s2.date).astype('timedelta64[s]'))))
    #         dt_days = abs(int((s1.date - s2.date).astype('timedelta64[D]')))
    #         if dt_days <= max_days:
    #             out.append((s2, s1))
    #     return out