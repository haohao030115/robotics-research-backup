"""Absolute-tick coverage only. No implicit startup or stale-output padding."""
import numpy as np


class UncoveredTick(RuntimeError):
    pass


class PlanBuffer:
    def __init__(self):
        self.entries = {}
        self.last_tick = None
        self.latest_start = None
        self.generation = -1
        self.log = []
        self.missed_starts = set()
        self.counts = dict(total_plans=0, newest_plan_ticks=0, overlap_fallback_ticks=0,
                           uncovered_ticks=0, repeated_ticks=0, skipped_ticks=0, deadline_misses=0)

    def add(self, start_tick, body, hand, available_tick):
        body, hand = np.asarray(body), np.asarray(hand)
        if body.shape != (10, 64) or hand.shape != (50, 14) or not np.isfinite(body).all() or not np.isfinite(hand).all():
            raise ValueError('Invalid V4 plan')
        if self.latest_start is not None and start_tick != self.latest_start + 5:
            raise ValueError('Plan starts must advance exactly five absolute ticks')
        self.generation += 1
        self.latest_start = start_tick
        self.counts['total_plans'] += 1
        if available_tick > start_tick:
            self.missed_starts.add(start_tick)
        self.counts['deadline_misses'] = len(self.missed_starts)
        for slot in range(10):
            tick = start_tick + slot
            if tick >= available_tick and (self.last_tick is None or tick > self.last_tick):
                self.entries[tick] = dict(plan_generation_id=self.generation, prediction_slot=slot,
                    intended_tick=tick, token=body[slot].copy(), hand_target=hand[slot].copy())

    def take(self, tick, expected_plan_start):
        if self.last_tick is not None and tick != self.last_tick + 1:
            key = 'repeated_ticks' if tick <= self.last_tick else 'skipped_ticks'
            self.counts[key] += 1
            raise RuntimeError(key)
        entry = self.entries.pop(tick, None)
        if entry is None:
            self.counts['uncovered_ticks'] += 1
            self.missed_starts.add(expected_plan_start)
            self.counts['deadline_misses'] = len(self.missed_starts)
            self.log.append(dict(intended_tick=tick, uncovered_tick=True))
            raise UncoveredTick(f'No valid prediction for absolute control tick {tick}')
        fallback = tick - entry['prediction_slot'] < expected_plan_start
        if fallback:
            self.missed_starts.add(expected_plan_start)
            self.counts['deadline_misses'] = len(self.missed_starts)
        self.counts['overlap_fallback_ticks' if fallback else 'newest_plan_ticks'] += 1
        self.last_tick = tick
        entry.update(overlap_fallback=fallback, uncovered_tick=False, deadline_miss=fallback)
        self.log.append(entry)
        return entry
