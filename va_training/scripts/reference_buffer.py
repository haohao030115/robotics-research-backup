"""Shared, absolute 50 Hz body/hand reference timeline for SONIC streaming."""
import numpy as np


class FutureReferenceBuffer:
    LOOKAHEAD = 46
    COMMIT_LEAD = 4  # Protect packets already in flight to SONIC's 100 Hz input.
    RESERVE = 64

    def __init__(self):
        self.q = np.empty((0, 43), dtype=np.float32)
        self.source = np.empty(0, dtype=np.int64)
        self.end = -1
        self.last_packet_start = -1
        self.cursor = -1
        self.stale_chunks = 0

    def replenish(self, anchor, prediction):
        prediction = np.asarray(prediction, dtype=np.float32)
        if prediction.shape != (50, 43) or not np.isfinite(prediction).all():
            raise ValueError('Expected finite [50,43] VA references')
        first = self.end < 0
        start = 0 if first else max(anchor, self.cursor+self.COMMIT_LEAD,
                                    self.last_packet_start+1)
        stop = anchor+len(prediction)
        if first and anchor != 0:
            raise ValueError('The initial prediction must start at reference row 0')
        if start >= stop:
            self.stale_chunks += 1
            return None
        # SONIC resets if either packet endpoint does not strictly increase.
        end = max(self.end+1, stop-1+self.LOOKAHEAD+self.RESERVE)
        old_end = self.end
        if end >= len(self.q):
            added = end+1-len(self.q)
            self.q = np.concatenate((self.q, np.empty((added, 43), np.float32)))
            self.source = np.concatenate((self.source, np.full(added, -1, np.int64)))
        values = prediction[start-anchor:].copy()
        # Blend only uncommitted future overlap; executed and in-flight frames
        # remain byte-identical. Body and hand use the same blend weights.
        overlap = min(5, len(values), max(0, old_end-start+1))
        if overlap:
            weight = (np.arange(1, overlap+1)/overlap)[:, None]
            values[:overlap] = self.q[start:start+overlap]*(1-weight)+values[:overlap]*weight
        self.q[start:stop] = values
        self.source[start:stop] = anchor
        # Endpoint holding is explicitly beyond the model horizon, never an
        # invented future prediction. These rows satisfy SONIC's lookahead while
        # the next asynchronous live prediction replenishes the actual horizon.
        self.q[stop:end+1] = prediction[-1]
        self.source[stop:end+1] = -1
        self.end = end
        self.last_packet_start = start
        return start, self.q[start:end+1].copy()

    def consume(self, body_reference):
        """Advance from SONIC's consumed target, never from elapsed wall time.

        Target CSV is the native control snapshot, in the same order as its
        50 Hz motion_playing log. At a genuine stall the hand cursor stays put.
        """
        body_reference = np.asarray(body_reference)
        next_frame = self.cursor+1
        error = float(np.max(np.abs(body_reference-self.q[next_frame, :29])))
        if error <= 1e-5:
            self.cursor = next_frame
            return True, error
        if self.cursor >= 0:
            current_error = float(np.max(np.abs(body_reference-self.q[self.cursor, :29])))
            if current_error <= 1e-5:
                return False, current_error
        raise RuntimeError(f'SONIC reference discontinuity at {next_frame}: {error:.6g} rad')

    def hand(self):
        return self.q[max(0, self.cursor), 29:]
