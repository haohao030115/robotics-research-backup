from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OPEN = np.array([0., -.1, -.1, .15, .15, .15, .15])
CLOSED = np.array([0., -.8, -1.2, 1.2, 1.3, 1.2, 1.3])


@dataclass
class ContactDrivenHand:
    """Event-driven Dex3 target; no absolute playback time is used."""
    amount: float = 0.0
    rate_per_second: float = 0.8
    contact_latched: bool = False

    def update(self, dt: float, enabled: bool, thumb_contact: bool,
               opposing_contact: bool) -> np.ndarray:
        if not enabled:
            self.contact_latched = False
            self.amount = max(0.0, self.amount - self.rate_per_second * dt)
        elif thumb_contact and opposing_contact:
            # Contact, not playback time, commits the grasp.  The one-shot
            # preload prevents the object slipping when the lift starts.
            if not self.contact_latched:
                self.amount = 1.0
                self.contact_latched = True
        elif not self.contact_latched:
            self.amount = min(1.0, self.amount + self.rate_per_second * dt)
        # Once latched, retain pressure even if a contact momentarily drops.
        smooth = self.amount * self.amount * (3 - 2 * self.amount)
        return OPEN + smooth * (CLOSED - OPEN)
