"""Timestamped paired token/hand buffer, independent of the inference thread."""
import numpy as np


class TokenChunkBuffer:
    def __init__(self):
        self.entries={}; self.expired=0; self.overwrites=0; self.empty_polls=0

    def add(self,prediction,current_tick,start=None):
        tokens=np.asarray(prediction['token']); hands=np.asarray(prediction['hand'])
        if tokens.ndim!=2 or tokens.shape[1]!=64 or hands.shape!=(len(tokens),14):
            raise ValueError('Expected paired [K,64] and [K,14] chunks')
        if not np.isfinite(tokens).all() or not np.isfinite(hands).all(): raise ValueError('Non-finite chunk')
        start=prediction['execution_index'] if start is None else start
        for i,(token,hand) in enumerate(zip(tokens,hands)):
            tick=start+i
            if tick<=current_tick: self.expired+=1; continue
            if tick in self.entries: self.overwrites+=1
            self.entries[tick]={**prediction,'token':token.copy(),'hand':hand.copy(),
                                'execution_index':tick,'chunk_index':i,'chunk_start':start}

    def take(self,tick):
        for old in [k for k in self.entries if k<tick]: del self.entries[old]; self.expired+=1
        result=self.entries.pop(tick,None)
        if result is None: self.empty_polls+=1
        return result
