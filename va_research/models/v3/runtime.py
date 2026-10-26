"""Current observation capture and exact two-slot scheduler for native SONIC."""
import queue
import threading
import time
import traceback
import numpy as np
from va_research.datasets.tokens import projected_gravity
from .data import targets


def current_state(runtime):
    data=runtime.data
    quat=data.qpos[runtime.root_qa+3:runtime.root_qa+7].copy()
    return dict(body_q=data.qpos[runtime.body.qa].copy(),body_dq=data.qvel[runtime.body.va].copy(),
        hand_q=data.qpos[runtime.hand.qa].copy(),hand_dq=data.qvel[runtime.hand.va].copy(),
        base_quat=quat,base_ang_vel=data.qvel[runtime.root_va+3:runtime.root_va+6].copy(),
        projected_gravity=projected_gravity(quat))


class TwoTickBuffer:
    """One stateless VA decision per two ticks; never shift a late chunk."""
    def __init__(self):
        self.entries={};self.expired=0;self.overwrites=0;self.empty_polls=0
        self.last_anchor=None

    def add(self,prediction,current_tick,start=None):
        anchor=prediction['anchor']
        if anchor < -1 or anchor%2!=1 or (self.last_anchor is not None and anchor!=self.last_anchor+2):
            raise ValueError('V3 decisions must be -1,1,3,...')
        if prediction['execution_index']!=anchor+1 or (start is not None and start!=anchor+1):
            raise ValueError('V3 must execute at observation anchor +1,+2')
        if np.shape(prediction['token'])!=(2,64) or np.shape(prediction['hand'])!=(2,14):
            raise ValueError('V3 requires exactly two body/hand slots')
        if not np.isfinite(prediction['token']).all() or not np.isfinite(prediction['hand']).all():
            raise ValueError('Non-finite prediction')
        self.last_anchor=anchor
        for slot,tick in enumerate(targets(anchor)):
            tick=int(tick)
            if tick<=current_tick: self.expired+=1; continue
            if tick in self.entries: raise RuntimeError('Overlapping V3 chunks')
            self.entries[tick]={**prediction,'token':prediction['token'][slot].copy(),
                'hand':prediction['hand'][slot].copy(),'execution_index':tick,'chunk_index':slot,'chunk_start':anchor+1}

    def take(self,tick):
        for old in [k for k in self.entries if k<tick]:del self.entries[old];self.expired+=1
        result=self.entries.pop(tick,None)
        if result is None:self.empty_polls+=1
        return result


class CurrentVisionWorker:
    def __init__(self,model,policy,render_shape=(120,160)):
        self.model=model;self.policy=policy;self.render_shape=render_shape
        self.requests=queue.Queue();self.results=queue.Queue()
        self.stop=threading.Event();self.ready=threading.Event();self.error=None
        self.renderer_info={};self.dropped_requests=0
        self.thread=threading.Thread(target=self.work,daemon=True);self.thread.start()

    def submit(self,anchor,data,state):
        if anchor%2!=1:raise ValueError('Invalid decision anchor')
        self.requests.put(dict(anchor=anchor,execution_index=anchor+1,qpos=data.qpos.copy(),qvel=data.qvel.copy(),
            state=state,requested=time.monotonic(),observation_sim_timestamp=float(data.time),
            rgb_sim_timestamp=float(data.time),proprio_sim_timestamp=float(data.time)))

    def work(self):
        renderer=None
        try:
            import mujoco
            from OpenGL import GL
            data=mujoco.MjData(self.model);renderer=mujoco.Renderer(self.model,height=self.render_shape[0],width=self.render_shape[1])
            self.renderer_info={k:GL.glGetString(v).decode() for k,v in [('vendor',GL.GL_VENDOR),('renderer',GL.GL_RENDERER),('version',GL.GL_VERSION)]}
            if hasattr(self,'information'):self.information.put(self.renderer_info)
            # Initialize graphics shaders and operator kernels before the reset
            # observation. This is compute warm-up, never policy/history padding.
            from .data import FIELDS, WIDTHS
            dummy={k:np.zeros(w,dtype=np.float32) for k,w in zip(FIELDS,WIDTHS)}
            for _ in range(5):
                mujoco.mj_forward(self.model,data);renderer.update_scene(data,camera='head_rgb')
                self.policy.predict(renderer.render().copy(),dummy)
            self.ready.set()
            while not self.stop.is_set():
                try:r=self.requests.get(timeout=.1)
                except queue.Empty:continue
                begin=time.monotonic();data.qpos[:]=r.pop('qpos');data.qvel[:]=r.pop('qvel')
                mujoco.mj_forward(self.model,data);renderer.update_scene(data,camera='head_rgb');rgb=renderer.render().copy()
                rendered=time.monotonic();out,timing=self.policy.predict(rgb,r.pop('state'));finished=time.monotonic()
                timing['rgb_acquisition_render']=(rendered-begin)*1000
                timing['observation_to_prediction']=(finished-r['requested'])*1000
                r.update(token=out['sonic_token'],hand=out['hand_ref'],finished=finished,
                    inference_start=rendered,inference_end=finished,render_seconds=rendered-begin,
                    inference_seconds=finished-rendered,total_seconds=finished-r['requested'],latency_ms=timing)
                self.results.put(r)
        except Exception:
            self.error=traceback.format_exc()
            if hasattr(self,'errors'):self.errors.put(self.error)
            self.ready.set()
        finally:
            if renderer is not None:renderer.close()

    def close(self):
        self.stop.set();self.thread.join(timeout=30)


def _vision_process(model_blob, weights, device, render_shape, requests, results, ready, stop, information, errors):
    import pickle
    from types import SimpleNamespace
    import torch
    from .model import V3Policy
    from .inference import CurrentPolicy
    torch.set_num_threads(1)
    if device.startswith('musa'):
        import torch_musa
        torch.musa.set_device(torch.device(device))
    try:
        model=pickle.loads(model_blob)
        network=V3Policy().to(device)
        network.load_state_dict({k:torch.from_numpy(v).to(device) for k,v in weights.items()})
        worker=SimpleNamespace(model=model,policy=CurrentPolicy(network),requests=requests,results=results,
            ready=ready,stop=stop,error=None,renderer_info={},information=information,errors=errors,render_shape=render_shape)
        CurrentVisionWorker.work(worker)
    except Exception:
        errors.put(traceback.format_exc());ready.set()


class ProcessVisionWorker:
    """Own graphics/inference process avoids Python GIL contention with physics."""
    def __init__(self,model,policy,render_shape=(120,160)):
        import multiprocessing as mp
        import pickle
        context=mp.get_context('spawn')
        self.requests=context.Queue();self.results=context.Queue()
        self.ready=context.Event();self.stop=context.Event()
        self.information=context.Queue();self.errors=context.Queue()
        self._error=None;self._info={};self.dropped_requests=0
        weights={k:v.detach().cpu().numpy().copy() for k,v in policy.model.state_dict().items()}
        self.process=context.Process(target=_vision_process,args=(pickle.dumps(model),weights,str(policy.device),render_shape,
            self.requests,self.results,self.ready,self.stop,self.information,self.errors))
        self.process.start()

    @property
    def error(self):
        try:self._error=self.errors.get_nowait()
        except queue.Empty:pass
        if not self.process.is_alive() and self.process.exitcode not in (None,0) and self._error is None:
            self._error=f'Vision process exited {self.process.exitcode}'
        return self._error

    @property
    def renderer_info(self):
        try:self._info=self.information.get_nowait()
        except queue.Empty:pass
        return self._info

    submit=CurrentVisionWorker.submit

    def close(self):
        self.stop.set();self.process.join(timeout=10)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
