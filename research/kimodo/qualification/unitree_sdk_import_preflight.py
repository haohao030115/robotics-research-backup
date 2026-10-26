"""Import-only check: no DDS participant, controller, thread or socket creation."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SDK=ROOT/'assets/unitree_sdk2_python'
manifest=json.loads((ROOT/'config/unitree_sdk_source_hashes.json').read_text())
for rel,h in manifest['sha256'].items():
    assert hashlib.sha256((SDK/rel).read_bytes()).hexdigest()==h,rel
# The import must resolve the frozen package, not a shared editable install.
sys.path.insert(0,str(SDK))
def reject_network(event,args):
    if event.startswith('socket.'):raise RuntimeError('Import preflight forbids network operations')
sys.addaudithook(reject_network)
import cyclonedds.domain
original_participant=cyclonedds.domain.DomainParticipant.__init__
original_domain=cyclonedds.domain.Domain.__init__
def prohibit_init(*args,**kwargs):raise RuntimeError('Import preflight forbids DDS initialization')
cyclonedds.domain.DomainParticipant.__init__=prohibit_init
cyclonedds.domain.Domain.__init__=prohibit_init
try:
    import unitree_sdk2py
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_,LowState_
    from unitree_sdk2py.utils.crc import CRC
    path=Path(unitree_sdk2py.__file__).resolve()
    assert path.is_relative_to(SDK.resolve()),str(path)
    assert callable(ChannelFactoryInitialize)
finally:
    cyclonedds.domain.DomainParticipant.__init__=original_participant
    cyclonedds.domain.Domain.__init__=original_domain
print(json.dumps(dict(passed=True,sdk_module=str(path),sdk_hashes_verified=True,DDS_initialized=False,controller_started=False,model_calls=0),indent=2))
