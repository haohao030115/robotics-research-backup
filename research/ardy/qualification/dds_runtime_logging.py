"""Redirect only DDS diagnostic logs; network, controller and physics unchanged."""
import os
from pathlib import Path
from xml.sax.saxutils import escape
def configure_dds_logging(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    from unitree_sdk2py.core import channel
    python_log=str(output/'dds_python.log')
    channel.ChannelConfigHasInterface=channel.ChannelConfigHasInterface.replace('/tmp/cdds.LOG',escape(python_log))
    native_log=str(output/'dds_sonic.log')
    os.environ['CYCLONEDDS_URI']='<CycloneDDS><Domain Id="any"><Tracing><Verbosity>config</Verbosity><OutputFile>'+escape(native_log)+'</OutputFile></Tracing></Domain></CycloneDDS>'
    return dict(python_log=python_log,native_log=native_log,network_interface='lo',domain_id=0,SDK_bytes_modified=False)
