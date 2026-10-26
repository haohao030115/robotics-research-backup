"""Isolate the ORIGINAL native controller; apply metadata/logging-only instrumentation.

Never overwrites the canonical executable or controller source. Fail if hook anchors change.
"""
import difflib
import shutil
import subprocess
from pathlib import Path
from .common import ROOT, SONIC, digest, dump, require

BUILD_ROOT=ROOT/'canonical_va/native_build'
DEPLOY=SONIC/'gear_sonic_deploy'
HEADER=r'''
#pragma once
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <stdexcept>
inline uint64_t canonical_ns() {
 return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline std::ofstream& canonical_stream(bool publication) {
 static std::ofstream decisions, publications;
 auto& out=publication?publications:decisions;
 if (!out.is_open()) {
  const char* dir=std::getenv("CANONICAL_TRACE_DIR");
  if (!dir) throw std::runtime_error("CANONICAL_TRACE_DIR required for instrumented build");
  out.open(std::string(dir)+(publication?"/publications.bin":"/decisions.bin"),std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open canonical trace");
 }
 return out;
}
template<class Range> inline void canonical_floats(std::ofstream& out,const Range& values) {
 std::vector<float> v; for(auto x:values) v.push_back(static_cast<float>(x));
 out.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(float));
}
'''

def prepare():
    require(not BUILD_ROOT.exists(),'Native build exists; do not overwrite its provenance')
    BUILD_ROOT.mkdir()
    for name in ('CMakeLists.txt','cmake','src'):
        p=DEPLOY/name
        if p.is_dir(): shutil.copytree(p,BUILD_ROOT/name)
        else: shutil.copy2(p,BUILD_ROOT/name)
    for name in ('thirdparty','scripts'):
        (BUILD_ROOT/name).symlink_to(DEPLOY/name,target_is_directory=True)
    base=Path('src/g1/g1_deploy_onnx_ref'); changes=[]
    def patch(rel,old,new,count=1):
        p=BUILD_ROOT/base/rel; s=p.read_text(); require(s.count(old)==count,'Instrumentation anchor changed: '+old[:70])
        p.write_text(s.replace(old,new)); changes.append(str(base/rel))
    (BUILD_ROOT/base/'include/canonical_trace.hpp').write_text(HEADER)
    patch('include/robot_parameters.hpp','struct MotorCommand {','struct MotorCommand {\n    uint32_t canonical_decision_id = 0; // bookkeeping only; not a motor field')
    patch('src/g1_deploy_onnx_ref.cpp','#include <cmath>','#include <cmath>\n#include "canonical_trace.hpp"')
    patch('src/g1_deploy_onnx_ref.cpp','DataBuffer<LowState_> low_state_buffer_;',
          'uint64_t canonical_frame_=0, canonical_playing_=0, canonical_observe_ns_=0, canonical_encode_start_=0, canonical_encode_end_=0;\n    DataBuffer<LowState_> low_state_buffer_;')
    patch('src/g1_deploy_onnx_ref.cpp','used_low_state_data_ = (low_state_data);','used_low_state_data_ = (low_state_data);\n      canonical_observe_ns_=canonical_ns();')
    patch('src/g1_deploy_onnx_ref.cpp','current_play_copy = operator_state.play;',
          'current_play_copy = operator_state.play;\n            canonical_frame_=current_frame_copy; canonical_playing_=current_play_copy;')
    patch('src/g1_deploy_onnx_ref.cpp','if (!encoder_engine_->Encode()) {','canonical_encode_start_=canonical_ns();\n      if (!encoder_engine_->Encode()) {')
    patch('src/g1_deploy_onnx_ref.cpp','auto& token_buffer = encoder_engine_->GetTokenBuffer();','canonical_encode_end_=canonical_ns();\n      auto& token_buffer = encoder_engine_->GetTokenBuffer();')
    patch('src/g1_deploy_onnx_ref.cpp','bool CreatePolicyCommand() {','bool CreatePolicyCommand() {\n      const auto canonical_previous_action=last_action;\n      const auto canonical_decode_start=canonical_ns();')
    old='apply_motor_gain_scales(motor_gain_scales_, motor_command_tmp);\n      motor_command_buffer_.SetData(motor_command_tmp);'
    # Anchor only inside CreatePolicyCommand; InitControl also uses gain scaling.
    p=BUILD_ROOT/base/'src/g1_deploy_onnx_ref.cpp'; s=p.read_text(); start=s.index('bool CreatePolicyCommand() {'); end=s.index('\n    /**',start)
    part=s[start:end]; require(part.count(old)==1,'CreatePolicyCommand anchor')
    new='''apply_motor_gain_scales(motor_gain_scales_, motor_command_tmp);
      motor_command_tmp.canonical_decision_id=static_cast<uint32_t>(logging_counter_);
      {
        auto& trace=canonical_stream(false);
        const auto ls=used_low_state_data_.data;
        const uint64_t header[]={static_cast<uint64_t>(logging_counter_),ls->reserve()[0],ls->tick(),canonical_frame_,canonical_playing_,canonical_observe_ns_,canonical_encode_start_,canonical_encode_end_,canonical_decode_start,canonical_ns()};
        trace.write(reinterpret_cast<const char*>(header),sizeof(header));
        canonical_floats(trace,encoder_engine_->GetInputBuffer());
        canonical_floats(trace,policy_engine_->GetInputBuffer());
        canonical_floats(trace,encoder_engine_->GetTokenBuffer());
        canonical_floats(trace,policy_engine_->GetActionBuffer());
        canonical_floats(trace,canonical_previous_action);
        canonical_floats(trace,motor_command_tmp.q_target);
        canonical_floats(trace,motor_command_tmp.dq_target);
        canonical_floats(trace,motor_command_tmp.kp);
        canonical_floats(trace,motor_command_tmp.kd);
        canonical_floats(trace,motor_command_tmp.tau_ff);
        trace.flush();
      }
      motor_command_buffer_.SetData(motor_command_tmp);'''
    p.write_text(s[:start]+part.replace(old,new)+s[end:])
    patch('src/g1_deploy_onnx_ref.cpp','dds_low_command.crc() = Crc32Core',
          '''static uint32_t canonical_publication_id=0;
        dds_low_command.reserve()[0]=mc->canonical_decision_id;
        dds_low_command.reserve()[1]=++canonical_publication_id;
        auto& trace=canonical_stream(true);
        const uint64_t header[]={mc->canonical_decision_id,canonical_publication_id,canonical_ns()};
        trace.write(reinterpret_cast<const char*>(header),sizeof(header)); trace.flush();
        dds_low_command.crc() = Crc32Core''')
    diff=''; hashes={}
    for rel in sorted(set(changes)):
        original=DEPLOY/rel; modified=BUILD_ROOT/rel
        hashes[rel]={'original':digest(original),'instrumented':digest(modified)}
        diff+=''.join(difflib.unified_diff(original.read_text().splitlines(True),modified.read_text().splitlines(True),fromfile='original/'+rel,tofile='instrumented/'+rel))
    (BUILD_ROOT/'instrumentation.patch').write_text(diff)
    dump(BUILD_ROOT/'sources.json',dict(source_deploy=str(DEPLOY),changes=hashes,original_binary_sha256=digest(DEPLOY/'target/release/g1_deploy_onnx_ref'),
        semantics='Only new trace writes, MotorCommand metadata, LowCmd reserved fields; weights/controller arithmetic/thread periods unchanged'))


def main():
    prepare()
    subprocess.run(['cmake','-S',str(BUILD_ROOT),'-B',str(BUILD_ROOT/'build'),'-DSONIC_MUSA=ON','-DCMAKE_BUILD_TYPE=Release'],check=True)
    subprocess.run(['cmake','--build',str(BUILD_ROOT/'build'),'-j','4'],check=True)
    dump(BUILD_ROOT/'build_validation.json',dict(binary=str(BUILD_ROOT/'target/release/g1_deploy_onnx_ref'),sha256=digest(BUILD_ROOT/'target/release/g1_deploy_onnx_ref')))

if __name__=='__main__': main()
