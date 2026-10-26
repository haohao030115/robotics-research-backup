
#pragma once
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <mutex>
#include <string>
inline uint64_t canonical_ns() {
 return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void canonical_startup_event(const char* name) {
 static std::mutex mutex;
 static std::ofstream stream;
 static bool initialized=false;
 std::lock_guard<std::mutex> guard(mutex);
 if (!initialized) {
  initialized=true;
  const char* path=std::getenv("CANONICAL_STARTUP_TIMING_FILE");
  if (path && *path) { stream.open(path,std::ios::out|std::ios::app); }
 }
 if (stream.is_open()) { stream << name << "\t" << canonical_ns() << "\n"; stream.flush(); }
}
inline void canonical_startup_event_detail(const std::string& name, const std::string& detail) {
 static std::mutex mutex;
 static std::ofstream stream;
 static bool initialized=false;
 std::lock_guard<std::mutex> guard(mutex);
 if (!initialized) {
  initialized=true;
  const char* path=std::getenv("CANONICAL_STARTUP_TIMING_FILE");
  if (path && *path) { stream.open(path,std::ios::out|std::ios::app); }
 }
 if (stream.is_open()) { stream << name << "\t" << canonical_ns() << "\t" << detail << "\n"; stream.flush(); }
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
