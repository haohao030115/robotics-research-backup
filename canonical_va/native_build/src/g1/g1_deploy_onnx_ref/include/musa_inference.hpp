#pragma once
// Preserve the engine buffer API; all controller observation and action code
// remains in the original native loop. Python owns only ONNX graph execution.
#include <Python.h>
#include <vector>
#include <string>
#include <mutex>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include "canonical_trace.hpp"

class MusaInference {
 public:
  MusaInference() = default;
  MusaInference(const MusaInference&) = delete;
  ~MusaInference() { Destroy(); }
  bool Initialize(const std::string& path, const char* output, size_t expected_output, bool fp16) {
    if (fp16) { std::cerr << "MUSA backend requires validated FP32 execution\n"; return false; }
    const std::string role = std::string(output) == "action" ? "policy" : "encoder";
    canonical_startup_event(("musa_" + role + "_initialize_start").c_str());
    static std::once_flag python_once;
    canonical_startup_event(("musa_" + role + "_python_init_wait_start").c_str());
    std::call_once(python_once, [] {
      canonical_startup_event("musa_python_initialize_start");
      Py_Initialize();
      PyObject* py_path = PySys_GetObject("path");
      PyObject* directory = PyUnicode_FromString(SONIC_MUSA_PYTHON_DIR);
      PyList_Append(py_path, directory); Py_DECREF(directory);
      PyEval_SaveThread();
      canonical_startup_event("musa_python_initialize_done");
    });
    canonical_startup_event(("musa_" + role + "_python_init_wait_done").c_str());
    auto gil = PyGILState_Ensure();
    canonical_startup_event(("musa_" + role + "_module_import_start").c_str());
    PyObject* module = PyImport_ImportModule("musa_onnx");
    PyObject* type = module ? PyObject_GetAttrString(module, "Session") : nullptr;
    canonical_startup_event(("musa_" + role + "_module_import_done").c_str());
    canonical_startup_event(("musa_" + role + "_session_create_start").c_str());
    session_ = type ? PyObject_CallFunction(type, "ss", path.c_str(), output) : nullptr;
    canonical_startup_event(("musa_" + role + "_session_create_done").c_str());
    Py_XDECREF(type); Py_XDECREF(module);
    if (session_) {
      canonical_startup_event(("musa_" + role + "_io_metadata_start").c_str());
      PyObject* dim = PyObject_GetAttrString(session_, "input_dimension");
      if (dim) { input_.resize(PyLong_AsSize_t(dim), 0.f); Py_DECREF(dim); }
      PyObject* dim_out = PyObject_GetAttrString(session_, "output_dimension");
      size_t count = dim_out ? PyLong_AsSize_t(dim_out) : 0;
      Py_XDECREF(dim_out);
      if (count != expected_output || input_.empty()) {
        PyErr_SetString(PyExc_ValueError, "SONIC MUSA model I/O dimension mismatch");
      } else { output_.resize(count, 0.f); initialized_ = true; }
      canonical_startup_event(("musa_" + role + "_io_metadata_done").c_str());
    }
    if (PyErr_Occurred()) { PyErr_Print(); initialized_ = false; }
    PyGILState_Release(gil);
    canonical_startup_event(("musa_" + role + "_initialize_done").c_str());
    return initialized_;
  }
  bool Run() {
    if (!initialized_) return false;
    auto gil = PyGILState_Ensure();
    PyObject* bytes = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(input_.data()), input_.size()*sizeof(float));
    PyObject* result = bytes ? PyObject_CallMethod(session_, "run", "O", bytes) : nullptr;
    Py_XDECREF(bytes);
    bool ok = result && PyBytes_Check(result) &&
      PyBytes_Size(result) == static_cast<Py_ssize_t>(output_.size()*sizeof(float));
    if (ok) {
      std::memcpy(output_.data(), PyBytes_AsString(result), output_.size()*sizeof(float));
      for (float value : output_) if (!std::isfinite(value)) ok = false;
    }
    if (PyErr_Occurred()) PyErr_Print();
    Py_XDECREF(result); PyGILState_Release(gil);
    return ok;
  }
  void Destroy() {
    if (session_) { auto gil = PyGILState_Ensure(); Py_DECREF(session_); session_=nullptr; PyGILState_Release(gil); }
    initialized_ = false;
  }
  bool IsInitialized() const { return initialized_; }
  bool CaptureGraph() { return initialized_; } // Session prepares the inference graph at initialization.
  size_t GetInputDimension() const { return input_.size(); }
  std::vector<float>& GetInputBuffer() { return input_; }
 protected:
  std::vector<float> input_, output_;
 private:
  PyObject* session_ = nullptr;
  bool initialized_ = false;
};
class EncoderEngine : public MusaInference {
 public:
  bool Initialize(const std::string& path, bool fp16=false) { return MusaInference::Initialize(path,"encoded_tokens",64,fp16); }
  bool Encode() { return Run(); }
  size_t GetTokenDimension() const { return output_.size(); }
  std::vector<float>& GetTokenBuffer() { return output_; }
};
class PolicyEngine : public MusaInference {
 public:
  bool Initialize(const std::string& path, bool fp16=false) { return MusaInference::Initialize(path,"action",29,fp16); }
  bool Infer() { return Run(); }
  size_t GetActionDimension() const { return output_.size(); }
  std::vector<float>& GetActionBuffer() { return output_; }
};
