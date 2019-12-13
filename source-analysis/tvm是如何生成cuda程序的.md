
TVM使用relay作为IR表示，在生成程序的时候，调用的是tvm.relay.build函数，该函数调用最终被C++侧的RelayBuildModule处理：

src/relay/backend/build_module.cc
```c++
class RelayBuildModule : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name,
                         const std::shared_ptr<ModuleNode>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetGraphJSON();
      });
    } else if (name == "get_module") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->GetModule();
      });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        CHECK_EQ(args.num_args, 3);
        this->Build(args[0], args[1], args[2]);
      });
    } else if (name == "list_params") {
    ......
```
Build函数调用了BuildRelay函数，在BuildRelay里，实现了Module的优化，然后做代码生成，最后调用tvm::build进行程序生成。

src/relay/backend/build_module.cc
```c++
void BuildRelay(
      Function func,
      const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
    if (params.size()) {
      func = BindParamsByName(func, params);
    }

    // Perform Module->Module optimizations.
    relay::Module relay_module = relay::ModuleNode::FromExpr(func);
    relay_module = Optimize(relay_module, targets_, params);
    CHECK(relay_module.defined());
    // Get the updated function.
    func = relay_module->Lookup("main");

    // Generate code for the updated function.
    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    graph_codegen_->Init(nullptr, targets_);
    graph_codegen_->Codegen(func);

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    auto lowered_funcs = graph_codegen_->GetLoweredFunc();
    if (lowered_funcs.size() != 0) {
      ret_.mod = tvm::build(
        lowered_funcs,
        target_host_,
        BuildConfig::Current());
    }
  }
```

在tvm::build函数里，根据不同的target, 分别调用DeviceBuild来生成程序
 
```c++
/ Build for heterogeneous execution.
runtime::Module build(const Map<Target, Array<LoweredFunc>>& inputs,
                      const Target& target_host,
                      const BuildConfig& config) {
  ......

  for (const auto& it : inputs) {
    auto host_dev_funcs =
        split_dev_host_funcs(it.second, it.first, target_host_val, config);
    auto& fhost = host_dev_funcs[0];
    auto& fdevice = host_dev_funcs[1];
    // Get the module for a certain target.
    runtime::Module mdev = DeviceBuild(fdevice, it.first);
    for (const auto& it : fhost) {
      fhost_all.push_back(it);
    }
    device_modules.push_back(mdev);
  }

  runtime::Module mhost = codegen::Build(fhost_all, target_host_val->str());
  ......
}
```
 
在DeviceBuild函数里，调用了codegen::Build来生成程序，在这个函数里，根据target查询对应的codegen.build_xxx函数，然后进行调用。对于cuda来说，调用的就是codegen.build_cuda()函数

src/codegen/codegen.cc
```c++
runtime::Module Build(const Array<LoweredFunc>& funcs,
                      const std::string& target) {
  std::string mode = target;
  size_t pos = mode.find(' ');
  if (pos != std::string::npos) {
    mode = mode.substr(0, pos);
  }
  std::string build_f_name = "codegen.build_" + mode;
  // the build function.
  const PackedFunc* bf = runtime::Registry::Get(build_f_name);
  CHECK(bf != nullptr)
      << "Target " << target << " is not enabled";
  runtime::Module m = (*bf)(funcs, target);
  return m;
}
```
codegen.build_cuda()函数是一个被注册的函数名，其实现是codegen里的buildCUDA函数，在其中又调用了注册的tvm_callback_cuda_compile函数

src/codegen/opt/build_cuda_on.cc
```c++
runtime::Module BuildCUDA(Array<LoweredFunc> funcs) {
  using tvm::runtime::Registry;l
  bool output_ssa = false;
  CodeGenCUDA cg;
  cg.Init(output_ssa);

  for (LoweredFunc f : funcs) {
    cg.AddFunction(f);
  }
  std::string code = cg.Finish();

  if (const auto* f = Registry::Get("tvm_callback_cuda_postproc")) {
    code = (*f)(code).operator std::string();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto* f = Registry::Get("tvm_callback_cuda_compile")) {
    ptx = (*f)(code).operator std::string();
    // Dirty matching to check PTX vs cubin.
    // TODO(tqchen) more reliable checks
    if (ptx[0] != '/') fmt = "cubin";
  } else {
    ptx = NVRTCCompile(code, cg.need_include_path());
  }
  return CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(funcs), code);
}

TVM_REGISTER_API("codegen.build_cuda")
.set_body_typed(BuildCUDA);
```

tvm_callback_cuda_compile函数是从python里注册的。

python/tvm/autotvm/measure/measure_methods.py
```python
@register_func
def tvm_callback_cuda_compile(code):
    """use nvcc to generate ptx code for better optimization"""
    ptx = nvcc.compile_cuda(code, target="ptx", arch=AutotvmGlobalScope.current.cuda_target_arch)
    return ptx
```

而在nvcc.compile_cuda函数里，使用了命令行调用nvcc来生成程序
```python
def compile_cuda(code,
                 target="ptx",
                 arch=None,
                 options=None,
                 path_target=None):
    """Compile cuda code with NVCC from env.

    Parameters
    ----------
    code : str
        The cuda code.

    target : str
        The target format

    arch : str
        The architecture

    options : str or list of str
        The additional options

    path_target : str, optional
        Output file.

    Return
    ------
    cubin : bytearray
        The bytearray of the cubin
    """
    temp = util.tempdir()
    if target not in ["cubin", "ptx", "fatbin"]:
        raise ValueError("target must be in cubin, ptx, fatbin")
    temp_code = temp.relpath("my_kernel.cu")
    temp_target = temp.relpath("my_kernel.%s" % target)

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    if arch is None:
        if nd.gpu(0).exist:
            # auto detect the compute arch argument
            arch = "sm_" + "".join(nd.gpu(0).compute_version.split('.'))
        else:
            raise ValueError("arch(sm_xy) is not passed, and we cannot detect it from env")

    file_target = path_target if path_target else temp_target
    cmd = ["nvcc"]
    cmd += ["--%s" % target, "-O3"]
    cmd += ["-arch", arch]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg = "Compilation error:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    data = bytearray(open(file_target, "rb").read())
    if not data:
        raise RuntimeError(
            "Compilation error: empty result is generated")
    return data
```
