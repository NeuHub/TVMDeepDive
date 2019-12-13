
我们需要尝试一下TVM对PyTorch模型的支持，但是TVM还不支持直接加载PyTorch的模型，在TVM的教程里有个加载ONNX模型的示例，根据这个示例我们先将PyTorch
的模型转为ONNX格式，然后再用TVM来加载执行。


这里是代码部分：
```python
import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata

onnx_model = onnx.load("resnet50/model.onnx")

from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], #这是imagenet數據集的均值
    std=[0.229, 0.224, 0.225]
)
 
tran=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

im = Image.open("cat.jpeg")
x = tran(im)
x.unsqueeze_(dim=0)

target = 'cuda'
target_host = 'llvm'
layout = "NCHW"
ctx = tvm.gpu(0)

#target = 'llvm'
#layout = "NCHW"
#ctx = tvm.cpu()
print(ctx.device_type)

#input_name = 'data'
input_name='gpu_0/data_0'
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=1):
    intrp = relay.build_module.create_executor('graph', mod, tvm.gpu(0), target)

dtype = 'float32'
x_numpy = x.numpy()
t = tvm.nd.array(x_numpy.astype(dtype))
%time it = intrp.evaluate()
    
%time tvm_output = it(t, **params).asnumpy()
```

为了进一步了解过程，对执行过程做了Pdb和gdb的跟踪，在这里记录一下，后续再详细分析

===============================================================
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
with relay.build_config(opt_level=1):
    intrp = relay.build_module.create_executor('graph', mod, tvm.gpu(0), target)
	return GraphExecutor(mod, ctx, target)
	GraphExecutor(_interpreter.Executor)



it = intrp.evaluate()  /home/heyunlong3/notespace/from_onnx.py:83
	GraphExecutor._make_executor()  /home/heyunlong3/tvm/tvm/python/tvm/relay/backend/interpreter.py(215)
		graph_json, mod, params = build(self.mod, target=self.target)  /home/heyunlong3/tvm/tvm/python/tvm/relay/build_module.py(240)
    			bld_mod = BuildModule()
    			graph_json, mod, params = bld_mod.build(func, target, target_host, params)
				self._build(func, target, target_host)
				[C++]RelayBuildModule：：GetFunction（“build”）
					this->Build(args[0], args[1], args[2]);
						BuildRelay(func, params_);
							relay::Module relay_module = relay::ModuleNode::FromExpr(func);
							relay_module = Optimize(relay_module, targets_, params);
							// Generate code for the updated function.
							graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
							graph_codegen_->Init(nullptr, targets_);
								GraphRuntimeCodegenModule::init()
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



		gmodule = _graph_rt.create(graph_json, mod, self.ctx)  /home/heyunlong3/tvm/tvm-debug/python/tvm/contrib/graph_runtime.py(25)
			fcreate = get_global_func("tvm.graph_runtime.create")
			return GraphModule(fcreate(graph_json_str, libmod, *device_type_id))
				[C++] GraphRuntimeCreate()  src/runtime/graph/graph_runtime.cc(509)  /home/heyunlong3/tvm/tvm-debug/src/runtime/graph/graph_runtime.cc(482)
					GraphRuntime.init()    /home/heyunlong3/tvm/tvm-debug/src/runtime/graph/graph_runtime.cc(71)
						this->Load(&reader);  
						this->SetupOpExecs();  
    							std::tie(op_execs_[nid], op_args) = CreateTVMOp(inode.param, args, inode.inputs.size());   /home/heyunlong3/tvm/tvm-debug/src/runtime/graph/graph_runtime.cc(349)
 
  								tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);   /home/heyunlong3/tvm/tvm-debug/src/runtime/graph/graph_runtime.cc(403) // Get compiled function from the module that contains both host and device  

  								auto fexec = [arg_ptr, pf]() {…


		if params:
    			gmodule.set_input(**params)


tvm_output = it(t, **params).asnumpy()
	_graph_wrapper(*args, **kwargs):
		for i, arg in enumerate(args):
    			gmodule.set_input(i, arg)
		# Run the module, and fetch the output.
		gmodule.run()
			GraphRuntime::Run()

