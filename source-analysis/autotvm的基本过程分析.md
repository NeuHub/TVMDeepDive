
使用atuotvm的时候，Tuner部分典型代码如所示，我们将要分析一下tuner.tune的过程。

```python
# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
task = autotvm.task.create(conv2d_no_batching,
                           args=(N, H, W, CO, CI, KH, KW, strides, padding),
                           target='cuda')
print(task.config_space)

# Use local gpu, measure 10 times for every config to reduce variance
# The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=20,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('conv2d.log')])
```
XGBTuner继承于ModelBasedTuner, ModelBasedTuner又继承于Tuner，其实现的tune函数
```python
        measure_batch = create_measure_batch(self.task, measure_option)
            
            configs = self.next_batch(min(n_parallel, n_trial - i))
            
            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            ...

            self.update(inputs, results)
            
            for callback in callbacks:
                callback(self, inputs, results)
```
基本的工作流程就是：
1. 切分config
2. 根据当前的config、task和target，生成相应的代码，然后在设备上运行
3. 更新结果

Tuner并没有实现update函数，这里用的是ModelBasedTuner的update函数，
python/tvm/autotvm/tuner/model_based_tuner.py
```python
    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0.0)

        # if we have enough new training samples
        if len(self.xs) >= self.plan_size * (self.train_ct + 1) \
                and self.flops_max > 1e-6:
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            if self.diversity_filter_ratio:
                candidate = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size * self.diversity_filter_ratio, self.visited)
                scores = self.cost_model.predict(candidate)
                knobs = [point2knob(x, self.dims) for x in candidate]
                pick_index = submodular_pick(0 * scores, knobs, self.plan_size, knob_weight=1)
                maximums = np.array(candidate)[pick_index]
            else:
                maximums = self.model_optimizer.find_maximums(
                    self.cost_model, self.plan_size, self.visited)

            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1
```
这里可以看到，训练时使用的feature就是inp.config.index，训练的代码是这一行：
self.cost_model.fit(self.xs, self.ys, self.plan_size)

回过头来看一下, XGBTunner里的cost model是XGBoostCostModel

python/tvm/autotvm/tuner/xgboost_tuner.py
```python
class XGBTuner(ModelBasedTuner):
    def __init__(self, task, plan_size=64,
                 feature_type='itervar', loss_type='rank', num_threads=None,
                 optimizer='sa', diversity_filter_ratio=None, log_interval=50):
        cost_model = XGBoostCostModel(task,
                                      feature_type=feature_type,
                                      loss_type=loss_type,
                                      num_threads=num_threads,
                                      log_interval=log_interval // 2)
        if optimizer == 'sa':
            optimizer = SimulatedAnnealingOptimizer(task, log_interval=log_interval)
        else:
            assert isinstance(optimizer, ModelOptimizer), "Optimizer must be " \
                                                          "a supported name string" \
                                                          "or a ModelOptimizer object."

        super(XGBTuner, self).__init__(task, cost_model, optimizer,
                                       plan_size, diversity_filter_ratio)
```

在XGBoostCostModel里，fit函数根据index获取实际的feature。
python/tvm/autotvm/tuner/xgboost_cost_model.py
```python
    def fit(self, xs, ys, plan_size):
        tic = time.time()
        self._reset_pool(self.space, self.target, self.task)

        x_train = self._get_feature(xs)
        
    def _get_feature(self, indexes):
        # free feature cache
        if self.feature_cache.size(self.fea_type) >= 100000:
            self.feature_cache.clear(self.fea_type)

        fea_cache = self.feature_cache.get(self.fea_type)

        indexes = np.array(indexes)
        need_extract = [x for x in indexes if x not in fea_cache]

        if need_extract:
            pool = self._get_pool()
            feas = pool.map(self.feature_extract_func, need_extract)
            for i, fea in zip(need_extract, feas):
                fea_cache[i] = fea

        feature_len = None
        for idx in indexes:
            if fea_cache[idx] is not None:
                feature_len = fea_cache[idx].shape[-1]
                break

        ret = np.empty((len(indexes), feature_len), dtype=np.float32)
        for i, ii in enumerate(indexes):
            t = fea_cache[ii]
            ret[i, :] = t if t is not None else 0
        return ret       
```

在_get_feature()的实现里，feature是从fea_cache里获取的，但是如果在fea_cache里没有找到，就需要调用_get_pool()来获取feature，pool事实上是对应的启动的运行任务，也就是需要从这些任务里调用feature_extract_func（）函数来获取feature。
python/tvm/autotvm/tuner/xgboost_cost_model.py
```python
   def __init__(self, task, feature_type, loss_type, num_threads=None, log_interval=25,
                 upper_model=None):
        ......
 
       if feature_type == 'itervar':
            self.feature_extract_func = _extract_itervar_feature_index
        elif feature_type == 'knob':
            self.feature_extract_func = _extract_knob_feature_index
        elif feature_type == 'curve':
            self.feature_extract_func = _extract_curve_feature_index
        else:
            raise RuntimeError("Invalid feature type " + feature_type)
            
def _extract_itervar_feature_index(index):
    """extract iteration var feature for an index in extract_space"""
    try:
        config = _extract_space.get(index)
        with _extract_target:
            sch, args = _extract_task.instantiate(config)
        fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)
        fea = np.concatenate((fea, list(config.get_other_option().values())))
        return fea
    except Exception:  # pylint: disable=broad-except
        return None
```
由于XGBoostTuner初始化的时候的feature_type='itervar',所以这里使用的函数是_extract_itervar_feature_index。
在fit()被调用的时候，调用了函数_reset_pool（）来设置task，这个task就是XGBoostCostModel初始化时，XGBoostTuner传过来的task。在这个Case里，也就是最早autotvm.task.create()所创建的task.
```python
    def _reset_pool(self, space, target, task):
        """reset processing pool for feature extraction"""

        if self.upper_model:  # base model will reuse upper model's pool,
            self.upper_model._reset_pool(space, target, task)
            return

        self._close_pool()

        # use global variable to pass common arguments
        global _extract_space, _extract_target, _extract_task
        _extract_space = space
        _extract_target = target
        _extract_task = task
        self.pool = multiprocessing.Pool(self.num_threads)
```

在autotvm.task.Task的instantiate（）函数里，根据config创建了schedule和arg_buf，
```python
   def instantiate(self, config):
        config.flop = 0
        with ApplyConfig(config):
            sch, arg_bufs = self.func(*self.args, **self.kwargs)
        if not self.flop:
            config.flop = config.flop or compute_flop(sch)
            self.flop = config.flop
        return sch, arg_bufs
```
        
接下来我们看fea = feature.get_itervar_feature_flatten(sch, args, take_log=True)。
python/autotvm/feature.py
```python
_get_itervar_feature_flatten = get_global_func("autotvm.feature.GetItervarFeatureFlatten")

def get_itervar_feature_flatten(sch, args, take_log=True):
    stmt = ana_lower(sch, args, simple_mode=True)
    feas = _get_itervar_feature_flatten(stmt, take_log)
    feas = struct.unpack('%df' % (len(feas)//4), feas)
    return feas
```
这里调用了C++的函数
src/autotvm/touch_extracter.cc
```c++
TVM_REGISTER_API("autotvm.feature.GetItervarFeatureFlatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  Stmt stmt = args[0];
  bool take_log = args[1];
  std::vector<float> ret_feature;

  GetItervarFeatureFlatten(stmt, take_log, &ret_feature);

  TVMByteArray arr;
  arr.size = sizeof(float) * ret_feature.size();
  arr.data = reinterpret_cast<char *>(ret_feature.data());
  *ret = arr;
});

void GetItervarFeatureFlatten(Stmt stmt, bool take_log, std::vector<float> *ret_feature) {
  // extract touch feature
  TouchExtractor touch_analyzer;
  touch_analyzer.Analyze(stmt);

  // sort according to order
  std::vector<VarExpr> vars;
  for (auto kv : touch_analyzer.itervar_map) {
    vars.push_back(kv.first);
  }
  std::sort(vars.begin(), vars.end(), [&](const VarExpr &lhs, const VarExpr &rhs) -> bool {
    return touch_analyzer.itervar_map[lhs].order < touch_analyzer.itervar_map[rhs].order;
  });

  // whether take log for numerical feature
  std::function<float(int64_t)> trans;
  if (take_log) {
    trans = [](int64_t x) {
      if (x < 0)
        return -std::log(-x+1) / std::log(2);
      x = x + 1;
      return std::log(x) / std::log(2);
    };
  } else {
    trans = [](int64_t x) {
      return x;
    };
  }

  // serialize for front end
  for (auto var : vars) {
    ItervarFeature &fea = touch_analyzer.itervar_map[var];

    ret_feature->push_back(trans(fea.length));
    ret_feature->push_back(fea.nest_level);
    ret_feature->push_back(trans(fea.topdown_product));
    ret_feature->push_back(trans(fea.bottomup_product));

    // one hot annotation
    for (int i = 0; i < kNum; i++) {
      ret_feature->push_back(i == fea.ann);
    }

    // arithmetic
    ret_feature->push_back(trans(fea.add_ct));
    ret_feature->push_back(trans(fea.mul_ct));
    ret_feature->push_back(trans(fea.div_ct));

    // touch map
    std::vector<TouchedBuffer> bufs;
    for (auto kv : fea.touch_feature) {
      bufs.push_back(kv.first);
    }
    std::sort(bufs.begin(), bufs.end());
    for (auto k : bufs) {
      TouchPattern &v = fea.touch_feature[k];
      ret_feature->push_back(trans(v.stride));
      ret_feature->push_back(trans(v.mod));
      ret_feature->push_back(trans(v.count));
      ret_feature->push_back(trans(v.reuse));
      ret_feature->push_back(trans(v.thread_count));
      ret_feature->push_back(trans(v.thread_reuse));
    }
  }
}

```
这里大概能看出使用了哪些用于调优的特征。

特征收集的关键步骤是调用TouchExtractor::Analyze()函数。
src/autotvm/touch_extracter.h
```c++
class TouchExtractor : public FeatureVisitor {
 public:
  void Analyze(Stmt stmt) {
    this->Visit(stmt);
  }
```

include/tvm/ir_visitor.h
```c++
class TVM_DLL IRVisitor {
 public:
  /*!
   * \brief recursively visit an IR node
   */
  virtual void Visit(const NodeRef& node) {
    static const FVisit& f = vtable();
    if (node.defined()) f(node, this);
  }
  
  using FVisit = NodeFunctor<void(const ObjectRef&, IRVisitor*)>;
  static FVisit& vtable();
  // overloadable visit function.
  virtual void Visit_(const Variable* op);
  virtual void Visit_(const LetStmt* op);
  virtual void Visit_(const AttrStmt* op);
  virtual void Visit_(const IfThenElse* op);
  virtual void Visit_(const For* op);
  virtual void Visit_(const Allocate* op);
  virtual void Visit_(const Load* op);
```

src/autotvm/pass/ir_visitor.cc
```c++
IRVisitor::FVisit& IRVisitor::vtable() {  // NOLINT(*)
  static FVisit inst; return inst;
}

#define DISPATCH_TO_VISIT(OP)                                \
  set_dispatch<OP>([](const ObjectRef& node, IRVisitor* v) { \
      v->Visit_(static_cast<const OP*>(node.get()));         \
    })
```
IRVisitor里定义了针对不同类型节点的Visit_()函数，并且通过宏DISPATCH_TO_VISIT将不同类型Node的Visit_()函数加到func_数组里。IRVisitor的子类可以通过重载必要的Visit_()函数来实现特殊的处理。

NodeFunction的实现在这里：
include/tvm/node/functor.h
```c++
template<typename R, typename ...Args>
class NodeFunctor<R(const ObjectRef& n, Args...)> {
 private:
  /*! \brief internal function pointer type */
  typedef R (*FPointer)(const ObjectRef&n, Args...);
  /*! \brief refer to itself. */
  using TSelf = NodeFunctor<R (const ObjectRef& n, Args...)>;
  /*! \brief internal function table */
  std::vector<FPointer> func_;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*!
   * \brief Whether the functor can dispatch the corresponding Node
   * \param n The node to be dispatched
   * \return Whether dispatching function is registered for n's type.
   */
  bool can_dispatch(const ObjectRef& n) const {
    uint32_t type_index = n->type_index();
    return type_index < func_.size() && func_[type_index] != nullptr;
  }
  /*!
   * \brief invoke the functor, dispatch on type of n
   * \param n The Node argument
   * \param args The additional arguments
   * \return The result.
   */
  R operator()(const ObjectRef& n, Args... args) const {
    CHECK(can_dispatch(n))
        << "NodeFunctor calls un-registered function on type "
        << n->GetTypeKey();
    return (*func_[n->type_index()])(n, std::forward<Args>(args)...);
  }
  /*!
   * \brief set the dispacher for type TNode
   * \param f The function to be set.
   * \tparam TNode the type of Node to be dispatched.
   * \return reference to self.
   */
  template<typename TNode>
  TSelf& set_dispatch(FPointer f) {  // NOLINT(*)
    uint32_t tindex = TNode::RuntimeTypeIndex();
    if (func_.size() <= tindex) {
      func_.resize(tindex + 1, nullptr);
    }
    CHECK(func_[tindex] == nullptr)
        << "Dispatch for " << TNode::_type_key
        << " is already set";
    func_[tindex] = f;
    return *this;
  }
···
在调用NodeFunction函数时，会在func_的数组里根据node的typeindex（typeindex是tvm运行时动态生成的）选择合适的Visit_函数进行调用。同时NodeFunction提供了set_dispatch()工具方法，支持向func_数组内注册不同的函数。

那么我们再来看看特征是如何通过Visit()函数来实现收集的。在TouchExtractor这个类的实现里，重载对很多Node的处理函数Visit_()。

src/autotvm/touch_extracter.h
```c++
class TouchExtractor : public FeatureVisitor {
 public:
  void Analyze(Stmt stmt) {
    this->Visit(stmt);
  }
  
  // arithmetic stats
  void Visit_(const Add *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].add_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Sub *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].add_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Mul *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].mul_ct++;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Div *op) {
    if (op->type.is_float())
      itervar_map[itervar_stack_.back()].div_ct++;
    IRVisitor::Visit_(op);
  }  
  
    std::unordered_map<VarExpr, ItervarFeature, tvm::ExprHash, tvm::ExprEqual> itervar_map;
```
从这里可以看到，TouchExtractor统计了op的数量，并且保存在itervar_map里。
feature的类型是ItervarFeature。
src/autotvm/touch_extractor.h
```c++
// all the feature of an iter var
struct ItervarFeature {
  ItervarFeature(VarExpr var,
                 int64_t extent,
                 int nest,
                 AnnotationType ann_type,
                 int64_t topdown,
                 int counter)
      : length(extent), nest_level(nest), ann(ann_type), topdown_product(topdown), order(counter) {}
  ItervarFeature() {}

  // Axis Attributes
  int64_t length;
  int nest_level;
  AnnotationType ann;         // one-hot axis type
  int64_t topdown_product;    // accumulative product of axis length, in top-down order
  int64_t bottomup_product;   // accumulative product of axis length, in bottom-up order
  // bottomup_product = reuse * count for any touched buffer

  int order;  // used for soring axis

  // Arithmetic feature
  int add_ct{0};
  int mul_ct{0};
  int div_ct{0};

  // Memory Touch Feature
  std::unordered_map<TouchedBuffer, TouchPattern> touch_feature;
};
```

知道了feature的生成过程，我们再来看看优化任务是怎么运行的。
measure_batch 函数的实现在这里：
python/tvm/autotvm/measure/measure.py
```
    def measure_batch(measure_inputs):
        build_results = builder.build(measure_inputs)
        results = runner.run(measure_inputs, build_results)
        return results

    measure_batch.n_parallel = builder.n_parallel
    measure_batch.attach_objects = attach_objects
    return measure_batch
```
可见主要步骤是先调用builder生成代码，然后调用了runner的run函数。其参数meature_inputs是由target、task和config组成的。
从前面代码可以知道builder是autotvm.LocalBuilder()的实例。
python/tvm/autotvm/measure/measure_methods.py
```python
class LocalBuilder(Builder):
    def __init__(self, timeout=10, n_parallel=None, build_func='default'):
        super(LocalBuilder, self).__init__(timeout, n_parallel)

        if isinstance(build_func, str):
            if build_func == 'default':
                build_func = tar.tar
            elif build_func == 'ndk':
                build_func = ndk.create_shared
            else:
                raise ValueError("Invalid build_func" + build_func)
        self.build_func = _wrap_build_func(build_func)
        self.executor = LocalExecutor(timeout=timeout)
        self.tmp_dir = tempfile.mkdtemp()

    def build(self, measure_inputs):
        results = []

        shutil.rmtree(self.tmp_dir)
        self.tmp_dir = tempfile.mkdtemp()

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for inp in measure_inputs[i:i + self.n_parallel]:
                ret = self.executor.submit(self.build_func,
                                           inp,
                                           self.tmp_dir,
                                           **self.build_kwargs)
                futures.append(ret)
            。。。。。。

        return results
```
由于创建LocalBuilder时没有指定参数，参数build_func的值是"default"，submit的就是_wrap_build_func(tar.tar)。
至于LocalExecuter后面也会用到，会再详细看一下。我们先看一下_wrap_build_func

python/tvm/autotvm/measure/measure_methods.py
```python
def _build_func_common(measure_input, check_gpu=None, cuda_arch=None, build_option=None):
    """Common part for building a configuration"""
    target, task, config = measure_input
    with target:
        s, args = task.instantiate(config)

        # check invalidity of template and code hash consistency
        if not config.valid():
            raise InstantiationError(config.errors)

        opts = build_option or {}
        if check_gpu:  # Add verify pass to filter out invalid configs in advance.
            opts["add_lower_pass"] = [(2, gpu_verify_pass(**check_gpu))]
        if cuda_arch:
            set_cuda_target_arch(cuda_arch)

        # if target is vta, we need to use vta build
        if hasattr(measure_input.target, 'device_name') and \
            measure_input.target.device_name == 'vta':
            import vta
            func = vta.build(s, args, target_host=task.target_host)
        else:
            with build_config(**opts):
                func = build(s, args, target_host=task.target_host)
    return func, tuple((get_const_tuple(x.shape), x.dtype) for x in args)
    
def _wrap_build_func(build_func):
    if not hasattr(build_func, "output_format"):
        raise AttributeError("Expect build_func to have the attribute output_format.")
    output_format = build_func.output_format

    def _wrapped(measure_input, tmp_dir, **kwargs):
        tic = time.time()
        try:
            filename = os.path.join(tmp_dir, "tmp_func_%0x.%s" % (
                getrandbits(64), output_format))
            # TODO(tvm-team) consider linline _build_func_common
            func, arg_info = _build_func_common(measure_input, **kwargs)
            func.export_library(filename, build_func)
        except Exception as e:  # pylint: disable=broad-except
            return BuildResult(None, None, e, time.time() - tic)
        return BuildResult(filename, arg_info, None, time.time() - tic)
    return _wrapped
```
这一句：
func = build(s, args, target_host=task.target_host)
调用了tvm.build()函数，用来生成实际的代码。
后面需要分析一下tunner部分是如何根据参数进行调整的。

再看一下Runner部分。
LocalRunner继承于RPCRunner，其中run函数实现如下：
python/tvm/autotvm/measure/measure_methods.py
```python
    def run(self, measure_inputs, build_results):
        results = []
        remote_args = (self.key, self.host, self.port, self.priority, self.timeout)

        for i in range(0, len(measure_inputs), self.n_parallel):
            futures = []
            for measure_inp, build_res in zip(measure_inputs[i:i+self.n_parallel],
                                              build_results[i:i+self.n_parallel]):
                ret = self.executor.submit(run_through_rpc,
                                           measure_inp,
                                           build_res,
                                           self.number,
                                           self.repeat,
                                           self.min_repeat_ms,
                                           self.cooldown_interval,
                                           remote_args,
                                           self.ref_input,
                                           self.ref_output)
                futures.append(ret)

            for future in futures:
                res = future.get()
                if isinstance(res, Exception):   # executor error or timeout
                    results.append(MeasureResult((str(res),), MeasureErrorNo.RUN_TIMEOUT,
                                                 self.timeout, time.time()))
                else:
                    results.append(res)

        return results
 ```
 
 ```python
 def _execute_func(func, queue, args, kwargs):
    """execute function and return the result or exception to a queue"""
    try:
        res = func(*args, **kwargs)
    except Exception as exc:  # pylint: disable=broad-except
        res = exc
    queue.put(res)


def call_with_timeout(queue, timeout, func, args, kwargs):
    """A wrapper to support timeout of a function call"""

    # start a new process for timeout (cannot use thread because we have c function)
    p = Process(target=_execute_func, args=(func, queue, args, kwargs))
    p.start()
    p.join(timeout=timeout)

    queue.put(executor.TimeoutError())

    kill_child_processes(p.pid)
    p.terminate()
    p.join()
    
 ass LocalExecutor(executor.Executor):
    """Local executor that runs workers on the same machine with multiprocessing.

    Parameters
    ----------
    timeout: float, optional
        timeout of a job. If time is out. A TimeoutError will be returned (not raised)
    do_fork: bool, optional
        For some runtime systems that do not support fork after initialization
        (e.g. cuda runtime, cudnn). Set this to False if you have used these runtime
        before submitting jobs.
    """
    def __init__(self, timeout=None, do_fork=True):
        self.timeout = timeout or executor.Executor.DEFAULT_TIMEOUT
        self.do_fork = do_fork

        if self.do_fork:
            if not psutil:
                raise RuntimeError("Python package psutil is missing. "
                                   "please try `pip install psutil`")

    def submit(self, func, *args, **kwargs):
        if not self.do_fork:
            return LocalFutureNoFork(func(*args, **kwargs))

        queue = Queue(2)
        process = Process(target=call_with_timeout,
                          args=(queue, self.timeout, func, args, kwargs))
        process.start()
        return LocalFuture(process, queue)
```

因此，实际被调用的函数在这里：
python/tvm/autotvm/measure/measure_methods.py
```python
def run_through_rpc(measure_input, build_result,
                    number, repeat, min_repeat_ms, cooldown_interval,
                    remote_args, ref_input=None, ref_output=None):
    """Run a generated library through rpc

    Parameters
    ----------
    measure_input: MeasureInput
        The raw measure input
    build_result: BuildResult
        The result returned from Builder. This contains the path to the generated library.
    number: int
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int, optional
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first one is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms: int, optional
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval: float
        The cool down interval between two measurements
    remote_args: Tuple
        The argument for request_remote
    ref_input: List of np.ndarray
        The reference input used for checking correctness
    ref_output: List of np.ndarray
        The reference output used for checking correctness
    """
    if isinstance(build_result, MeasureResult):
        return build_result

    tic = time.time()
    errno = MeasureErrorNo.NO_ERROR
    try:
        # upload built module
        remote = request_remote(*remote_args)
        # Program the FPGA every single time when targeting VTA
        if hasattr(measure_input.target, 'device_name') and \
            measure_input.target.device_name == 'vta':
            from vta import program_fpga, reconfig_runtime
            program_fpga(remote, None)
            reconfig_runtime(remote)
        remote.upload(build_result.filename)
        func = remote.load_module(os.path.split(build_result.filename)[1])
        ctx = remote.context(str(measure_input.target), 0)
        time_f = func.time_evaluator(
            func.entry_name, ctx, number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)

        # set input
        if ref_input:
            args = [nd.array(x, ctx=ctx) for x in ref_input]
        else:
            # create empty arrays on the remote device and copy them once.
            # This can avoid some memory issues that make the measurement results unreliable.
            args = [nd.empty(x[0], dtype=x[1], ctx=ctx) for x in build_result.arg_info]
            args = [nd.array(x, ctx=ctx) for x in args]
            ctx.sync()

        costs = time_f(*args).results

        # clean up remote files
        remote.remove(build_result.filename)
        remote.remove(os.path.splitext(build_result.filename)[0] + '.so')
        remote.remove('')

        if len(costs) > 2:  # remove largest and smallest value to reduce variance
            costs = list(costs)
            costs.sort()
            costs = tuple(costs[1:-1])

        # check correctness of output
        if ref_output:
            for expected, real in zip(ref_output, args):
                if not np.allclose(expected, real.asnumpy(), rtol=1e-4):
                    logger.warning("Wrong Answer!")
                    errno = MeasureErrorNo.WRONG_ANSWER
    except TVMError as exc:
        msg = str(exc)
        if "Stack trace returned" in msg:
            msg = msg[:msg.index("Stack trace returned")]
        if "CUDA Source" in msg:
            msg = msg[:msg.index("CUDA Source")]
        costs = (RuntimeError(msg[:1024]),)
        errno = MeasureErrorNo.RUNTIME_DEVICE
    tstamp = time.time()
    time.sleep(cooldown_interval)
    return MeasureResult(costs, errno, tstamp - tic + build_result.time_cost, tstamp)
```
