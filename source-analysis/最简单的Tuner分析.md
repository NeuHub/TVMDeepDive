
在TVM官网上有一个简单的自动Tuner的例子,其中关于schedule的部分代码如下：

```python
# Matmul V1: List candidate values
@autotvm.template  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg['tile_y'].val)
    xo, xi = s[C].split(x, cfg['tile_x'].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]
``` 

这里比较重要的概念是scheduler、config、knob，下面逐一从代码层面分析。

create_schedule调用了C++实现的函数create_schedule

python/tvm/schedule.py
```python
def create_schedule(ops):
    if not isinstance(ops, (list, _container.Array)):
        ops = [ops]
    return _api_internal._CreateSchedule(ops)
```

src/api/api_lang.cc
```c++
TVM_REGISTER_API("_CreateSchedule")
.set_body_typed(create_schedule);
```

include/tvm/schedule.h
```c++
inline Schedule create_schedule(Array<Operation> ops) {
  return ScheduleNode::make(ops);
}
```

ScheduleNode的继承关系是：ScheduleNode->Node->NodeBase，让人稍感麻烦的是这几个类定义在不同的地方，分别是
include/tvm/schedule.h include/tvm/node/node.h  include/tvm/runtime/node_base.h

ScheduleNode::make(Array<Operation> ops)定义在这里：
    
src/schedule/schedule_lang.cc
```c++
Schedule ScheduleNode::make(Array<Operation> ops) {
  auto n = make_node<ScheduleNode>();
  Schedule sch(n);
  n->outputs = ops;
  auto g = schedule::CreateReadGraph(n->outputs);
  Array<Operation> post_order = schedule::PostDFSOrder(n->outputs, g);
  // output set.
  std::unordered_set<Operation> output_set;
  for (Operation x : ops) {
    output_set.insert(x);
  }
  for (Operation op : post_order) {
    Stage stage(op);
    stage->is_output = output_set.count(op) != 0;
    n->stages.push_back(stage);
    n->stage_map.Set(op, stage);
    // mark scan updates.
    if (const ScanOpNode* scan = op.as<ScanOpNode>()) {
      Array<Tensor> inputs;
      for (Tensor t : scan->state_placeholder) {
        inputs.push_back(t);
      }
      for (Tensor t : scan->inputs) {
        inputs.push_back(t);
      }
      // Create the scan group.
      Stage scan_group = sch.create_group(scan->update, inputs, false);
      scan_group->attach_type = kScanUpdate;
      scan_group->attach_stage = stage;

      for (size_t i = 0; i < scan->update.size(); ++i) {
        Stage s = n->stage_map[scan->update[i]->op];
        CHECK(scan_group.same_as(s->group));
      }
    }
  }
  return sch;
}
```
从这段代码可以看出，返回的是一个Schedule对象，继承关系为 Schedule->NodeRef
Schedule控制所有的优化过程，优化包含很多步骤，每个步骤称为一个Stage(也是继承于NodeRef)，Stage也是定义在include/tvm/schedule.h里的，在Stage里实现了很多优化的方法，比如split,tile, fuse, reorder, bind, compute_at, compute_inline,vectorize,tensorize,prefetch等，关于这些优化方法的介绍，最好的方式是参考TVM的文档和源码，后面有机会我也会做一些分析介绍。

一些参考资料：
* https://docs.tvm.ai/tutorials/language/schedule_primitives.html
* https://blog.csdn.net/sayandroid/article/details/88784933#2_stage_50(上一篇的中文版)
* https://docs.tvm.ai/tutorials/language/tensorize.html


下一步是获取config，get_config()定义在这里

python/tvm/autotvm/task/task.py
```python
def get_config():
    """Get current config object
    Returns
    -------
    cfg: ConfigSpace or ConfigEntity
        The current config
    """
    return DispatchContext.current.query(None, None)
```
    def query(self, target, workload):

缺省的DispatchContext是FallbackContext, 在其内部维护了一个hashmap，用来保存ConfigSpace，这个例子里获取的是缺省的ConfigSpace，实现类为FallbackConfigEntity
python/tvm/autotvm/task/dispatcher.py
```python
DispatchContext.current = FallbackContext()
```

接下来是调用了ConfigSpace.define_knob，也就是在Config里增加了一个transform操作。TransformSpace有多个子类，包括VirtualAxis，SplitSpace，ReorderSpace，AnnotateSpace，OtherOptionSpace
```python
    def define_knob(self, name, candidate):
        """Define a tunable knob with a list of candidates

        Parameters
        ----------
        name: str
            name key of that option
        candidate: list
            list of candidates
        """
        return self._add_new_transform(OtherOptionSpace, name, [], None, candidate=candidate)
```
```python
    def _add_new_transform(self, space_class, name, axes, policy, **kwargs):
        """Add a new transform space in template"""
        if self._collect:
            # convert schedule axis to space definition axis
            axes = [x if isinstance(x, (VirtualAxis, Axis)) else self.axis(x) for x in axes]

            # add subspace (knob)
            space = space_class(axes, policy, **kwargs)
            self.space_map[name] = space
            self._entity_map[name] = space[0]
            return [Axis(space, i) for i in range(space.num_output)]
        return [Axis(None, i) for i in range(space_class.get_num_output(axes, policy, **kwargs))]
```

最后我们看一下split是如何工作的

src/schedule/schedule_lang.cc
```c++
Stage& Stage::split(
    IterVar parent, Expr factor, IterVar* p_outer, IterVar* p_inner) {  // NOLINT(*)
  Split(operator->(), parent, factor, Expr(), p_outer, p_inner);
  return *this;
}

void Split(StageNode* self,
           IterVar parent,
           Expr factor,
           Expr nparts,
           IterVar* p_outer,
           IterVar* p_inner) {
  // Check if split is valid.
  CHECK(parent->iter_type == kDataPar ||
        parent->iter_type == kCommReduce ||
        parent->iter_type == kOrdered)
      << "Cannot split on " << IterVarType2String(parent->iter_type);
  IterVar outer = IterVarNode::make(
      Range(), parent->var.copy_with_suffix(".outer"), parent->iter_type);
  IterVar inner = IterVarNode::make(
      Range(), parent->var.copy_with_suffix(".inner"), parent->iter_type);
  *p_outer = outer;
  *p_inner = inner;
  // The splits
  ArrayNode* all_vars = self->all_iter_vars.CopyOnWrite();
  ArrayNode* leaf_vars = self->leaf_iter_vars.CopyOnWrite();
  size_t pos = FindLeafVar(all_vars, leaf_vars, parent);
  self->relations.push_back(SplitNode::make(parent, outer, inner, factor, nparts));
  // add vars to all vars
  all_vars->data.push_back(outer.node_);
  all_vars->data.push_back(inner.node_);
  // replace the position.
  leaf_vars->data.erase(leaf_vars->data.begin() + pos);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, inner.node_);
  leaf_vars->data.insert(leaf_vars->data.begin() + pos, outer.node_);
}
```

这里StageNode有几个关键的成员：relations, all_iter_vars, leaf_iter_vars，具体解释在StageNode的类定义里：

```c++
/*!
 * \brief represents a stage.
 *
 *  relations form a Directed acylic hypergraph in bipartite manner.
 *  With each node is represented by a IterVar,
 *  and each hyper-edge is represented by a IterVarRelation.
 *  The relations connects the IterVars in the graph.
 *
 *  Besides typical stage that corresponds to operations.
 *  There is also group stage, which groups stages together.
 *  Each stage's group(given by group) represent an constraint,
 *  the stage can only be attached to stages within the group.
 *
 *  The group stage node can be attached to IterVars as in normal stage.
 */
class StageNode : public Node {
 public:
 
  /*! \brief All the nodes in the iter var */
  Array<IterVar> all_iter_vars;
  /*! \brief The current active leaf iter vars in the stage. */
  Array<IterVar> leaf_iter_vars;

  Array<IterVarRelation> relations;
  /*! \brief additional attributes about iter var. */
  Map<IterVar, IterVarAttr> iter_var_attrs;
  
  ...
}
···
IterVars代表的是DAG图的节点，IterVarRelation代表的是DAG图中的边。

未完待续
