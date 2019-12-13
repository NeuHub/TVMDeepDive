
TOPI可以看做是一组上层的API，用户在创建计算图的时候，如果使用tvm api或者relay的算子，要实现复杂的计算就比较麻烦，topi为了简化这些功能，提供了很多
定义好的功能。

关于TOPI的文档不多，我们看看在TVM里TOPI的代码是怎样起作用的。

由于还不知道上层的入口如何进入到TOPI，我们先从代码看一下哪里用到了topi。在src里搜索可以看到很多地方用到了，比较典型的一个是：
relay/op/tensor/binary.cc:.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::multiply));

我们看一下具体的实现：
src/relay/op/tensor/binary.cc
```c++
#define RELAY_BINARY_COMPUTE(FTOPI)                        \
  [] (const Attrs& attrs,                                  \
      const Array<Tensor>& inputs,                         \
      const Type& out_type,                                \
      const Target& target) -> Array<Tensor> {             \
    CHECK_EQ(inputs.size(), 2U);                           \
    return {FTOPI(inputs[0], inputs[1])};                  \
  }   

// Addition
RELAY_REGISTER_BINARY_OP("add")
.describe("Elementwise add with with broadcasting")
.set_support_level(1)
.set_attr<FTVMCompute>("FTVMCompute", RELAY_BINARY_COMPUTE(topi::add));
```
可见在relay里，通过宏RELAY_REGISTER_BINARY_OP注册了函数add，其实现为topi::add，注册的key为FTVMCompute。

我们看看RELAY_REGISTER_BINARY_OP这个宏具体做了什么：

src/relay/op/op_common.h
```c++
#define RELAY_REGISTER_BINARY_OP(OpName)                          \
  TVM_REGISTER_API("relay.op._make." OpName)                      \
    .set_body_typed<Expr(Expr, Expr)>([](Expr lhs, Expr rhs) {    \
        static const Op& op = Op::Get(OpName);                    \
        return CallNode::make(op, {lhs, rhs}, Attrs(), {});       \
      });                                                         \
  RELAY_REGISTER_OP(OpName)                                       \
    .set_num_inputs(2)                                            \
    .add_argument("lhs", "Tensor", "The left hand side tensor.")  \
    .add_argument("rhs", "Tensor", "The right hand side tensor.") \
    .add_type_rel("Broadcast", BroadcastRel)                      \
    .set_attr<TOpPattern>("TOpPattern", kBroadcast)               \
    .set_attr<TOpIsStateful>("TOpIsStateful", false)              \
    .set_attr<FInferCorrectLayout>("FInferCorrectLayout",         \
                                   BinaryBroadcastLayout)
```

include/relay/op.h
```c++
#define RELAY_REGISTER_OP(OpName)                        \
  DMLC_STR_CONCAT(RELAY_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::relay::OpRegistry::Registry()               \
          ->__REGISTER_OR_GET__(OpName)                  \
          .set_name()
```

可见宏RELAY_REGISTER_BINARY_OP做了两件事：
1. 注册一个TVM的全局API，relay.op._make.add，其实现是根据参数OpName取得对应的Op，并返回一个CallNode。
2. 向relay注册一个Op，并且设置Name, argument，type_relation，以及一些属性。关于这些属性我们暂时不去关注
3. 在relay里对应该Op的EntryType里设置属性FTVMCompute为topi::add函数。

既然在这里设置了属性，在生成代码的时候肯定要获取该属性，很容易查到
relay/backend/compile_engine.cc:        Op::GetAttr<FTVMCompute>("FTVMCompute");

该调用发生在函数VisitExpr_（）里。
Array<Tensor> VisitExpr_(const CallNode* call_node) final {
  
======== 待续===========
