
在TVM的教程里，有一个自定义OP的例子：
https://docs.tvm.ai/dev/relay_add_op.html

```c++
RELAY_REGISTER_OP("add")
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side tensor.")
    .add_argument("rhs", "Tensor", "The right hand side tensor.")
    .set_support_level(1)
    .add_type_rel("Broadcast", BroadcastRel);
```

RELAY_REGISTER_OP是一个宏，在使用时创建了一个OpRegistry，并注册到全局的dmlc::Registry对象里。

```c++
class OpRegistry {
 public:
  ......

  TVM_DLL static ::dmlc::Registry<OpRegistry>* Registry();

 private:
  std::string name;
  Op op_;
};


#define RELAY_REGISTER_OP(OpName)                        \
  DMLC_STR_CONCAT(RELAY_REGISTER_VAR_DEF, __COUNTER__) = \
      ::tvm::relay::OpRegistry::Registry()               \
          ->__REGISTER_OR_GET__(OpName)                  \
          .set_name()
```

