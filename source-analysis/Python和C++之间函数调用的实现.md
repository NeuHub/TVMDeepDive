
TVM很多核心代码是用C++实现的，但是为了方便工程师使用，提供了python的前端，这样就经常涉及到在python里调用C++的代码，为了扩展方便，在实现时定义了几个宏用来注册函数和查找函数，在include/tvm/runtime/registry.h里，定义了宏TVM_REGISTER_GLOBAL，在include/tvm/api_registry.h里定义了宏TVM_REGISTER_API(OpName)，事实上也是用了宏TVM_REGISTER_GLOBAL.

#define TVM_REGISTER_API(OpName) TVM_REGISTER_GLOBAL(OpName)

```c++
/*!
 * \brief Register a function globally.
 * \code
 *   TVM_REGISTER_GLOBAL("MyPrint")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_GLOBAL(OpName)                              \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) =            \
      ::tvm::runtime::Registry::Register(OpName)
```

通过这个宏定义的函数能够被python等前端语言调用。具体的说明可以看这个类Registry的说明，建议的使用方法是使用时顺便调用set_body方法，
并传递一个lambda函数作为参数。比较容易理解的是在前端语言里调用时会直接调用到这个lambda函数。

在注册的时候，TVM使用了一个全局的Manager对象，该对象中维护了一个function map，该函数的名字和指针就保存在这个map里，以后的查找也是通过这个方式进行
查找。
```c++
Registry& Registry::Register(const std::string& name, bool override) {  // NOLINT(*)
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) {
    Registry* r = new Registry();
    r->name_ = name;
    m->fmap[name] = r;
    return *r;
  } else {
    CHECK(override)
      << "Global PackedFunc " << name << " is already registered";
    return *it->second;
  }
}
```

```c++
class Registry {
 public:
  TVM_DLL Registry& set_body(PackedFunc f);  // NOLINT(*)
  Registry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }
```

set_body函数的实现在src/tvm/runtime/registry.c

TVM是是通过ctyps和ffi机制来实现在python语言里调用c++语言里的函数的。在查找时是通过在python里调用c++实现的TVMFuncGetGlobal函数来得到的。

python/tvm/_ffi/function.py
```python
def get_global_func(name, allow_missing=False):
    handle = FunctionHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return Function(handle, False)

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)
```

可以看到，C++的函数在Python里被封装成一个Function类，而这个类是基于ctyps的FunctionBase定义的

```python
from ._ctypes.function import FunctionBase as _FunctionBase

class Function(_FunctionBase):
    """The PackedFunc object used in TVM.

    Function plays an key role to bridge front and backend in TVM.
    Function provide a type-erased interface, you can call function with positional arguments.

    The compiled module returns Function.
    TVM backend also registers and exposes its API as Functions.
    For example, the developer function exposed in tvm.ir_pass are actually
    C++ functions that are registered as PackedFunc

    The following are list of common usage scenario of tvm.Function.

    - Automatic exposure of C++ API into python
    - To call PackedFunc from python side
    - To call python callbacks to inspect results in generated code
    - Bring python hook into C++ backend

    See Also
    --------
    tvm.register_func: How to register global function.
    tvm.get_global_func: How to get global function.
    """
```

python/tvm/_ffi/_ctypes/node.py
```python
def _return_node(x):
    """Return node function"""
    handle = x.v_handle
    if not isinstance(handle, NodeHandle):
        handle = NodeHandle(handle)
    tindex = ctypes.c_int()
    check_call(_LIB.TVMNodeGetTypeIndex(handle, ctypes.byref(tindex)))
    cls = NODE_TYPE.get(tindex.value, NodeBase)
    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    node = cls.__new__(cls)
    node.handle = handle
    return node
```

那么，在python中是如何识别这些函数并调用的呢？在python代码中我们可以找到初始化的代码：

python/tvm/module.py
```python
_init_api("tvm.module")
_set_class_module(Module)
```

具体实现时，先通过调用C++函数TVMFuncListGlobalNames获取所有注册的函数列表，再根据函数名一个一个的生成PackedFunction，并绑定到指定模块的函数名上，这样在python端调用该函数时，会直接调用该PackedFunction的__call__函数，具体见下面的代码。

python/tvm/_ffi/function.py
```python
def list_global_func_names():
    """Get list of global functions registered.

    Returns
    -------
    names : list
       List of global functions names.
    """
    plist = ctypes.POINTER(ctypes.c_char_p)()
    size = ctypes.c_uint()

    check_call(_LIB.TVMFuncListGlobalNames(ctypes.byref(size),
                                           ctypes.byref(plist)))
    fnames = []
    for i in range(size.value):
        fnames.append(py_str(plist[i]))
    return fnames

def get_global_func(name, allow_missing=False):
    """Get a global function by name

    Parameters
    ----------
    name : str
        The name of the global function

    allow_missing : bool
        Whether allow missing function or raise an error.

    Returns
    -------
    func : tvm.Function
        The function to be returned, None if function is missing.
    """
    handle = FunctionHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))
    if handle.value:
        return Function(handle, False)

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)
    
def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name in list_global_func_names():
        if prefix == "api":
            fname = name
            if name.startswith("_"):
                target_module = sys.modules["tvm._api_internal"]
            else:
                target_module = module
        else:
            if not name.startswith(prefix):
                continue
            fname = name[len(prefix)+1:]
            target_module = module

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = ("TVM PackedFunc %s. " % fname)
        setattr(target_module, ff.__name__, ff)
```
那么还有一个问题，在Python和C++里，各自有对象的表示，在函数调用时，参数和返回值的表示是需要转换的，因为ctypes还做不到玩去自动转换。从前面可以看到Function是继承于FunctionBase的，在函数调用时，缺省调用的是FunctionBase的__call__函数，在函数调用的开头，_make_tvm_args做了函数参数的封装转换，在函数调用的末尾，RETURN_SWITCH[ret_tcode.value](ret_val)对返回值做了转换。

```python
class FunctionBase(object):
    """Function base."""
    __slots__ = ["handle", "is_global"]
    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        """Initialize the function with handle

        Parameters
        ----------
        handle : FunctionHandle
            the handle to the underlying function.

        is_global : bool
            Whether this is a global function in python
        """
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.TVMFuncFree(self.handle) != 0:
                raise get_last_ffi_error()

    def __call__(self, *args):
        """Call the function with positional arguments

        args : list
           The positional arguments to the function call.
        """
        temp_args = []
        values, tcodes, num_args = _make_tvm_args(args, temp_args)
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        if _LIB.TVMFuncCall(
                self.handle, values, tcodes, ctypes.c_int(num_args),
                ctypes.byref(ret_val), ctypes.byref(ret_tcode)) != 0:
            raise get_last_ffi_error()
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)
```

在这里定义了缺省的集中返回值处理方法
python/tvm/_ffi/_ctypes/types.py
```python
RETURN_SWITCH = {
    TypeCode.INT: lambda x: x.v_int64,
    TypeCode.FLOAT: lambda x: x.v_float64,
    TypeCode.HANDLE: _return_handle,
    TypeCode.NULL: lambda x: None,
    TypeCode.STR: lambda x: py_str(x.v_str),
    TypeCode.BYTES: _return_bytes,
    TypeCode.TVM_CONTEXT: _return_context
}
```

这里定义了TypeCode：
python/tvm/_ffi/runtime_ctypes.py
```python
class TypeCode(object):
    """Type code used in API calls"""
    INT = 0
    UINT = 1
    FLOAT = 2
    HANDLE = 3
    NULL = 4
    TVM_TYPE = 5
    TVM_CONTEXT = 6
    ARRAY_HANDLE = 7
    NODE_HANDLE = 8
    MODULE_HANDLE = 9
    FUNC_HANDLE = 10
    STR = 11
    BYTES = 12
    NDARRAY_CONTAINER = 13
    OBJECT_CELL = 14
    EXT_BEGIN = 15
```

有一些TypeCode的处理函数是在其他模块指定的，比如对于Node类型的返回值处理是在这里定义的，至于根据NodeType再选择合适的类型实例化，那是更进一步的处理了，需要在python代码和C++代码之间协商好了。

python/tvm/_ffi/_ctypes/node.py
```python
def _return_node(x):
    """Return node function"""
    handle = x.v_handle
    if not isinstance(handle, NodeHandle):
        handle = NodeHandle(handle)
    tindex = ctypes.c_int()
    check_call(_LIB.TVMNodeGetTypeIndex(handle, ctypes.byref(tindex)))
    cls = NODE_TYPE.get(tindex.value, NodeBase)
    # Avoid calling __init__ of cls, instead directly call __new__
    # This allows child class to implement their own __init__
    node = cls.__new__(cls)
    node.handle = handle
    return node
    
RETURN_SWITCH[TypeCode.NODE_HANDLE] = _return_node
```

在C++侧，函数调用也是先被注册到Registry里的，我们需要看看最终是如何调用到注册的函数以及如何处理参数和返回值的。比如调用tvm.placeholder函数时，最终应该调用到placeholder函数
src/api/api_lang.cc
```c++
TVM_REGISTER_API("_Placeholder")
.set_body_typed<Tensor(Array<Expr>, Type, std::string)>([](
  Array<Expr> shape, Type dtype, std::string name
) {
  return placeholder(shape, dtype, name);
});
```

src/op/placeholder_op.cc
```c++
Tensor placeholder(Array<Expr> shape, Type dtype, std::string name) {
  return PlaceholderOpNode::make(name, shape, dtype).output(0);
}
```

在C++侧，函数调用也是先被封装成了PackedFunc，然后注册到了Registry里，对于像placeholder这样的op，注册时使用了TypedPackedFunc，在调用到实际代码前，先经过detail::unpack_call的几个函数对参数进行转换，最终调用到了实际的代码。

并且PackedFunc重载了运算符()，这样就可以被直接调用。
include/tvm/runtime/registry.h
```c++
class Registry {
 public:
  template<typename FType, typename FLambda>
  Registry& set_body_typed(FLambda f) {
    return set_body(TypedPackedFunc<FType>(f).packed());
  }
};  
```

include/tvm/runtime/packed_func.h
```c++
template<typename R, typename ...Args>
class TypedPackedFunc<R(Args...)> {
 public:
  template<typename FLambda,
           typename = typename std::enable_if<
             std::is_convertible<FLambda,
                                 std::function<R(Args...)>
                                 >::value>::type>
  TypedPackedFunc(const FLambda& typed_lambda) {  // NOLINT(*)
    this->AssignTypedLambda(typed_lambda);
  }
  
 private:
  friend class TVMRetValue;
  /*! \brief The internal packed function */
  PackedFunc packed_;
  template<typename FLambda>
  inline void AssignTypedLambda(FLambda flambda);
};

template<typename R, typename ...Args>
template<typename FType>
inline void TypedPackedFunc<R(Args...)>::AssignTypedLambda(FType flambda) {
  packed_ = PackedFunc([flambda](const TVMArgs& args, TVMRetValue* rv) {
      detail::unpack_call<R, sizeof...(Args)>(flambda, args, rv);
    });
}

template<typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&& ...args) const {
  const int kNumArgs = sizeof...(Args);
  const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
  TVMValue values[kArraySize];
  int type_codes[kArraySize];
  detail::for_each(TVMArgsSetter(values, type_codes),
                   std::forward<Args>(args)...);
  TVMRetValue rv;
  body_(TVMArgs(values, type_codes, kNumArgs), &rv);
  return rv;
}

```
在C++处理参数时，参数一步步的被转换为TVMArgs的形式，全部转换后，根据是否需要返回值，调用不同的unpack_call_dispatcher函数实现。
```c++
namespace detail {
template<typename R, int nleft, int index, typename F>
struct unpack_call_dispatcher {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    unpack_call_dispatcher<R, nleft - 1, index + 1, F>
        ::run(f, args_pack, rv,
              std::forward<Args>(unpacked_args)...,
              args_pack[index]);
  }
};

template<typename R, int index, typename F>
struct unpack_call_dispatcher<R, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    *rv = R(f(std::forward<Args>(unpacked_args)...));
  }
};

template<int index, typename F>
struct unpack_call_dispatcher<void, 0, index, F> {
  template<typename ...Args>
  static void run(const F& f,
                  const TVMArgs& args_pack,
                  TVMRetValue* rv,
                  Args&&... unpacked_args) {
    f(std::forward<Args>(unpacked_args)...);
  }
};

template<typename R, int nargs, typename F>
inline void unpack_call(const F& f, const TVMArgs& args, TVMRetValue* rv) {
  unpack_call_dispatcher<R, nargs, 0, F>::run(f, args, rv);
}

template<typename R>
struct typed_packed_call_dispatcher {
  template<typename ...Args>
  static inline R run(const PackedFunc& pf, Args&& ...args) {
    return pf(std::forward<Args>(args)...);
  }
};

}
```
