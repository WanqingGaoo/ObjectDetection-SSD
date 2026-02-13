def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    """
    实现了一个轻量级的「模块注册表」，核心作用是帮你统一管理项目中的各类模块（比如模型、骨干网络、预测器等）
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_module_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    """

    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        # 装饰器式注册的关键是嵌套函数 register_fn，保证不修改原模块功能，只增加注册副作用
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
