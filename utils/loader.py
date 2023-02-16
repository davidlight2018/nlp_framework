import importlib
import logging
import traceback


def flexible_loader(cls_path):
    """Work situations:
        # full path with module and target obj
        from a.b.some_module import some_func : flexible_loader("a.b.some_module.some_func")
        from a.b.some_module import SomeClass : flexible_loader("a.b.some_module.some_class")
        # import obj with same name as it's module
        from a.b.some_func import some_func   : flexible_loader("a.b.some_func")
        from a.b.some_class import SomeClass  : flexible_loader("a.b.some_class")
    """
    if not cls_path:
        return None
    module = None
    final_cls_path = None
    if '.' in cls_path:
        trimmed_cls_path, raw_cls_name = cls_path.rsplit('.', 1)
        try:
            module = importlib.import_module(cls_path)
            final_cls_path = cls_path
        except ImportError as e1:
            if cls_path.split('.')[-1] not in e1.msg:
                print(traceback.format_exc())
                raise e1
            try:
                module = importlib.import_module(trimmed_cls_path)
                final_cls_path = trimmed_cls_path
            except ImportError as e2:
                if trimmed_cls_path.split('.')[-1] not in e2.msg:
                    print(traceback.format_exc())
                    raise e2
                msg = "failed to import path %s: %s; %s" % (
                    cls_path, str(e1), str(e2))
                logging.root.getChild("engine").error(
                    ("action", "flexible_loader"),
                    ("status", False),
                    ("error", msg),
                )
                raise Exception(msg)
    else:
        module = importlib.import_module(cls_path)
        raw_cls_name = cls_path

    cls = getattr(module, raw_cls_name, None)
    if not cls:
        cls_name = ''.join([
            seq.capitalize()
            for seq in raw_cls_name.split('_')
        ])
        cls = getattr(module, cls_name, None)
        if not cls:
            msg = "path %s do not contains class named %s" % (
                final_cls_path, cls_name)
            logging.root.getChild("engine").error(
                ("action", "flexible_loader"),
                ("status", False),
                ("error", msg),
            )
            raise Exception(msg)
    return cls
