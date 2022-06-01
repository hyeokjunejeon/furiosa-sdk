from typing import Any, Callable, List, Optional

from furiosa.registry import Model
from furiosa.runtime import session

class LazyPipeLine:
    def __init__(self, value: object):
        if isinstance(value, Callable):
            self.compute = value
        else:
            def return_val():
                return value
            self.compute = return_val

    def bind(self, f: Callable, *args, **kwargs) -> 'LazyPipeLine':
        def f_compute():
            return f(self.compute(), *args, **kwargs)
        return LazyPipeLine(f_compute)

class SessionWrapper(object):
    sess: Optional[Any] = None
    model: Optional[Model] = None

    def __init__(self, model: Model):
        self.model = model

    def open_session(self):
        self.sess = session.create(self.model.model)

    def close_session(self):
        if not (self.sess is None):
            self.sess.close()

    def inference(self, *args: Any) -> Any:
        return (LazyPipeLine(*args)
            .bind(self.model.preprocess)
            .bind(self.sess.run)
            .bind(self.model.postprocess)
            .compute())

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    def __del__(self):
        self.close_session()
