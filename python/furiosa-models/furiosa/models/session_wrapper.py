from functools import partial
from typing import Any, Callable, List, Optional

from furiosa.registry import Model
from furiosa.runtime import session


def pipeline(*args: Any, ops: List[Callable]):
    _out = ops[0](*args)
    for op in ops[1:]:
        _out = op(_out)
    return _out


class SessionWrapper(object):
    sess: Optional[Any] = None
    mode: Optional[Model] = None

    def __init__(self, model: Model):
        self.model = model

    def open_session(self):
        self.sess = session.create(self.model.model)

    def close_session(self):
        if not (self.sess is None):
            self.sess.close()

    def inference(self, *args: Any) -> Any:
        return pipeline(*args, ops=[self.model.preprocess, self.sess.run, self.model.postprocess])

    def __enter__(self):
        self.open_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()

    def __del__(self):
        self.close_session()
