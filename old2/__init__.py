import numpy as np
from .layer import Layer
from .activator import Activator
from .loss import Loss
from .descriptor import Descriptor
from .neural_network import NN
from .data import Data
from .question_type import QuestionType
from .mini_batch_strategy import MiniBatchStrategy
from .test_data import * 
from .learning_machine import LearningMachine
from .update_strategy import UpdateStrategy

test_inp = [
    [.2,.3,.4],
    [.5,.0,.1],
    [.4,.1,.8],
]
test_out = [
    [.8,.4],
    [.1,.9],
    [.2,.5],
]
def get_sample_kwargs(): 
    return {
        "learn_rate": 0.2,
        "descriptor": "sample",
        "iteration": 300,
        "mini_batch_strategy": None,
    }

def activator_sample():
    kwargs = get_sample_kwargs()
    for func in Activator.list_up():
        kwargs["func"] = func
        lm = LearningMachine(**kwargs)
        learn_with_template_model(lm)

def update_sample():
    kwargs = get_sample_kwargs()
    kwargs["descriptor"] = Descriptor.create('sample', target_attr_name = 'default_update_strategy')
    kwargs["func"] = 'tanh'
    for upd in UpdateStrategy.list_up():
        kwargs['update_strategy'] = upd
        lm = LearningMachine(**kwargs)
        learn_with_template_model(lm)

def learn_with_template_model(lm):
    lm.add_layer('affine', 3)
    lm.add_layer('affine', 3)
    lm.add_layer('affine', 2)
    lm.learn(test_inp, test_out)

def lm_test():
    d = test_data()
    inp = d["inp"]
    out = d["out"]
    mini_batch_strategy = MiniBatchStrategy.create('flat', blocks = 20)
    lm = LearningMachine(func = 'sigmoid', learn_rate = 0.01, mini_batch_strategy = None)
    lm.add_layer('affine', 10)
    lm.add_layer('affine', 7)
    lm.add_layer('affine', 5)
    lm.add_layer('affine', 2)
    lm.learn(inp, out)
def lm_sin():
    d = test_sin()
    inp = d["inp"]
    out = d["out"]
    mini_batch_strategy = MiniBatchStrategy.create('flat', blocks = 20)
    lm = LearningMachine(func = 'sigmoid', learn_rate = 0.01, mini_batch_strategy = None)
    lm.add_layer('affine', 10)
    lm.add_layer('affine', 5)
    lm.add_layer('affine', 1)
    lm.learn(inp, out)
