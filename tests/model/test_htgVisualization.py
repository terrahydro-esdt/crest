from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
from crest.model.Node import Node
import os

class Add(Node):
    def __init__(self, name):
        super().__init__(lambda X: {'sum':  X['x_1'] + X['x_2']},
                         name=name,
                         inputs={'x_1': None, 'x_2': None},
                         outputs={'sum': None}
                         )


class AddTen(Node):
    def __init__(self, name):
        super().__init__(lambda X: {'sumten':  X['x'] + 10},
                         name=name,
                         inputs={'x': None},
                         outputs={'sumten': None}
                         )


class Multiply(Node):
    def __init__(self, name):
        super().__init__(lambda X: {'product': X['scalar']*X['x']},
                         name=name,
                         inputs={'scalar': None, 'x': None},
                         outputs={'product': None}
                         )


class Subtract(Node):
    def __init__(self, name):
        super().__init__(lambda X: {'difference': X['x_3'] - X['x_4']},
                         name=name,
                         inputs={'x_3': None, 'add_mult': None},
                         outputs={'difference': None}
                         )


class AddMult(HierarchalTensorGraph):
    """Crest HTG to add and multiply a number"""

    registry_name = 'CREST_VIZ_ADD_MULT'
    def __init__(self, name):
        super().__init__(name=name,
                         edges=[('input', Add('adder')), 
                                ('input', Multiply('multiplier')), 
                                ('adder', 'multiplier',{'rename' : {'sum':'x'}}), 
                                ('multiplier', 'output',{'rename' : {'product','add_mult'}})
                                ]
                         )


class AddMultSub(HierarchalTensorGraph):
    """Crest HTG to add and multiply a number"""

    def __init__(self, name):
        super().__init__(name=name,
                         edges=[('input', AddMult('add_mult')), 
                                ('input', Subtract('subtractor')), 
                                ('add_mult', 'subtractor',{'rename' : {'add_mult' : 'x_2'}}), 
                                 ('subtractor', 'output',{'rename' : {'difference':'add_mult_sub'}})
                                ]
                         )

class AddTenAddMultSub(HierarchalTensorGraph):
    """Crest HTG to add and multiply a number"""

    def __init__(self, name):
        super().__init__(name=name,
                         edges=[('input', AddTen('add_ten')), 
                                ('input', AddMultSub('add_mult_sub')), 
                                ('add_ten', 'add_mult_sub',{'rename':{'sumten':'x_4'}}), 
                                ('add_mult_sub', 'output',{'rename':{'add_mult_sub' : 'result'}})
                                ]
                         )
#TODO: merge in draw int
def test_draw():
    htg = AddTenAddMultSub('test')
    #htg.draw_int()
    #assert(os.path.exists('interactive_graph.html'))
