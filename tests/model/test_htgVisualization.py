from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
import os

class Add(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        # call the basenode constructor
        super().__init__(lambda X: {'sum':  X['x_1'] + X['x_2']},
                         name=name,
                         inputs={'x_1': None, 'x_2': None},
                         outputs={'sum': None}
                         )


class AddTen(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        # call the basenode constructor
        super().__init__(lambda X: {'sumten':  X['x'] + 10},
                         name=name,
                         inputs={'x': None},
                         outputs={'sumten': None}
                         )


class Multiply(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        # call the basenode constructor
        super().__init__(lambda X: {'product': X['scalar']*X['x']},
                         name=name,
                         inputs={'scalar': None, 'x': None},
                         outputs={'product': None}
                         )


class Subtract(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        # call the basenode constructor
        super().__init__(lambda X: {'difference': X['x_3'] - X['x_4']},
                         name=name,
                         inputs={'x_3': None, 'add_mult': None},
                         outputs={'difference': None}
                         )


class AddMult(HierarchalTensorGraph):
    """Crest HTG to add and multiply a number"""

    def __init__(self, name):
        super().__init__(name=name,
                         inputs={'scalar': None, 'x_1': None, 'x_2': None},
                         outputs={'add_mult': None},
                         edges=[('input', Add('adder')), ('input', Multiply(
                             'multiplier')), ('adder', 'multiplier'), ('multiplier', 'output')]
                         )

        # renaming of keys
        self.rename_io(inputs_map={'sum': 'x'}, outputs_map={
                       'product': 'add_mult'}, node='multiplier')


class AddMultSub(HierarchalTensorGraph):
    """Crest HTG to add and multiply a number"""

    def __init__(self, name):
        super().__init__(name=name,
                         inputs={'scalar': None, 'x_1': None,
                                 'x_2': None, 'x_3': None},
                         outputs={'add_mult_sub': None},
                         edges=[('input', AddMult('add_mult')), ('input', Subtract(
                             'subtractor')), ('add_mult', 'subtractor'), ('subtractor', 'output')]
                         )

        # renaming of keys
        self.rename_io(inputs_map={'add_mult': 'x_4'}, outputs_map={
                       'difference': 'add_mult_sub'}, node='subtractor')


class AddTenAddMultSub(HierarchalTensorGraph):
    """Crest HTG to add and multiply a number"""

    def __init__(self, name):
        super().__init__(name=name,
                         inputs={'scalar': None, 'x_1': None,
                                 'x_2': None, 'x_3': None},
                         outputs={'result': None},
                         edges=[('input', AddTen('add_ten')), ('input', AddMultSub(
                             'add_mult_sub')), ('add_ten', 'add_mult_sub'), ('add_mult_sub', 'output')]
                         )

        # renaming of keys
        self.rename_io(inputs_map={'sumten': 'x_4'}, outputs_map={
                       'difference': 'result'}, node='add_mult_sub')

def test_draw():
    htg = AddTenAddMultSub('test')
    htg.draw_int()
    assert(os.path.exists('interactive_graph.html'))
