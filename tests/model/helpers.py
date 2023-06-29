from ...src import HierarchalTensorGraph
from math import exp

class Adder(HierarchalTensorGraph):
    def __init__(self):
        super().__init__(
                lambda X : {'sum' : X['x_1'] + X['x_2']},
                name = 'adder',
                inputs={'x_1':None, 'x_2':None},
                outputs={'sum':None}
                )

class Multiplier(HierarchalTensorGraph):
    def __init__(self):
        super().__init__(
                lambda X : {'product' : X['scalar']*X['x']},
                name = 'multiplier',
                inputs={'x':None, 'scalar':None},
                outputs={'product':None}
                )

class Exponentiator(HierarchalTensorGraph):
    def __init__(self):
        super().__init__(
                lambda X : {'exp' : exp(X['x'])},
                name = 'exponentiator',
                inputs={'x':None},
                outputs={'exp':None}
                )

class AddMult(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(
                name='add_mult',
                inputs={'x_1':None, 'x_2':None, 'scalar':None},
                outputs={'add_mult_res':None}
                )
        adder = Adder()
        mult  = Multiplier()
        self.add_edge('input',adder)
        self.add_edge('input',mult)
        self.add_edge(adder,mult)
        self.add_edge(mult,'output')
        self.add_edge(adder,mult)
        self.rename_io(inputs_map={'sum' : 'x'},
                       outputs_map={'product' : 'add_mult_res'}, node=mult)

class AddMultExp(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(
                name='add_mult_exp',
                inputs={'x_1':None, 'x_2':None, 'scalar':None},
                outputs={'add_mult_exp_res':None}
                )

        add_mult = AddMult()
        expo = Exponentiator()
        self.add_edge('input',add_mult)
        self.add_edge(add_mult,expo)
        self.add_edge(expo,'output')
        self.rename_io(inputs_map={'add_mult_res':'x'},
                       outputs_map={'exp':'add_mult_exp_res'},node=expo)
