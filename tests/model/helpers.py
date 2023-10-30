from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
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
        
class AddSequentialLayer(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(
                name='add_sequential_layer',
                inputs={'x_1':None, 'x_2':None, 'scalar':None},
                outputs={'add_sequential_layer_res':None}
                )

        from tensorflow.keras.layers import Dense

        dense_layer = HierarchalTensorGraph(lambda d: {'y': Dense(1)(d['x'])}, name='dense_layer')
        
        self.add_edge('input', dense_layer)
        self.add_edge(dense_layer, 'output')
        self.rename_io(
            inputs_map={'dense_layer':'x'}, 
            outputs_map={'y':'add_sequential_layer_res'}, 
            node=dense_layer)
        
class AddSquare(HierarchalTensorGraph):
    
    def __init__(self,name='add_square'):
        # Construct a multinode HTG and specify I/O
        super().__init__(name=name,
                         inputs = {'x_1' : None, 'x_2' : None},
                         outputs = {'add_square' : None}
        )
        
        # Create an add basenode and specify I/O
        add = HierarchalTensorGraph(
            lambda X : {'sum' : X['x_1'] + X['x_2']},
            inputs = {'x_1' : None, 'x_2' : None},
            outputs = {'sum' : None},
            name = 'add'
        )
        
        # Create a square basenode and specify I/O
        square = HierarchalTensorGraph(
            lambda X : {'square' : X['x']*X['x']},
            inputs = {'x' : None},
            outputs = {'square' : None},
            name = 'square'
        )
            
        # Create the graph    
        self.add_edge('input',add)
        self.add_edge(add,square)
        self.add_edge(square,'output')
        
        # Don't forget to rename the output of add
        self.rename_io(outputs_map={'sum' : 'x'}, node='add')
        
        # And of square since we specified add_square for the output
        self.rename_io(outputs_map={'square' : 'add_square'}, node=square)

class LogAddSquare(HierarchalTensorGraph):

    def __init__(self,name='log_add_square'):
        from math import log

        super().__init__(
            name=name,
            inputs = {'x_1' : None, 'x_2' : None},
            outputs = {name : None}
        )
        
        # Create log basenode
        lg = HierarchalTensorGraph(
            name = 'log',
            node = lambda X : {'log' : log(X['x'])},
            inputs = {'x' : None},
            outputs = {'log' : None}
        )
        
        # Create graph
        self.add_edge('input',AddSquare())
        self.add_edge('add_square',lg)
        self.add_edge('log','output')
        self.add_edge('add_square','output')
        # Rename log add_square output to match log input
        self.rename_io(outputs_map={'add_square' : 'x'}, node='add_square')

        # Rename log output to match output of LogAddSquare
        self.rename_io(outputs_map={'log' : 'log_add_square'}, node='log')

class SquareRoot:

    custom_lambda  = None

    def __init__(self):
        from math import sqrt
        self.custom_lambda = lambda X : {'square' : sqrt(X['x'])}

    def __call__(self, input):
        return self.custom_lambda(input)

    def serialize(self):
        return '<<serialized--SquareRootCustomSerial>>'
    
    def deserialize(self, serialized_obj):
        return self