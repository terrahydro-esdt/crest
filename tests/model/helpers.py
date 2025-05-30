from crest import HierarchalTensorGraph
from crest.model.Node import Node
from math import exp
from crest.model.KerasNode import KerasNode
from crest.model.LambdaNode import LambdaNode
import json
import crest.model.TensorSpec as ts
import numpy as np

import crest.model.TensorSpec as ts
import numpy as np


class Adder(Node):
    def __init__(self):
        super().__init__(
            node=lambda X: {'sum': X['x_1'] + X['x_2']},
            name='adder',
            inputs={'x_1': (1,), 'x_2': (1,)},
            outputs={'sum': (1,)}
        )

class Multiplier(Node):
    def __init__(self):
        super().__init__(
            node=lambda X: {'product': X['scalar']*X['x']},
            name='multiplier',
            inputs={'x': (1,), 'scalar': (1,)},
            outputs={'product': (1,)}
        )

class Exponentiator(Node):
    def __init__(self):
        super().__init__(
            node=lambda X: {'exp': exp(X['x'])},
            name='exponentiator',
            inputs={'x': (1,)},
            outputs={'exp': (1,)}
        )

class AddMult(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(name='add_mult')
        adder = Adder()
        mult = Multiplier()
        self.add_edge('input', adder)
        self.add_edge('input', mult,features='scalar')
        self.add_edge(adder,mult,rename={'sum' : 'x'})
        self.add_edge(mult,'output',rename={'product' : 'add_mult_res'})

class AddMultExp(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(name='add_mult_exp')
        add_mult = AddMult()
        expo = Exponentiator()
        self.add_edge('input', add_mult)
        self.add_edge(add_mult, expo,rename={'add_mult_res' : 'x'})
        self.add_edge(expo, 'output',rename={'exp' : 'add_mult_exp_res'})

class AddSequentialLayer(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(
            name='add_sequential_layer',
            inputs={'x_1': None, 'x_2': None, 'scalar': None},
            outputs={'add_sequential_layer_res': None}
        )

        from tensorflow.keras.layers import Dense

        dense_layer = HierarchalTensorGraph(
            lambda d: {'y': Dense(1)(d['x'])}, name='dense_layer')

        self.add_edge('input', dense_layer)
        self.add_edge(dense_layer, 'output')

def AddKerasModel():

    from tensorflow.keras.layers import Dense
    import tensorflow.keras as keras
    import tensorflow as tf

    layers = [keras.layers.Dense(32, activation="relu"), keras.layers.Dropout(
        0.5), keras.layers.Dense(10, activation="softmax")]
    inputs = tf.keras.Input(shape=(32, ))

    outputs = inputs
    for layer in layers:
        outputs = layer(outputs)

    model = tf.keras.Model(inputs={'x': inputs}, outputs={'y': outputs})
    node = KerasNode(model, name='test')

    return node


class AddSquare(HierarchalTensorGraph):

    def __init__(self, name='add_square'):
        # Construct a multinode HTG and specify I/O
        super().__init__(name=name)

        # Create an add basenode and specify I/O
        add = Node(
            node=lambda X: {'sum': X['x_1'] + X['x_2']},
            inputs={'x_1': None, 'x_2': None},
            outputs={'sum': None},
            name='add'
        )

        # Create a square basenode and specify I/O
        square = Node(
            node=lambda X: {'square': X['x']*X['x']},
            inputs={'x': None},
            outputs={'square': None},
            name='square'
        )

        # Create the graph
        self.add_edge('input', add)
        self.add_edge(add, square,rename={'sum','x'})
        self.add_edge(square, 'output',rename={'square','add_square'})

def AddSquareNode():

    # Create and add basenode and specify I/O
    add = LambdaNode(
        lambda X: {'sum': X['x_1'] + X['x_2']},
        inputs={'x_1': None, 'x_2': None},
        outputs={'sum': None},
        name='add'
    )

    # Create a square basenode and specify I/O
    square = LambdaNode(name='square')

    add_square_htg = HierarchalTensorGraph(name="add_square")
    
    # Create the graph
    self.add_edge('input', add)
    self.add_edge(add, square,rename={'sum','x'})
    self.add_edge(square, 'output',rename={'square','add_square'})

    return add_square_htg


class LogAddSquare(HierarchalTensorGraph):

    def __init__(self, name='log_add_square'):
        from math import log

        super().__init__(name=name)

        # Create log basenode
        lg = Node(
            name='log',
            node=lambda X: {'log': log(X['x'])},
            inputs={'x': None},
            outputs={'log': None}
        )

        # Create graph
        self.add_edge('input', AddSquare())
        self.add_edge('add_square', lg,rename={'add_square' : 'x'})
        self.add_edge('log', 'output',rename={'log': 'log_add_square'})
        self.add_edge('add_square', 'output')

def LogAddSquareNode():

    # Create log basenode
    lg = LambdaNode(
        lambda X: {'log': math.log(X['x'])},
        name='log',
        inputs={'x': None},
        outputs={'log': None}
    )

    input_htg = AddSquareNode()

    htg = HierarchalTensorGraph(name="log_add_square")

    # Create graph
    htg.add_edge('input', AddSquare())
    htg.add_edge('add_square', lg,rename={'add_square' : 'x'})
    htg.add_edge('log', 'output',rename={'log': 'log_add_square'})
    htg.add_edge('add_square', 'output')

    return htg


class SquareRoot(HierarchalTensorGraph):

    custom_lambda = None

    def __init__(self, name='square_root', inputs={'x': None}, outputs={'square_root': None}):
        from math import sqrt
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

        super().__init__(
            lambda X: {'square': sqrt(X['x'])},
            name=self.name,
            inputs=self.inputs,
            outputs=self.outputs)

    def to_json(self):
        sq_json = self.__dict__.copy()

        sq_json.pop('graph')
        sq_json.pop('output')
        if ('parent' in sq_json):
            sq_json.pop('parent')

        sq_json['node'] = 'sqroot_node'
        sq_json['node_class'] = self.__class__.__name__
        sq_json['node_module'] = self.__module__

        print(sq_json)
        return json.dumps(sq_json)

    @staticmethod
    def from_json(json_obj):
        return SquareRoot()


class LogCustom():

    custom_lambda = None

    def __init__(self, name='log_custom', inputs={'x': None}, outputs={'log_custom': None}):
        from math import log

        self.custom_lambda = lambda X: {'log': log(X['x'])}
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

    def to_json(self):
        return {
            'name': self.name,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'custom_lambda': 'custom_lambda',
            'node_class': self.__class__.__name__,
            'module': self.__module__
        }

    @staticmethod
    def from_json(json_obj):
        return LogCustom()

    def __call__(self, X):
        return self.custom_lambda(X)

class Identity(Node):
    def __init__(self, name,inputs):
        self.name = name
        super().__init__(
            node=lambda x:x,
            inputs=inputs, 
            outputs=inputs,
            name=name
        )

class Add1_Single(Node):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            v = s_1 + X['x_1']
            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_1': (1, 3), 's_1': (1,)},
                         outputs={'s_1': (1,)},
                         recurrent=True,
                         return_seq=True
                         )

class SingleRecurrent(HierarchalTensorGraph):

    # input, recurrent, output
    def __init__(self):
        super().__init__(name='parent')

        add_single = Add1_Single('add_1')
        self.add_edge('input', add_single)
        self.add_edge(add_single, add_single)
        self.add_edge(add_single, 'output')


class IdentityPlusSingleRecurrent(HierarchalTensorGraph):

    # input, connector, recurrent, output
    def __init__(self):
        super().__init__(name='parent')

        sr = Add1_Single('add_single')
        ident = Identity('add_identity',sr.outputs)
        
        self.add_edge('input', sr)
        self.add_edge(sr,sr)
        self.add_edge(sr,ident)
        self.add_edge(ident, 'output')
        

class Add1_Dual(Node):
    def __init__(self, name):
        
        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            v = s_1 + X['x_1']
            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_1': (1, 3), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_1': (1,)},
                         recurrent=True,
                         reture_seq=False
                         )


class Add2_Dual(Node):
    def __init__(self, name):
 
        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            v = s_1 + s_2 + X['x_2']
            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_2': (1, 3), 's_2':(1,), 's_1': (1,)},
                         outputs={'s_2': (1,)},
                         recurrent=True,
                         reture_seq=False
                         )


class DualRecurrent(HierarchalTensorGraph):

    # hyper connected graph with two recurrent nodes
    def __init__(self):
        super().__init__(name='parent')
        add_1 = Add1_Dual('add_1')
        add_2 = Add2_Dual('add_2')

        self.add_edge('input', add_1,features='x_1')
        self.add_edge('input', add_2,features='x_2')
        self.add_edge(add_1, add_2)
        self.add_edge(add_2, add_1)
        self.add_edge(add_1, add_1)
        self.add_edge(add_2, add_2)
        self.add_edge(add_1, 'output')
        self.add_edge(add_2, 'output')


class Add1_Triple(Node):
    def __init__(self, name):

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            s_3 = X['s_3'] if X['s_3'] is not None else 0
            v = s_1 + s_2 + s_3 + X['x_1']
            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_1': (1, 3), 's_3': (1,), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_1': (1,)},
                         recurrent=True,
                         )


class Add2_Triple(Node):
    def __init__(self, name):
        
        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            s_3 = X['s_3'] if X['s_3'] is not None else 0
            v = s_1 + s_2 + s_3 + X['x_2']
            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_2': (1, 3), 's_3': (1,), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_2': (1,)},
                         recurrent=True
                         )


class Add3_Triple(Node):
    def __init__(self, name):

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            s_3 = X['s_3'] if X['s_3'] is not None else 0
            v = s_1 + s_2 + s_3 + X['x_3']
            return {'s_3':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_3': (1, 3), 's_3': (1,), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_3': (1,)},
                         recurrent=True
                         )


class TripleRecurrent(HierarchalTensorGraph):
    # hyper connected graph with three recurrent nodes
    def __init__(self):
        super().__init__(name='parent')
        nodes = [
            Add1_Triple('add_1'),
            Add2_Triple('add_2'),
            Add3_Triple('add_3')
        ]
        for i in zip(nodes,range(1,4)):
            self.add_edge('input',i[0],features='x_' + str(i[1]))
            self.add_edge(i[0],'output')

        for i in nodes:
            for j in nodes:
                self.add_edge(i,j)