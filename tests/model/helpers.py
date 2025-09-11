from crest.model.HierarchalTensorGraph import HierarchalTensorGraph
from crest.model.Node import Node
#from crest.nodes.tensorflow.KerasNode import KerasNode
from functools import cached_property
import json
import crest.model.TensorSpec as ts
import numpy as np
from math import exp,log,sqrt


class Adder(Node):
    def __init__(self):
        super().__init__(
            node=lambda X: {'sum': X['x_1'] + X['x_2']},
            name='adder',
            inputs={'x_1': (1,), 'x_2': (1,)},
            outputs={'sum': (1,)}
        )

    @property
    def json_dict(self):
        pass

class Square(Node):
    def __init__(self):
      super().__init__(**{
          'node': lambda X: {'square': X['x']*X['x']},
          'inputs': {'x': (1,)},
          'outputs': {'square': (1,)},
          'name': 'square'
      })

    @property
    def json_dict(self):
        pass

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

        def to_json(self):
            return super().to_json()

        @staticmethod
        def from_json(json_str):
            jd = HierarchalTensorGraph.from_json(json_str)
            obj = AddMult(False)
            return super(AddMult,obj).build_from_json(jd)

class AddMultExp(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(name='add_mult_exp')
        add_mult = AddMult()
        expo = Exponentiator()
        self.add_edge('input', add_mult)
        self.add_edge(add_mult, expo,rename={'add_mult_res' : 'x'})
        self.add_edge(expo, 'output',rename={'exp' : 'add_mult_exp_res'})

#def AddKerasModel():
#
#    from tensorflow.keras.layers import Dense
#    import tensorflow.keras as keras
#    import tensorflow as tf
#
#    layers = [
#        keras.layers.Dense(32, activation="relu"),
#        keras.layers.Dropout(0.5),
#        keras.layers.Dense(10, activation="softmax")
#    ]
#
#    inputs = tf.keras.Input(shape=(32, ))
#    outputs = inputs
#    for layer in layers:
#        outputs = layer(outputs)
#
#    model = tf.keras.Model(inputs={'x': inputs}, outputs={'y': outputs})
#    node = KerasNode(model, name='test')
#
#    return node


class AddSquare(HierarchalTensorGraph):

    def __init__(self, name='add_square'):
        # Construct a multinode HTG and specify I/O
        super().__init__(name=name)

        add = Adder()
        square = Square()

        # Create the graph
        self.add_edge('input', add)
        self.add_edge(add, square,rename={'sum' : 'x'})
        self.add_edge(square, 'output',rename={'square' : 'add_square'})

    @classmethod
    def from_json(cls,json_str):
        jd = json.loads(json_str)
        obj = cls(jd['name'])
        obj.super().from_json(json_str)
        return


class LogAddSquare(HierarchalTensorGraph):
    from math import log
    def __init__(self, name='log_add_square',build=True):
        super().__init__(name=name)

        if(build):
            # Create log basenode
            lg = Node(**{
                'name': 'log',
                'node': lambda X: {'log': log(X['x'])},
                'inputs': {'x': None},
                'outputs': {'log': None}
            })

            self.add_edge('input', AddSquare())
            self.add_edge('add_square',lg,rename={'add_square' : 'x'})
            self.add_edge('log', 'output',rename={'log': 'log_add_square'})
            self.add_edge('add_square', 'output')

        def to_json():
            super().to_json()

        @staticmethod
        def from_json(json_str):
            jd = HierarchalTensorGraph.load_json(json_str)
            obj = LogAddSquare(jd['name'],False)
            super(LogAddSquare,obj).build_from_json(jd)
            return obj


class SquareRoot(HierarchalTensorGraph):

    custom_lambda = None

    def __init__(self, name='square_root', inputs={'x': None}, outputs={'square_root': None}):
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

        def add(X):
            v = X['s_1'] + X['x_1']
            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_1': (1, 3), 's_1': (1,)},
                         outputs={'s_1': (1,)},
                         recurrent=True,
                         return_seq=True
                         )

    def initial_state(self,X):
        return {'s_1' : 0}



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
            v = X['s_1'] + X['x_1']
            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_1': (1, 3), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_1': (1,)},
                         recurrent=True,
                         reture_seq=False
                         )

    def initial_state(self,X):
        return {'s_1' : 0}


class Add2_Dual(Node):
    def __init__(self, name):

        def add(X):
            v = X['s_1'] + X['s_2'] + X['x_2']
            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_2': (1, 3), 's_2':(1,), 's_1': (1,)},
                         outputs={'s_2': (1,)},
                         recurrent=True,
                         reture_seq=False
                         )

    def initial_state(self,X):
        return {'s_2': 0}



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
            v = X['s_1'] + X['s_2'] + X['s_3'] + X['x_1']
            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(node=add,
                         name=name,
                         inputs={'x_1': (1, 3), 's_3': (1,), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_1': (1,)},
                         recurrent=True,
                         )

    def initial_state(self,X):
        return {'s_1' : 0}



class Add2_Triple(Node):
    def __init__(self, name):

        def add(X):
            v = X['s_1'] + X['s_2'] + X['s_3'] + X['x_2']
            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_2': (1, 3), 's_3': (1,), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_2': (1,)},
                         recurrent=True
                         )

    def initial_state(self,X):
        return {'s_2' : 0}


class Add3_Triple(Node):
    def __init__(self, name):

        def add(X):
            v = X['s_1'] + X['s_2'] + X['s_3'] + X['x_3']
            return {'s_3':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_3': (1, 3), 's_3': (1,), 's_2': (1,), 's_1': (1,)},
                         outputs={'s_3': (1,)},
                         recurrent=True
                         )

    def initial_state(self,X):
        return {'s_3' : 0}


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
