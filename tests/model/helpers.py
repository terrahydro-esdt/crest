from crest import HierarchalTensorGraph
from math import exp
from crest.model.KerasNode import KerasNode
from crest.model.LambdaNode import LambdaNode
import json
import crest.model.TensorSpec as ts
import numpy as np

import crest.model.TensorSpec as ts
import numpy as np


class Adder(HierarchalTensorGraph):
    def __init__(self):
        super().__init__(
            lambda X: {'sum': X['x_1'] + X['x_2']},
            name='adder',
            inputs={'x_1': None, 'x_2': None},
            outputs={'sum': None}
        )


class Multiplier(HierarchalTensorGraph):
    def __init__(self):
        super().__init__(
            lambda X: {'product': X['scalar']*X['x']},
            name='multiplier',
            inputs={'x': None, 'scalar': None},
            outputs={'product': None}
        )


class Exponentiator(HierarchalTensorGraph):
    def __init__(self):
        super().__init__(
            lambda X: {'exp': exp(X['x'])},
            name='exponentiator',
            inputs={'x': None},
            outputs={'exp': None}
        )


class AddMult(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(
            name='add_mult',
            inputs={'x_1': None, 'x_2': None, 'scalar': None},
            outputs={'add_mult_res': None}
        )
        adder = Adder()
        mult = Multiplier()
        self.add_edge('input', adder)
        self.add_edge('input', mult)
        self.add_edge(adder, mult)
        self.add_edge(mult, 'output')
        self.add_edge(adder, mult)
        self.rename_io(inputs_map={'sum': 'x'},
                       outputs_map={'product': 'add_mult_res'}, node=mult)


class AddMultExp(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(
            name='add_mult_exp',
            inputs={'x_1': None, 'x_2': None, 'scalar': None},
            outputs={'add_mult_exp_res': None}
        )

        add_mult = AddMult()
        expo = Exponentiator()
        self.add_edge('input', add_mult)
        self.add_edge(add_mult, expo)
        self.add_edge(expo, 'output')
        self.rename_io(inputs_map={'add_mult_res': 'x'},
                       outputs_map={'exp': 'add_mult_exp_res'}, node=expo)


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
        self.rename_io(
            inputs_map={'dense_layer': 'x'},
            outputs_map={'y': 'add_sequential_layer_res'},
            node=dense_layer)


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
        super().__init__(name=name,
                         inputs={'x_1': None, 'x_2': None},
                         outputs={'add_square': None}
                         )

        # Create an add basenode and specify I/O
        add = HierarchalTensorGraph(
            lambda X: {'sum': X['x_1'] + X['x_2']},
            inputs={'x_1': None, 'x_2': None},
            outputs={'sum': None},
            name='add'
        )

        # Create a square basenode and specify I/O
        square = HierarchalTensorGraph(
            lambda X: {'square': X['x']*X['x']},
            inputs={'x': None},
            outputs={'square': None},
            name='square'
        )

        # Create the graph
        self.add_edge('input', add)
        self.add_edge(add, square)
        self.add_edge(square, 'output')

        # Don't forget to rename the output of add
        self.rename_io(outputs_map={'sum': 'x'}, node='add')

        # And of square since we specified add_square for the output
        self.rename_io(outputs_map={'square': 'add_square'}, node='square')


def AddSquareNode():

    # Create an add basenode and specify I/O
    add = LambdaNode(
        lambda X: {'sum': X['x_1'] + X['x_2']},
        inputs={'x_1': None, 'x_2': None},
        outputs={'sum': None},
        name='add'
    )

    # Create a square basenode and specify I/O
    square = LambdaNode(
        lambda X: {'square': X['x']*X['x']},
        inputs={'x': None},
        outputs={'square': None},
        name='square'
    )

    add_square_htg = HierarchalTensorGraph(name="add_square")

    # Create the graph
    add_square_htg.add_edge('input', add)
    add_square_htg.add_edge(add, square)
    add_square_htg.add_edge(square, 'output')

    # Don't forget to rename the output of add
    add_square_htg.rename_io(outputs_map={'sum': 'x'}, node='add')

    # And of square since we specified add_square for the output
    add_square_htg.rename_io(
        outputs_map={'square': 'add_square'}, node='square')

    return add_square_htg


class LogAddSquare(HierarchalTensorGraph):

    def __init__(self, name='log_add_square'):
        from math import log

        super().__init__(
            name=name,
            inputs={'x_1': None, 'x_2': None},
            outputs={name: None}
        )

        # Create log basenode
        lg = HierarchalTensorGraph(
            name='log',
            node=lambda X: {'log': log(X['x'])},
            inputs={'x': None},
            outputs={'log': None}
        )

        # Create graph
        self.add_edge('input', AddSquare())
        self.add_edge('add_square', lg)
        self.add_edge('log', 'output')
        self.add_edge('add_square', 'output')
        # Rename log add_square output to match log input
        self.rename_io(outputs_map={'add_square': 'x'}, node='add_square')

        # Rename log output to match output of LogAddSquare
        self.rename_io(outputs_map={'log': 'log_add_square'}, node='log')


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

    htg.add_edge('input', input_htg)
    htg.add_edge(input_htg, lg)
    htg.add_edge(lg, 'output')
    htg.add_edge(input_htg, 'output')

    htg.rename_io(outputs_map={'add_square': 'x'}, node='add_square')
    htg.rename_io(outputs_map={'log': 'log_add_square'}, node='log')

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


class Add_Identity(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add_identity(X):
            s = X['x_1'] + X['x_2']
            return {'x_1': X['x_1'], 'x_2': X['x_2'], 's': s}

        # call the basenode constructor
        super().__init__(add_identity, inputs={'x_1': None, 'x_2': None}, outputs={
            'x_1': None, 'x_2': None, 's': None})


class Add2_Identity(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def identity_2(X):
            return X

        # call the basenode constructor
        super().__init__(identity_2, inputs={
            's_1': None}, outputs={'s_1': None})


class Add3_Identity(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def identity(X):
            print(f'ID {X=}')
            return X

        # call the basenode constructor
        super().__init__(identity, inputs={'x_1': None}, outputs={'x_1': None})


class Add1_Single(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0

            v = s_1 + X['x_1']

            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_1': np.ones((1, 3)), 's_1': np.ones((1))},
                         outputs={'s_1': None}
                         )


class SingleRecurrent(HierarchalTensorGraph):

    # input, recurrent, output
    def __init__(self):
        super().__init__(name='parent')

        add_single = Add1_Single('add_1')

        self.add_edge('input', add_single, rollout_axis=1)
        self.add_edge(add_single, add_single, rollout_axis=1)
        self.add_edge(add_single, 'output')


class IdentityPlusSingleRecurrent(HierarchalTensorGraph):

    # input, connector, recurrent, output
    def __init__(self):
        super().__init__(name='parent')

        add_single = Add1_Single('add_single')
        add_identity = Add3_Identity('add_identity')

        self.add_edge('input', add_identity)
        self.add_edge(add_identity, add_single, rollout_axis=1)
        self.add_edge(add_single, add_single, rollout_axis=1)
        self.add_edge(add_single, 'output')


class Add1_Dual(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0

            v = s_1 + X['x_1']

            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_1': np.ones((1, 3)), 's_2': np.ones(
                             (1)), 's_1': np.ones((1))},
                         outputs={'s_1': None}
                         )


class Add2_Dual(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0

            v = s_1 + s_2 + X['x_2']

            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_2': np.ones((1, 3)), 's_2': np.ones(
                             (1)), 's_1': np.ones((1))},
                         outputs={'s_2': None}
                         )


class DualRecurrent(HierarchalTensorGraph):

    # hyper connected graph with two recurrent nodes
    def __init__(self):
        super().__init__(name='parent')
        add_1 = Add1_Dual('add_1')
        add_2 = Add2_Dual('add_2')

        self.add_edge('input', add_1, rollout_axis=1)
        self.add_edge('input', add_2, rollout_axis=1)
        self.add_edge(add_1, add_2, rollout_axis=1)
        self.add_edge(add_2, add_1, rollout_axis=1)
        self.add_edge(add_1, add_1, rollout_axis=1)
        self.add_edge(add_2, add_2, rollout_axis=1)
        self.add_edge(add_1, 'output')
        self.add_edge(add_2, 'output')


class Add1_Triple(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            s_3 = X['s_3'] if X['s_3'] is not None else 0
            v = s_1 + s_2 + s_3 + X['x_1']

            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_1': np.ones((1, 3)), 's_3': np.ones(
                             (1)), 's_2': np.ones((1)), 's_1': np.ones((1))},
                         outputs={'s_1': None}
                         )


class Add2_Triple(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            s_3 = X['s_3'] if X['s_3'] is not None else 0

            v = s_1 + s_2 + s_3 + X['x_2']

            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_2': np.ones((1, 3)), 's_3': np.ones(
                             (1)), 's_2': np.ones((1)), 's_1': np.ones((1))},
                         outputs={'s_2': None}
                         )


class Add3_Triple(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0
            s_3 = X['s_3'] if X['s_3'] is not None else 0

            v = s_1 + s_2 + s_3 + X['x_3']

            return {'s_3':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_3': np.ones((1, 3)), 's_3': np.ones(
                             (1)), 's_2': np.ones((1)), 's_1': np.ones((1))},
                         outputs={'s_3': None}
                         )


class TripleRecurrent(HierarchalTensorGraph):

    # hyper connected graph with three recurrent nodes
    def __init__(self):
        super().__init__(name='parent')

        add_1 = Add1_Triple('add_1')
        add_2 = Add2_Triple('add_2')
        add_3 = Add3_Triple('add_3')

        self.add_edge('input', add_1, rollout_axis=1)
        self.add_edge('input', add_2, rollout_axis=1)
        self.add_edge('input', add_3, rollout_axis=1)
        self.add_edge(add_1, add_2, rollout_axis=1)
        self.add_edge(add_2, add_1, rollout_axis=1)
        self.add_edge(add_2, add_3, rollout_axis=1)
        self.add_edge(add_3, add_2, rollout_axis=1)
        self.add_edge(add_3, add_1, rollout_axis=1)
        self.add_edge(add_1, add_3, rollout_axis=1)
        self.add_edge(add_1, add_1, rollout_axis=1)
        self.add_edge(add_2, add_2, rollout_axis=1)
        self.add_edge(add_3, add_3, rollout_axis=1)
        self.add_edge(add_1, 'output')
        self.add_edge(add_2, 'output')
        self.add_edge(add_3, 'output')


class Add1_Dual_v2(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0

            v = s_1 + X['x_1']

            return {'s_1':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'x_1': np.ones((1, 3)), 's_1': np.ones((1))},
                         outputs={'s_1': None}
                         )


class Add2_Dual_v2(HierarchalTensorGraph):
    def __init__(self, name):
        self.name = name

        def add(X):
            s_1 = X['s_1'] if X['s_1'] is not None else 0
            s_2 = X['s_2'] if X['s_2'] is not None else 0

            v = s_1 + s_2

            return {'s_2':  v}

        # call the basenode constructor
        super().__init__(add,
                         name=name,
                         inputs={'s_2': np.ones(
                             (1)), 's_1': np.ones((1))},
                         outputs={'s_2': None}
                         )


class IdentityPlusDualRecurrent(HierarchalTensorGraph):

    # input, connector, recurrent, output
    def __init__(self):
        super().__init__(name='parent')

        add1_dual = Add1_Dual_v2('add_1_dual')
        add2_dual = Add2_Dual_v2('add_2_dual')
        add_identity = Add_Identity('add_identity')
        add2_identity = Add2_Identity('add2_identity')

        self.add_edge('input', add_identity)
        self.add_edge(add_identity, add1_dual, rollout_axis=1)
        self.add_edge(add1_dual, add2_identity, rollout_axis=1)
        self.add_edge(add2_identity, add2_dual, rollout_axis=1)
        self.add_edge(add1_dual, add1_dual, rollout_axis=1)
        self.add_edge(add2_dual, add2_dual, rollout_axis=1)
        self.add_edge(add2_dual, 'output')
