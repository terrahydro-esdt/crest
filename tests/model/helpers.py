from ...src import HierarchalTensorGraph
from math import exp

class AddMult(HierarchalTensorGraph):
    """
    Crest Model to add and multiply a number

    input keys = 'x_1','x_2','scalar'
    output keys = 'res_add_mult'

    """
    def __init__(self):
        super().__init__(name = 'add_mult')

        def adder(X):
            """ Add two numbers

            input keys = x_1,x_2
            output keys = 'sum'

            """
            return {'sum' : X['x_1'] + X['x_2']}

        def multiplier(X):
            """ Scale a number

            input keys = scalar,x
            output keys = 'product'

            """
            return {'product' : X['scalar']*X['x']}

        #add adder and specify i/o feature names
        self.add_edge('input',adder)
        #self['adder'].set_io_feature_names({'input' : ['x_1','x_2'],'output' : 'sum'})
        self['adder'].input_features = ['x_1','x_2']
        self['adder'].output_features = 'sum'
        #add multiplier
        self.add_edge('input',multiplier)

        #set io. Note, passing a key-value pair instead  of a str is used
        #to change the key, sum, to vec
        #self['multiplier'].set_io_feature_names({'input' : ['scalar', {'sum' : 'x'}]})
        self['multiplier'].input_features = ['scalar', {'sum' : 'x'}]
        self.add_edge(adder,multiplier)
        self.add_edge(multiplier,'output')

        #set i/o for add_mult
        #self.set_io_feature_names({'input' : ['x_1','x_2','scalar'],
         #'output' : [{'product' : 'res_add_mult'}]})
        self.input_features = ['x_1','x_2','scalar']
        self.output_features = {'product' : 'res_add_mult'}

class AddMultExp(HierarchalTensorGraph):

    def __init__(self):
        super().__init__(name='add_mult_exp')

        def exponentiator(X):
            """ exp(X)

            input keys = x
            output keys = 'exp'

            """
            return {'exp' : exp(X['x'])}

        #define model
        self.add_edge('input',AddMult())
        self.add_edge('add_mult',exponentiator)
        self.add_edge('exponentiator','output')
        self['exponentiator'].input_features = {'res_add_mult' : 'x'}
        #set i/o for add_mult
        self.input_features = ['x_1','x_2','scalar']
        self.output_features = {'exp' : 'res_add_mult_exp'}
