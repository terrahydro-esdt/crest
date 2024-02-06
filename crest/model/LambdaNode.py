import marshal
import base64
import types
import json
import traceback
import crest.model.HierarchalTensorGraph as HierarchalTensorGraph

class LambdaNode(HierarchalTensorGraph):
    def __init__(self, node, name: None | str = None, inputs: dict = {}, outputs: dict = {}):
        super().__init__(node=node, name=name,
                         inputs=inputs, outputs=outputs)

    def to_json(self):
        func_serial = marshal.dumps(self.node.__code__)
        func_json = base64.b64encode(func_serial).decode('utf-8')

        lambda_json = self.__dict__.copy()
        
        lambda_json.pop('graph')
        lambda_json.pop('output')

        lambda_json['node'] = func_json
        lambda_json['node_class'] = self.__class__.__name__
        lambda_json['node_module'] = self.__module__
        
        return json.dumps(lambda_json)

    @staticmethod
    def from_json(lambda_json):
        lambda_dict = lambda_json
        if (isinstance(lambda_json, str)):
            lambda_dict = json.loads(lambda_json)

        func_json = lambda_dict['node']

        decoded_str = base64.b64decode(func_json)
        code = marshal.loads(decoded_str)
        lambda_func = types.FunctionType(code, globals(), "lambda")

        ln = LambdaNode(lambda_func, lambda_dict['name'], lambda_dict['inputs'], lambda_dict['outputs'])

        for k, v in lambda_dict.items():
            if k not in ['node']:
                setattr(ln, k, v)

        return ln

    def save(self, path='lambda.json'):
        lamnda_json = self.to_json()
        with open(path, 'w') as f:
            json.dump(lamnda_json, f)
            f.close()

        return True

    @staticmethod
    def load(path='lambda.json'):
        lamnda_json = None
        with open(path, 'r') as f:
            lamnda_json = json.load(f)
            f.close()
        return LambdaNode.from_json(lamnda_json)
