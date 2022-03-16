# This for experiements parser
# The parser should parse the experiment file then return a dictionary with the parameters
import json
import numpy as np
from math import sqrt

class Parser:
    @staticmethod
    def dict_filter(data, shared):
        keys = list(data.keys())
        for k in keys:
            if(isinstance(data[k], str) and data[k].lower() == "none"):
                data[k] = None
            # Remove the comments from the json file
            if(k.startswith("__")):
                del data[k]
                continue

            # Put the variables inplace by using the followed of $ as the key ($key)
            if(isinstance(data[k], str) and data[k].startswith("$")):
                data[k] = data[data[k][1:]]

            # Put the variables inplace by using the followed of ~ as the key is in the "shared" dictionary
            if(isinstance(data[k], str) and data[k].startswith("~")):
                if(shared is None):
                    raise ValueError("shared dictionary is empty in the configuration and there is a reference to the shared")
                data[k] = shared[data[k][1:]]
            
            if(isinstance(data[k], list)):
                for i,e in enumerate(data[k]):
                    if(isinstance(e, str) and e.startswith("~")):
                        if(shared is None):
                            raise ValueError("shared dictionary is empty in the configuration and there is a reference to the shared")
                        data[k][i] = shared[e[1:]]


            # Recursive call to filtering the data dictionary
            if(isinstance(data[k], dict)):
                Parser.dict_filter(data[k], shared)

                
    @staticmethod
    def load(filename):
        filename = filename if filename.endswith('.json') else filename+'.json'
        data = None
        with open(filename, 'r') as f:
            data = json.load(f)
            Parser.dict_filter(data, data.get("shared", None))

        return data
    
    @staticmethod
    def save(filename, data):
        filename = filename if filename.endswith('.json') else filename+'.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


class ExperimentParser(Parser):
    @staticmethod
    def load(filename, full=False):
        data = super(ExperimentParser, ExperimentParser).load(filename)
        if(full):
            return data
        experiment = data["experiment"]
        evaluation = data.get("evaluation", {})
        testing = data.get("testing", {})
        agents = {}
        for k in data.keys():
            if(k.startswith("agent")):
                agents[data[k]["name"]] = data[k]
        return experiment, agents, evaluation, testing

    @staticmethod
    def save(filename, experiment, agents, evaluation, testing):
        data = {}
        data["experiment"] = experiment
        data["evaluation"] = evaluation
        data["testing"] = testing
        for k in agents.keys():
            data[f"agent{agents[k]['id']}"] = agents[k]
        super(ExperimentParser, ExperimentParser).save(filename, data)



class WandbHeatMapParser(Parser):
    @staticmethod
    def json2csv(filename, out_filename=None, shapes="auto"):
        filename = filename if filename.endswith('.json') else filename+'.json'
        if(out_filename is None):
            out_filename = filename[:-5]
        out_filename = out_filename if out_filename.endswith('.csv') else out_filename+'.csv'
        data = super(ExperimentParser, ExperimentParser).load(filename)["data"]
        data_shapes = None
        if(shapes == "auto"):
            shape = int(sqrt(len(data)))
            data_shapes = [shape, shape]
        elif(shapes == "limits"):
            pass
        if(isinstance(shapes, list) or isinstance(shapes, tuple)):
            data_shapes = shapes
        data_np = np.zeros(data_shapes)
        for d in data:
            data_np[d[1], d[0]] = d[2]
        np.savetxt(out_filename, data_np, delimiter=",")
        return data_np 

if __name__=="__main__":
    import pprint 
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(ExperimentParser.load("default.json"))
