# This for experiements parser
# The parser should parse the experiment file then return a dictionary with the parameters
from bach_utils.logger import get_logger
clilog = get_logger()

import json
import numpy as np
from math import sqrt

import pprint 
pp = pprint.PrettyPrinter(indent=4)

class Parser:

    @staticmethod
    def merger(data):
        def merge_1lvl(base, extra):
            new = {}
            for k in base.keys():
                new[k] = dict(base[k], **extra[k])
            return new
        header_filename = data["header"]
        shared_filename = data["shared"]
        env_agents_filename = data["env_agents"]
        inner_filename = data["inner_opt"]
        outer_filename = data["outer_opt"]
        testing_filename = data["testing"]

        # print(header_filename, shared_filename)
        shared = Parser.load(shared_filename)["shared"]
        clilog.debug("==============SHARED==============")
        header = Parser.load(header_filename)
        # header = Parser.load(header_filename, shared)
        clilog.debug("=============HEADER===============")


        env_agents = Parser.load(env_agents_filename, shared=False)
        # env_agents = Parser.load(env_agents_filename, shared)
        clilog.debug("=============ENV_agents===============")
        # print(env_agents["experiment"])
        # print("============================")
        # print(header)
        # print("============================")
        header = merge_1lvl(header, env_agents)
        del env_agents["experiment"]
        clilog.debug("============HEADER+ENV_AGENTS================")
        # pp.pprint(header)
        # print("============================")
        inner = Parser.load(inner_filename, shared=False)
        clilog.debug("=============Inner===============")
        env_agents = merge_1lvl(env_agents, inner)
        clilog.debug("==============ENV_AGENTS+INNER==============")
        outer = Parser.load(outer_filename, shared=False)
        clilog.debug("==============OUTER==============")
        header = merge_1lvl(header, outer)
        del outer["experiment"]
        clilog.debug("=============HEADER+OUTER===============")
        env_agents = merge_1lvl(env_agents, outer)
        clilog.debug("=============ENV_AGENTS+OUTER===============")
        header = header["experiment"]

        testing = Parser.load(testing_filename)
        pp.pprint(header)
        # print("========================")
        # pp.pprint(env_agents)
        # print("========================")
        # pp.pprint(shared)
        # print("========================")
        
        # pp.pprint(env_agents)
        # pp.pprint(inner)
        # pp.pprint(outer)
        # new_data = {"experiment": header}
        new_data = {"experiment": header, "shared": shared, "testing": testing}
        new_data = dict(new_data, **env_agents)
        # pp.pprint(new_data)
        clilog.debug("========================")
        return new_data

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
    def load(filename, shared=None):
        filename = filename if filename.endswith('.json') else filename+'.json'
        data = None
        with open(filename, 'r') as f:
            data = json.load(f)
            if(shared != False):
                Parser.dict_filter(data, data.get("shared", shared))

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
        data_combined = data
        if(full):
            return data
        if(data.get("header", None) is not None):   # to make it compatible with the old configs
            data_combined = super(ExperimentParser, ExperimentParser).merger(data)
            super(ExperimentParser, ExperimentParser).dict_filter(data_combined, data_combined.get("shared", None))
            data = data_combined
        experiment = data["experiment"]
        evaluation = data.get("evaluation", {})
        testing = data.get("testing", {})
        agents = {}
        for k in data.keys():
            if(k.startswith("agent")):
                agents[data[k]["name"]] = data[k]
        return experiment, agents, evaluation, testing, data_combined

    @staticmethod
    def save(filename, experiment, agents, evaluation, testing):
        data = {}
        data["experiment"] = experiment
        data["evaluation"] = evaluation
        data["testing"] = testing
        for k in agents.keys():
            data[f"agent{agents[k]['id']}"] = agents[k]
        ExperimentParser.save_combined(filename, data)
        
    @staticmethod
    def save_combined(filename, data):
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
    # pp.pprint(ExperimentParser.load("default.json"))
    gen = ExperimentParser.load("tmp_configs/main.json", full=False)
    pp.pprint(gen[4])
    ExperimentParser.save_combined("tmp_configs/tmp_gen.json", gen[4])
