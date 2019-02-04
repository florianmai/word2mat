"""
Some auxiliary functions for saving results, doing hyperparameter optimization, parsing optimizer parameters.
"""

import re
import inspect
from torch import optim
from itertools import product
from bayes_opt import BayesianOptimization
from hyperopt import fmin, rand, hp
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import os, csv

def _update_options(options, **parameters):
    for param_name, param_value in parameters.items():
        print("In automatic optimization trying parameter:", param_name, "with value", param_value)
        
        try:
            setattr(options, param_name, param_value)
        except AttributeError:
            print("Can't find parameter ", param_name, "so we'll not use it.")
            continue
        
    return options


def _make_space(options):
    
    space = {}
    inits = {}
    with open(options.optimization_spaces) as optimization_file:
        for line in optimization_file:
            
            # escape comments
            if line.startswith("#"):
                continue
            
            line = line.strip()
            info = line.split(",")
            param_name = info[0]
            
            if options.optimization == "random":
                left_bound, right_bound = float(info[1]), float(info[2])
                
                param_type = info[3]
                
                try:
                    param_type = getattr(hp, param_type)
                except AttributeError:
                    print("hyperopt has no attribute", param_type)
                    continue
                
                space[param_name] = param_type(param_name, left_bound, right_bound)  
            elif options.optimization == "bayesian":
                left_bound, right_bound = float(info[1]), float(info[2])
                
                init_values = list(map(float, info[3:]))
                num_init_vals = len(init_values)
                inits[param_name] = init_values
                space[param_name] = (left_bound, right_bound)
                
            elif options.optimization == "grid":
                
                param_type = info[1]
                def get_cast_func(some_string_type):
                    cast_func = None
                    if some_string_type == "int":
                        cast_func = int
                    elif some_string_type == "float":
                        cast_func = float
                    elif some_string_type == "string":
                        cast_func = str
                    elif some_string_type == "bool":
                        cast_func = bool
                    return cast_func

                cast_func = get_cast_func(param_type)
                if cast_func is None:                
                    if param_type.startswith("list"):
                        # determine type in list
                        list_type = get_cast_func(param_type.split("-")[1])

                        # assume they are seperated by semicolon
                        def extract_items(list_string):
                            return [list_type(x) for x in list_string.split(";")]

                        cast_func = extract_items
                    
                
                # all possible values
                space[param_name] = list(map(cast_func, info[2:]))
    
    if options.optimization == "bayesian":
        return space, inits, num_init_vals
    else:
        return space
    
def _all_option_combinations(space):
    
    names = [name for name, _ in space.items()]
    values = [values for _, values in space.items()]
    
    val_combinations = product(*values)
    
    combinations = []
    for combi in val_combinations:
        new_param_dict = {}
        for i, val in enumerate(combi):
            new_param_dict[names[i]] = val
        
        combinations.append(new_param_dict)
    
    return combinations
    
def run_hyperparameter_optimization(options, run_exp):
    """
    This function performs hyperparameter optimization using bayesian optimization, random search, or gridsearch.

    It takes an argparse object holding the parameters for configuring an experiments, and a function
    'run_exp' that takes the argparse object, runs an experiments with the respective configuration, and
    returns a score from that configuration.
    It then uses the hyperparameter optimization method to adjust the parameters and run the new configuration.

    Parameters:
    ================
    
    argparse :
        The argparse object holding the parameters. In particular, it must contain the following two parameters.
        'optimization' : str, Specifies the optimization method. Either 'bayesian', 'random', or 'grid'.
        'optimization_spaces' : str, Specifies the path to a file that denotes the parameters to do search over and
        their possible values (in case of grid search) or possible spaces. See file 'default_optimization_space' for
        details.

    run_exp : function
        A function that takes the argparse object as input and returns a float that is interpreted as the
        score of the configuration (higher is better).
    
    """
        
    if options.optimization:
        
        def optimized_experiment(**parameters):
            
            current_options = _update_options(options, **parameters)
            result = run_exp(current_options)
        
            # return the f1 score of the previous experiment
            return result
        
        if options.optimization == "bayesian":
            
            gp_params = {"alpha": 1e-5, "kernel" : Matern(nu = 5 / 2)}
            space, init_vals, num_init_vals = _make_space(options)
            bayesian_optimizer = BayesianOptimization(optimized_experiment, space)
            bayesian_optimizer.explore(init_vals)
            bayesian_optimizer.maximize(n_iter=options.optimization_iterations - num_init_vals,
                                        acq = 'ei',
                                        **gp_params)
            
        elif options.optimization == "random":
            
            fmin(lambda parameters : optimized_experiment(**parameters),
                        _make_space(options),
                        algo=rand.suggest,
                        max_evals=options.optimization_iterations,
                        rstate = np.random.RandomState(1337))
            
        elif options.optimization == "grid":
            # perform grid-search by running every possible parameter combination
            combinations = _all_option_combinations(_make_space(options))
            for combi in combinations:
                optimized_experiment(**combi)
            
    else:
        raise Exception("No hyperparameter method specified!")

def write_to_csv(score, opt):
    """
    Writes the scores and configuration to csv file.
    """
    f = open(opt.output_file, 'a')
    if os.stat(opt.output_file).st_size == 0:
        for i, (key, _) in enumerate(opt.__dict__.items()):
            f.write(key + ";")
        for i, (key, _) in enumerate(score.items()):
            if i < len(score.items()) - 1:
                f.write(key + ";")
            else:
                f.write(key)
        f.write('\n')
        f.flush()
    f.close()

    f = open(opt.output_file, 'r')
    reader = csv.reader(f, delimiter=";")
    column_names = next(reader)
    f.close();

    f = open(opt.output_file, 'a')
    for i, key in enumerate(column_names):
        if i < len(column_names) - 1:
            if key in opt.__dict__:
                f.write(str(opt.__dict__[key]) + ";")
            else:
                f.write(str(score[key]) + ";")
        else:
            if key in opt.__dict__:
                f.write(str(opt.__dict__[key]))
            else:
                f.write(str(score[key]))
    f.write('\n')
    f.flush()
    f.close()

def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'sparseadam':
        optim_fn = optim.SparseAdam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
