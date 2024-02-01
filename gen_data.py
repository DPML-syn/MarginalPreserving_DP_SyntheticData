import numpy as np
import itertools
from mbi import Dataset, GraphicalModel, FactoredInference, Domain
from mechanisms.mechanism import Mechanism
from collections import defaultdict
from hdmm.matrix import Identity
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse
import json

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }

def filter_candidates(candidates, model, size_limit):
    ans = { }
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans

class AIM(Mechanism):
    def __init__(self,epsilon,delta,prng=None,rounds=None,max_model_size=80,structural_zeros={}):
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt) 

        max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data, W):
        rounds = self.rounds or 16*len(data.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2*0.9*self.rho))
        epsilon = np.sqrt(8*0.1*self.rho/rounds)
       
        measurements = []
        print('Initial Sigma', sigma)
        rho_used = len(oneway)*0.5/sigma**2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma,x.size)
            I = Identity(y.size) 
            measurements.append((I, y, sigma, cl))

        zeros = self.structural_zeros
        engine = FactoredInference(data.domain,iters=1000,warm_start=True,structural_zeros=zeros)
        model = engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2*(0.5/sigma**2 + 1.0/8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                epsilon = np.sqrt(8*0.1*remaining)
                terminate = True

            rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
            size_limit = self.max_model_size*rho_used/self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n) 
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = engine.estimate(measurements)
            w = model.project(cl).datavector()
            print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n:
                print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma/2)
                sigma /= 2
                epsilon *= 2

        print('Generating Data...')
        engine.iters = 2500
        model = engine.estimate(measurements)
        synth = model.synthetic_data()

        return synth


dataset_name=['adult','churn', 'compas', 'law', 'dutch', 'heart']

l1_err={}
for i in range(len(dataset_name)):
    print(dataset_name[i])
    l1_err[dataset_name[i]]={}
    df=pd.read_csv ('preprocessd_data/%s.csv'%(dataset_name[i]))
    n=df.shape[0]

    params = {}
    params['dataset'] = 'preprocessd_data/%s.csv'%(dataset_name[i])
    params['domain'] = 'preprocessd_data/%s_domain.json'%(dataset_name[i])
    params['delta'] = 1/n**2
    params['noise'] = 'gaussian'
    params['max_model_size'] = 100
    params['degree'] = 4
    params['num_marginals'] = None
    params['max_cells'] = 1000000     #combination numbers within one marginal

    for j in range(10):     #how many sets of syn want to generate
        l1_err[dataset_name[i]]['set_%i'%j]={}

        # Epsilon = [1/8, 5/32, 6/32, 7/32, 1/4, 1/2, 1, 2]
        Epsilon = [1/4, 2/4, 3/4, 1, 5/4, 6/4, 7/4, 2]
        # Epsilon = [1]
        for epsilon in Epsilon:
            l1_err[dataset_name[i]]['set_%i'%j]['eps=%.3f'%epsilon]={}

            params['epsilon'] = epsilon
            params['save']='output/syn_data/set_%s_syn_%s_eps=%s.csv'%(j, dataset_name[i], epsilon)


            data = Dataset.load(params['dataset'], params['domain'])

            workload = list(itertools.combinations(data.domain, params['degree']))
            workload = [cl for cl in workload if data.domain.size(cl) <= params['max_cells']]
            if params['num_marginals'] is not None:
                workload = [workload[i] for i in prng.choice(len(workload), params['num_marginals'], replace=False)]

            workload = [(cl, 1.0) for cl in workload]
            mech = AIM(params['epsilon'], params['delta'], max_model_size=params['max_model_size'])
            synth = mech.run(data, workload)

            if params['save'] is not None:
                synth.df.to_csv(params['save'], index=False)

            errors = []
            for proj, wgt in workload:
                X = data.project(proj).datavector()
                Y = synth.project(proj).datavector()
                e = 0.5*wgt*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
                errors.append(e)
            print('Average Error: ', np.mean(errors))
            l1_err[dataset_name[i]]['set_%i'%j]['eps=%f3'%epsilon]=np.mean(errors)

file_path = "output/syn_data/l1_error.json"  # You can change the filename and path as needed

# Save the dictionary to a JSON file
with open(file_path, "w") as json_file:
    json.dump(l1_err, json_file, indent=4)
