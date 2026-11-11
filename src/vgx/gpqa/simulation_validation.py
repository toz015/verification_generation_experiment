"""Exact finite-tree experiments with known joint verifier distributions.

The oracle knows distributions, not the current answer's hidden correctness or
future signals. It obeys the same fixed order and cannot skip a verifier.
"""
from __future__ import annotations

from functools import lru_cache
import itertools
import math

import numpy as np

from vgx.gpqa.planner import NestedPlanner
from vgx.gpqa.score import VerifierLikelihood, fit_verifier_likelihood


def joint_distribution(positive_probabilities,rho=0.):
    """Mix independent signals and shared-uniform signals, preserving marginals."""
    if not 0<=rho<=1 or any(not 0<=p<=1 for p in positive_probabilities):
        raise ValueError('invalid signal probabilities or mixture weight')
    result={}
    for bits in itertools.product((0,1),repeat=len(positive_probabilities)):
        independent=math.prod(p if bit else 1-p for bit,p in zip(bits,positive_probabilities))
        lower=max([p for bit,p in zip(bits,positive_probabilities) if not bit] or [0.])
        upper=min([p for bit,p in zip(bits,positive_probabilities) if bit] or [1.])
        result[bits]=(1-rho)*independent+rho*max(0.,upper-lower)
    return result


class VerificationWorld:
    def __init__(self,sensitivity,specificity,costs,*,reward=1.,loss=19.,rho=0.):
        if len(sensitivity)!=len(specificity) or len(costs)!=len(sensitivity):
            raise ValueError('one sensitivity, specificity and cost per verifier')
        self.joints=(joint_distribution([1-s for s in specificity],rho),joint_distribution(sensitivity,rho))
        self.costs=tuple(costs);self.reward=reward;self.loss=loss
        self.channels=tuple(VerifierLikelihood((0.,.5,1.),(1-se,se),(sp,1-sp)) for se,sp in zip(sensitivity,specificity))

    def mass(self,y,prefix):
        return sum(p for bits,p in self.joints[y].items() if bits[:len(prefix)]==prefix)

    def belief(self,prior,prefix):
        n=prior*self.mass(1,prefix);m=n+(1-prior)*self.mass(0,prefix)
        return (n/m if m else 0.),m

    @lru_cache(maxsize=None)
    def oracle(self,prior,prefix=()):
        b,mass=self.belief(prior,prefix)
        assertion=(self.reward+self.loss)*b-self.loss
        stop=max(0.,assertion);stage=len(prefix)
        query=None
        if stage<len(self.costs) and mass:
            query=-self.costs[stage]
            for signal in (0,1):
                _,child_mass=self.belief(prior,(*prefix,signal))
                if child_mass:
                    query+=child_mass/mass*self.oracle(prior,(*prefix,signal))['value']
        action='query' if query is not None and query>stop else ('assert' if assertion>=0 else 'abstain')
        return {'value':max(stop,query) if query is not None else stop,'query_value':query,'action':action}

    def expected(self,prior,kind,*,reported_prior=None,grid_size=1001,channels=None,planner=None):
        from vgx.gpqa.offline_validation import decide_policy
        reported_prior=prior if reported_prior is None else reported_prior
        planner=planner or NestedPlanner(channels or self.channels,self.costs,self.reward,self.loss,grid_size=grid_size)
        values={'utility':0.,'coverage':0.,'wrong_release_mass':0.,'queries':0.,'cost':0.}
        for y in (0,1):
            for bits,p in self.joints[y].items():
                weight=p*(prior if y else 1-prior)
                if not weight:continue
                if kind=='oracle':
                    stage=0
                    while self.oracle(prior,bits[:stage])['action']=='query':stage+=1
                    action=self.oracle(prior,bits[:stage])['action'];used=stage
                else:
                    d=decide_policy(reported_prior,[.75 if bit else .25 for bit in bits],planner,kind)
                    action,used=d['action'],d['used']
                release=action=='assert';cost=sum(self.costs[:used])
                values['utility']+=weight*((self.reward if y else -self.loss)*release-cost)
                values['coverage']+=weight*release;values['wrong_release_mass']+=weight*release*(1-y)
                values['queries']+=weight*used;values['cost']+=weight*cost
        values['accuracy_among_released']=1-values['wrong_release_mass']/values['coverage'] if values['coverage'] else None
        values['predicted_nested_value']=planner.decide(0,reported_prior)['value']
        values['oracle_value']=self.oracle(prior)['value']
        values['regret']=max(0.,values['oracle_value']-values['utility'])
        return values


def simulation_study(config):
    priors=[.50,.75,.85,.90,.94,.95,.96,.98,.99]
    definitions=[
        ('independent_correct_prior',(.60,.75,.90),(.60,.75,.90),(.002,.01,.05),19.,0.,0.),
        ('correlated_correct_prior',(.60,.75,.90),(.60,.75,.90),(.002,.01,.05),19.,.8,0.),
        ('independent_overconfident_prior',(.60,.75,.90),(.60,.75,.90),(.002,.01,.05),19.,0.,1.2),
        ('correlated_overconfident_prior',(.60,.75,.90),(.60,.75,.90),(.002,.01,.05),19.,.8,1.2),
        ('uninformative_positive_cost',(.5,.5),(.5,.5),(.01,.01),1.,0.,0.),
        ('uninformative_gateway_to_useful_verifier',(.5,.9),(.5,.9),(.001,.01),1.,0.,0.),
    ]
    cases={}
    for name,se,sp,costs,loss,rho,bias in definitions:
        world=VerificationWorld(se,sp,costs,loss=loss,rho=rho)
        rows=[]
        planner=NestedPlanner(world.channels,costs,1.,loss)
        for b in priors:
            reported=1/(1+math.exp(-(math.log(b/(1-b))+bias))) if bias else b
            rows.append({'true_prior':b,'reported_prior':reported,'policies':{
                k:world.expected(b,k,reported_prior=reported,planner=planner) for k in ('none','always','myopic','nested','oracle')}})
        cases[name]={'sensitivity':se,'specificity':sp,'costs':costs,'reward':1.,'loss':loss,
                     'dependence_mixture_rho':rho,'reported_log_odds_bias':bias,'prior_grid':rows,
                     'mean_over_prior_grid':{k:{m:float(np.mean([r['policies'][k][m] for r in rows]))
                         for m in ('utility','coverage','wrong_release_mass','queries','cost','regret','predicted_nested_value')}
                         for k in ('none','always','myopic','nested','oracle')}}
    known=VerificationWorld((.60,.75,.90),(.60,.75,.90),(.002,.01,.05))
    grid_validation=[]
    for size in (101,1001,10001):
        planner=NestedPlanner(known.channels,known.costs,known.reward,known.loss,grid_size=size)
        error=[];regret=[];disagree=0
        for b in np.linspace(.001,.999,101):
            b=float(b);exact=known.oracle(b)
            d=planner.decide(0,b)
            error.append(abs(d['value']-exact['value']))
            actual=known.expected(b,'nested',planner=planner)
            regret.append(actual['regret'])
            if d['action']!=exact['action'] and exact['query_value'] is not None and abs(exact['query_value']-max(0.,20*b-19))>1e-9:
                disagree+=1
        grid_validation.append({'grid_size':size,'max_value_error':max(error),'mean_value_error':float(np.mean(error)),
                                'max_policy_regret':max(regret),'initial_action_disagreements':disagree})
    rng=np.random.default_rng(config['seed']);learning=[]
    sample_priors=np.array(priors)
    for n in config['simulation_learning_sizes']:
        regrets=[];utilities=[];negative_counts=[];failed=0
        for _ in range(config['simulation_learning_repeats']):
            p=rng.choice(sample_priors,n);y=rng.binomial(1,p)
            signals=[]
            for label in y:
                paths=list(known.joints[int(label)])
                signals.append(paths[int(rng.choice(len(paths),p=list(known.joints[int(label)].values())))])
            negative_counts.append(int((1-y).sum()))
            try:
                fits=[fit_verifier_likelihood(y.tolist(),[.75 if bits[i] else .25 for bits in signals],bins=2) for i in range(3)]
            except ValueError:
                failed+=1;continue
            pp=NestedPlanner(fits,known.costs,known.reward,known.loss)
            stats=[known.expected(b,'nested',planner=pp) for b in priors]
            regrets.append(float(np.mean([r['regret'] for r in stats])))
            utilities.append(float(np.mean([r['utility'] for r in stats])))
        learning.append({'calibration_n':n,'successful_replicates':len(regrets),'failed_replicates':failed,
                         'mean_incorrect_calibration_n':float(np.mean(negative_counts)),
                         'mean_regret':float(np.mean(regrets)),'regret_interval_across_simulated_calibrations':np.quantile(regrets,[.025,.975]).tolist(),
                         'mean_utility':float(np.mean(utilities))})
    return {'integration':'Exact summation over correctness and binary signal paths; no Monte Carlo evaluation noise.',
            'oracle':'Bayes-optimal under the true joint distribution and same fixed order; never observes future signals or hidden correctness.',
            'cases':cases,'grid_validation':grid_validation,'finite_calibration_learning':learning,
            'limits':'Constructed scenarios validate logic and illustrate failure modes; they do not establish empirical GPQA benefit or paper-wide guarantees.'}
