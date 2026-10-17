from pathlib import Path
import json,sys,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments_v6 import model,make_record,dump
from sparse_clock import SparseClockOracle
import shared_fields,shared_node_support
shared_fields.ClockOracle=SparseClockOracle;shared_node_support.ClockOracle=SparseClockOracle
z=dict(np.load(ROOT/'data/nonlinear_test/test_83000.npz'));cfg=json.loads((ROOT/'data/nonlinear_frozen_budget.json').read_text());rec,_=make_record(z,183000,True)
W=np.array([[2.,2.5],[2.25,2.75]]);w=np.array([[-1.,0.],[1.,0.]])
a=model(z,rec,W,cfg)[0].interval(w)[0];corrupt=dict(z)
for key in ['F','E','N','roll','roll_rate','vy']:
 corrupt[key]=np.full_like(z[key],1e8)
b=model(corrupt,rec,W,cfg)[0].interval(w)[0]
assert np.array_equal(a,b)
P=json.loads((ROOT/'reproducibility/protocol_v6.json').read_text());assert not(set(P['development_seeds']) & set(range(P['test_seed_start'],P['test_seed_start']+P['test_runs'])))
dump(ROOT/'results/input_contract_check.json',{'passed':True,'force_and_auxiliary_label_mutation_max_interval_change':float(max(abs(a-b))),'disjoint_development_test_seeds':True,'scope':'single explicit mutation regression plus documented inference argument separation; not a general security proof'})
print('INPUT CONTRACT PASS')
