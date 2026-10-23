import unittest
import numpy as np
import pandas as pd
from rcf.construct import minute_prices,causal_lookup
from rcf.numerics import project_batch,weights

class CoreTests(unittest.TestCase):
    def test_observation_availability(self):
        fills=pd.DataFrame({'market_id':['a','a','a'],'timestamp':[61,119,181],'price':[.1,.4,.6]})
        h=minute_prices(fills)
        v,ok=causal_lookup(h,np.array([119,120,180,239,240]),60)
        self.assertEqual(ok.tolist(),[False,True,False,False,True])
        np.testing.assert_allclose(v[ok],[.4,.6])
    def test_shared_variable_joint_constraints(self):
        # A=>B and B=>C. Enforcing only A<=B may leave B>C.
        a=np.array([[1.,-1.,0.],[0.,1.,-1.]])
        b=np.zeros(2);x=np.array([[.9,.1,.05]])
        partial=project_batch(x,a[:1],b[:1]);joint=project_batch(x,a,b)
        self.assertGreater((partial@a.T-b)[0,1],.1)
        self.assertLessEqual(np.max(joint@a.T-b),1e-8)
        np.testing.assert_allclose(joint,[[.35,.35,.35]],atol=1e-8)
    def test_original_hierarchical_weights(self):
        f=pd.DataFrame({'component':['a','a','a','b'],'vid':[1,1,2,3],'market_id':['x','y','z','w']})
        np.testing.assert_allclose(weights(f),[.125,.125,.25,.5])

if __name__=='__main__':unittest.main()
