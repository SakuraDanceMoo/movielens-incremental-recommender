"""Small behavioral tests for the actual rating model and ablation switches."""
import unittest

import numpy as np
import torch

from transformer_regression import RatingTransformer, FeatureBlock, blend_selection


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        rng=np.random.default_rng(7)
        self.als={'user':rng.normal(size=(12,16)).astype('f')*.2,
                  'item':rng.normal(size=(18,16)).astype('f')*.2,
                  'known_user':np.ones(12,dtype=bool),'known_item':np.ones(18,dtype=bool),
                  'mean':np.array(3.5)}
        self.als['known_item'][17]=False
        self.semantic=rng.normal(size=(18,128)).astype('f')
        self.meta=rng.normal(size=(18,21)).astype('f')

    def model(self,variant):
        result=RatingTransformer(self.als,self.semantic,self.meta,variant).eval()
        if variant!='mf_only':
            with torch.no_grad():result.head[-1].weight.normal_(std=.1)
        return result

    def test_no_semantics_really_ignores_encoder_values(self):
        model=self.model('no_semantics')
        u,i=torch.tensor([1,2,3]),torch.tensor([2,4,6])
        before=model(u,i)[0].detach()
        model.semantics.mul_(17)
        self.assertTrue(torch.equal(before,model(u,i)[0]))

    def test_full_model_uses_semantic_values(self):
        model=self.model('full')
        u,i=torch.tensor([1,2,3]),torch.tensor([2,4,6])
        before=model(u,i)[0].detach()
        model.semantics.mul_(-3)
        self.assertGreater(float((before-model(u,i)[0]).abs().max().detach()),1e-6)

    def test_missing_factor_has_mean_fallback(self):
        model=RatingTransformer(self.als,self.semantic,self.meta,'mf_only')
        result,_=model(torch.tensor([1]),torch.tensor([17]))
        self.assertAlmostEqual(result.item(),3.5)

    def test_attention_ablation_changes_block(self):
        self.assertIsInstance(self.model('full').block,FeatureBlock)
        self.assertNotIsInstance(self.model('no_attention').block,FeatureBlock)

    def test_fusion_never_worse_than_baseline_on_selection_labels(self):
        y=np.array([1.,3.,5.]);base=np.array([2.,3.,4.]);neural=np.array([4.,1.,2.])
        weight,result=blend_selection(y,base,neural)
        self.assertTrue(0<=weight<=1)
        self.assertLessEqual(np.mean((np.clip(result,.5,5)-y)**2),np.mean((base-y)**2)+1e-12)


if __name__=='__main__':unittest.main()
