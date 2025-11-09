import torch
import copy
import math
import functools as fn
import torch.nn.functional as F
from collections import defaultdict


from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.distributions.binomial import Binomial

from triflow.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)


from triflow.utils.se3 import utils as du
from triflow.utils.rigid_utils import Rigid
from triflow.mpnn_data.data_transforms import make_one_hot
from triflow.utils.loss import scale_trans

def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    
    return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)


def _masked_categorical(num_batch, num_res, device):
    return torch.ones(
        num_batch, num_res, device=device) * du.MASK_TOKEN_INDEX


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._aatypes_cfg = cfg.aatypes
        self._sample_cfg = cfg.sampling

        self.num_tokens = 21 if self._aatypes_cfg.interpolant_type == "masking" else 20

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        if self._aatypes_cfg.interpolant_type == "masking":
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t) # (B, N)
            aatypes_t[corruption_mask] = du.MASK_TOKEN_INDEX
            aatypes_t = aatypes_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)

        else:
            raise ValueError(f"Unknown aatypes interpolant type {self._aatypes_cfg.interpolant_type}")

        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)
    
    

    def corrupt_rand_cond_batch(self, batch):
            #this is for base triflow and triflow trained on pmpnn dataset
            #50% of training time corrupt a portion of the sequence and 50% corrupt the entire sequence

            noisy_batch = copy.deepcopy(batch)
            aatypes_1 = batch['aatype'][...,-1]

            # [B, N]
            res_mask = batch['seq_mask'][...,-1]                                
            diffuse_mask = batch['seq_mask'][...,-1].clone()
            if torch.rand(1, device = self._device) < 0.5:
                mask_percent = torch.rand(1, device = self._device)
                ones = diffuse_mask.nonzero(as_tuple=False)
                num_flip = max(1, int(mask_percent * ones.size(0)))
                flip_idx = ones[torch.randperm(ones.size(0))[:num_flip]]
                diffuse_mask[flip_idx[:, 0], flip_idx[:, 1]] = 0  
                
            num_batch, num_res = diffuse_mask.shape
            
            # [B, 1]
            t = self.sample_t(num_batch)[:, None]
            cat_t = t        
            noisy_batch['cat_t'] = cat_t

            if self._aatypes_cfg.corrupt:
                aatypes_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask, diffuse_mask)
            else:
                aatypes_t = aatypes_1
            noisy_batch['aatypes_t'] = aatypes_t[...,None] #temp add back recycling dim
            noisy_batch["target_feat_t"] = make_one_hot(aatypes_t.to(torch.int64), 21)[...,None] #temp add back recycling dim

            noisy_batch['aatypes_sc'] = torch.zeros_like(
                aatypes_1)[..., None].repeat(1, 1, self.num_tokens)[...,None] #temp add back recycling dim                    
            noisy_batch["diffuse_mask"] = diffuse_mask[...,None]
            
            return noisy_batch



    def _regularize_step_probs(self, step_probs, aatypes_t):
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        # TODO replace with torch._scatter
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs


    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t, temp=0.1):
        # S = 21
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 21
        device = logits_1.device
        
        mask_one_hot = torch.zeros((S,), device=device)
        mask_one_hot[du.MASK_TOKEN_INDEX] = 1.0

        logits_1[:, :, du.MASK_TOKEN_INDEX] = -1e9    
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1) # (B, D, S)
        

        aatypes_t_is_mask = (aatypes_t == du.MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
        step_probs = d_t * pt_x1_probs * ((1+ self._aatypes_cfg.noise*t) / ((1 - t))) # (B, D, S)
        step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * self._aatypes_cfg.noise        
        step_probs = self._regularize_step_probs(step_probs, aatypes_t)

        
        return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)



    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t, temp=0.0000001):
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 21        
        device = logits_1.device

        logits_1_wo_mask = logits_1[:, :, 0:-1] # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / temp, dim=-1) # (B, D, S-1)
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0] # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != du.MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True) # (B, D)

        unmask_probs = (d_t * ( (1 + self._aatypes_cfg.noise * t) / (1-t)).to(device)).clamp(max=1) # scalar

        number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == du.MASK_TOKEN_INDEX, dim=-1).float(),
                                          prob=unmask_probs)
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S-1), num_samples=1).view(batch_size, num_res)

        # Vectorized version of:
        # for b in range(B):
        #     for d in range(D):
        #         if d < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprobs_idcs[b, d]] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1-mask1) * inital_val_max_logprob_idcs).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=device))
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask_mask = (u < d_t * self._aatypes_cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + du.MASK_TOKEN_INDEX * re_mask_mask
        

        return aatypes_t
    

    def score(self, batch, model, frames, aa_init = None, temp = None):

        num_batch = batch["all_atom_positions"].shape[0]
        num_res= batch["all_atom_positions"].shape[1]        
        diffuse_mask = batch["diffuse_mask"][...,-1]
        
        
        if aa_init is None:
            aatypes_0 = _masked_categorical(num_batch, num_res, self._device)

        else:
            aatypes_0 = aa_init.clone()

        aatypes_1 = torch.zeros((num_batch, num_res), device = self._device).long()
        logits_1 = torch.nn.functional.one_hot(
            aatypes_1,
            num_classes=self.num_tokens
        ).float()

        num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]      
        
        aatypes_t_1 = aatypes_0
        clean_traj = []
        prot_traj = []

        #cache the pair representation
        z = model.run_embedder(batch)   

        for t_2 in ts[1:]:
            batch['aatypes_t'] = aatypes_t_1

            t = torch.ones((num_batch, 1), device = self._device) * t_1
            batch['cat_t'] = t
            target_feat_t = make_one_hot(batch["aatypes_t"].to(torch.int64), 21)
            d_t = t_2 - t_1
            with torch.no_grad():         

                # model_out = model.run_blocks(batch, z, rigid_frames=frames, seq=target_feat_t, temp=self._aatypes_cfg.temp)
                model_out = model.run_blocks(batch, z, rigid_frames=frames, seq=target_feat_t, temp=temp)

            pred_aatypes_1 = model_out['pred_aatypes']
            pred_logits_1 = model_out['pred_logits']
            clean_traj.append(pred_aatypes_1.detach().cpu())
            
            
            aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1, temp)
            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)

            aatypes_t_1 = aatypes_t_2
            prot_traj.append( aatypes_t_2.cpu().detach())

            t_1 = t_2

        #we sampled until min need to do one more step

        t_1 = ts[-1]
        batch['aatypes_t'] = aatypes_t_1
        
        with torch.no_grad():
            target_feat_t = make_one_hot(batch["aatypes_t"].to(torch.int64), 21) #temp add back recycling dim            
            model_out = model.run_blocks(batch,z, rigid_frames=frames, seq=target_feat_t, temp=temp)

        pred_aatypes_1 = model_out["pred_aatypes"]
        pred_aatypes_1 = _aatypes_diffuse_mask(pred_aatypes_1, aatypes_1, diffuse_mask).to(torch.int64) #only update the positions with diffuse_mask
        clean_traj.append(pred_aatypes_1.detach().cpu())
        prot_traj.append(pred_aatypes_1.detach().cpu())

        return prot_traj, clean_traj


    def aa_sample(self, batch, model, frames, aa_init = None, temp = None, omit_AA=None, tied_weights=False, sample_purity=False, sample_priority=False, run_cfg=True):

        num_batch = batch["all_atom_positions"].shape[0]
        num_res= batch["all_atom_positions"].shape[1]        
        diffuse_mask = batch["diffuse_mask"][...,-1]
        dtype = next(model.parameters()).dtype
        
        if aa_init is None:
            aatypes_0 = _masked_categorical(num_batch, num_res, self._device)
            aatypes_1 = torch.zeros((num_batch, num_res), device = self._device).long()

        else:
            aatypes_0 = aa_init.clone()
            aatypes_1 = aa_init.clone()

        # aatypes_1 = torch.zeros((num_batch, num_res), device = self._device).long()
        logits_1 = torch.nn.functional.one_hot(
            aatypes_1,
            num_classes=self.num_tokens
        ).float()

        num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]      
        
        aatypes_t_1 = aatypes_0        
        prot_traj = []

        #cache the pair representation
        
        
        z = model.run_embedder(batch)   

        
        conf = 0

        for t_2 in ts[1:]:
            batch['aatypes_t'] = aatypes_t_1        
                        
            t = torch.ones((num_batch, 1), device = self._device) * t_1
            
            
            batch['cat_t'] = t
            target_feat_t = make_one_hot(batch["aatypes_t"].to(torch.int64), 21)
            d_t = t_2 - t_1
            with torch.no_grad():         
                model_out = model.run_blocks(batch, z, rigid_frames=frames, seq=target_feat_t, temp=temp)                
                pred_logits_1 = model_out['pred_logits']

                if run_cfg:                    
                    dummy_aatypes = torch.ones((num_batch, num_res), device = self._device) * 20
                    dummy_target_feat = make_one_hot(dummy_aatypes.to(torch.int64), 21)                                    
                    model_out_uncond = model.run_blocks(batch, z, rigid_frames=frames, seq=dummy_target_feat, temp=temp)                    
                    cfg_weight = 5
                    pred_logits_1 = ((1 + cfg_weight) * model_out['pred_logits']) + (cfg_weight * model_out_uncond["pred_logits"])

            #####

            if tied_weights:
                #tied weights here by averaging the logits at same residue index
                #this needs to be checked again. doesnt fully work yet                
                res_idx = batch["residue_index"][..., -1]  # shape: (B, N)
                unique_chains = torch.unique(batch["asym_id"][..., -1])
                #get the first position of each chain
                first_indices = []
                for val in unique_chains:                                    
                    idx = (batch["asym_id"][...,-1] == val).nonzero(as_tuple=True)[1][0]
                    first_indices.append(idx.item())

                
                length = (batch["asym_id"][...,-1] == unique_chains[0]).int().sum()
                for i in range(length):
                    for j in range(len(first_indices) - 1):
                        count = pred_logits_1[:,i,:] + pred_logits_1[:,first_indices[j+ 1] + i,:]

                    count = count / len(first_indices)         

                    for j in range(len(first_indices)):
                        pred_logits_1[:,first_indices[j],:] = count
                        
            ####
            if sample_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)    

            else:
                if sample_priority:
                    priority = torch.sort(torch.mean(batch["avg_distances"][..., -1], dim = -1))[1]                            
                    aatypes_t_2, conf_pred = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1, temp, return_confidence=True, omit_indices=omit_AA, priority=priority)    
                    conf = conf + conf_pred

                else:
                    aatypes_t_2, conf_pred = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1, temp, return_confidence=True, omit_indices=omit_AA)    
                    conf = conf + conf_pred

            
            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
            
            aatypes_t_1 = aatypes_t_2
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        #we sampled until min need to do one more step

        t_1 = ts[-1]
        batch['aatypes_t'] = aatypes_t_1
        
        with torch.no_grad():
            target_feat_t = make_one_hot(batch["aatypes_t"].to(torch.int64), 21) #temp add back recycling dim            
            model_out = model.run_blocks(batch,z, rigid_frames=frames, seq=target_feat_t, temp=temp)            

        
        pred_logits_1 = model_out["pred_logits"]
        probabilities = torch.nn.functional.softmax(pred_logits_1 / temp, dim=-1)
        pred_aatypes_1 = torch.multinomial(probabilities[0], num_samples=1).squeeze(dim=-1)[None]
        pred_aatypes_1 = _aatypes_diffuse_mask(pred_aatypes_1, aatypes_1, diffuse_mask).to(torch.int64) #only update the positions with diffuse_mask
        
        prot_traj.append(pred_aatypes_1.detach().cpu())        

        conf = torch.exp(-conf / num_res)
        # conf = torch.exp(-torch.tensor(conf, device=self._device) / num_res)
        # conf = -conf / (torch.log(torch.tensor(20, device=self._device)) * num_res)
        conf = round(conf.item(), 2)        
        
        return prot_traj, conf
### 


    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t, temp=0.1, return_confidence=False, omit_indices=None, priority=None):
        # S is the total number of tokens (should be 21)
        batch_size, num_res, S = logits_1.shape
        device = logits_1.device

        mask_one_hot = torch.zeros((S,), device=device)
        mask_one_hot[du.MASK_TOKEN_INDEX] = 1.0

        # Force the mask token to never be sampled
        logits_1[:, :, du.MASK_TOKEN_INDEX] = -1e9  

        # If provided, set logits for omitted residues to a very low value
        if omit_indices is not None:
            for idx in omit_indices:
                logits_1[:, :, idx] = -1e9

        # Compute probabilities from the modified logits
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

        if priority is not None:
            num_unmask = 0.1 * num_res
            x1 = torch.multinomial(pt_x1_probs.view(-1,S), num_samples = 1).view(batch_size, num_res).to(torch.float32)
            mask = (aatypes_t == du.MASK_TOKEN_INDEX).float()
            
            remove = torch.where(mask == 0)[1]

            if remove.numel() == 0:
                # When no indices are removed, use a default behavior (e.g. use all indices from priority)
                priority_filtered = priority[:,:int(num_unmask)]
            else:
                
                keep = ~torch.isin(priority[0], remove)
                priority_filtered = priority[0][keep][:int(num_unmask)][None]           

            final_samples = aatypes_t.clone().to(torch.float32)
            final_samples[0,priority_filtered[0,:]] = x1[0,priority_filtered[0,:]]
            
            return final_samples, torch.tensor(0.0, device=self._device)
            
            
        aatypes_t_is_mask = (aatypes_t == du.MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
        step_probs = d_t * pt_x1_probs * ((1 + self._aatypes_cfg.noise * t) / (1 - t))
        step_probs += d_t * (1 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * self._aatypes_cfg.noise

        if omit_indices is not None:
            for idx in omit_indices:
                step_probs[..., idx] = 0.0

        
        step_probs = self._regularize_step_probs(step_probs, aatypes_t)
        
        
        samples = torch.multinomial(step_probs.view(-1, S), num_samples=1).view(batch_size, num_res)
        

        if return_confidence:
            log_probs = F.log_softmax(logits_1, dim=-1)
            selected_log_probs = log_probs.gather(dim=-1, index=samples.unsqueeze(-1)).squeeze(-1)
            diff_mask = (aatypes_t != samples).float()  # Positions that were updated
            conf = torch.sum((selected_log_probs * diff_mask) / (1 - t), dim=1)
            return samples, conf


        return samples    
    


    def partial_flows(self, batch, model, frames, aa_init = None, temp = None, omit_AA=None, tied_weights=False, sample_purity=False, sample_priority=False, run_cfg=False, t=0.5):

        num_batch = batch["all_atom_positions"].shape[0]
        num_res= batch["all_atom_positions"].shape[1]        
        diffuse_mask = batch["diffuse_mask"][...,-1]
                
        #run the forward diffusion process
        cat_t = torch.tensor([t], device=self._device)[:,None]        
        aatypes_true = batch['aatype'][...,-1]
        res_mask = batch['seq_mask'][...,-1]
        diffuse_mask = batch['seq_mask'][...,-1]
        
        aatypes_t_1 =  self._corrupt_aatypes(aatypes_true, cat_t, res_mask, diffuse_mask)
        aatypes_1 = aatypes_t_1.clone()
    
        num_timesteps = self._sample_cfg.num_timesteps    
        ts = torch.linspace(self._cfg.min_t, t, num_timesteps)
        t_1 = ts[0]      
                
        prot_traj = []

        #cache the pair representation
        z = model.run_embedder(batch)   
        conf = 0

        for t_2 in ts[1:]:
            batch['aatypes_t'] = aatypes_t_1        
                        
            t = torch.ones((num_batch, 1), device = self._device) * t_1
            
            
            batch['cat_t'] = t
            target_feat_t = make_one_hot(batch["aatypes_t"].to(torch.int64), 21)
            d_t = t_2 - t_1
            with torch.no_grad():         
                model_out = model.run_blocks(batch, z, rigid_frames=frames, seq=target_feat_t, temp=temp)                
                pred_logits_1 = model_out['pred_logits']

                if run_cfg:                    
                    dummy_aatypes = torch.ones((num_batch, num_res), device = self._device) * 20
                    dummy_target_feat = make_one_hot(dummy_aatypes.to(torch.int64), 21)                                    
                    model_out_uncond = model.run_blocks(batch, z, rigid_frames=frames, seq=dummy_target_feat, temp=temp)                    
                    cfg_weight = 5
                    pred_logits_1 = ((1 + cfg_weight) * model_out['pred_logits']) + (cfg_weight * model_out_uncond["pred_logits"])

            #####

            if tied_weights:
                #tied weights here by averaging the logits at same residue index
                #this needs to be checked again. doesnt fully work yet                
                res_idx = batch["residue_index"][..., -1]  # shape: (B, N)
                unique_chains = torch.unique(batch["asym_id"][..., -1])
                #get the first position of each chain
                first_indices = []
                for val in unique_chains:                                    
                    idx = (batch["asym_id"][...,-1] == val).nonzero(as_tuple=True)[1][0]
                    first_indices.append(idx.item())

                
                length = (batch["asym_id"][...,-1] == unique_chains[0]).int().sum()
                for i in range(length):
                    for j in range(len(first_indices) - 1):
                        count = pred_logits_1[:,i,:] + pred_logits_1[:,first_indices[j+ 1] + i,:]

                    count = count / len(first_indices)         

                    for j in range(len(first_indices)):
                        pred_logits_1[:,first_indices[j],:] = count
                        
            ####
            if sample_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_logits_1, aatypes_t_1)    

            else:
                if sample_priority:
                    priority = torch.sort(torch.mean(batch["avg_distances"][..., -1], dim = -1))[1]                            
                    aatypes_t_2, conf_pred = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1, temp, return_confidence=True, omit_indices=omit_AA, priority=priority)    
                    conf = conf + conf_pred

                else:
                    aatypes_t_2, conf_pred = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1, temp, return_confidence=True, omit_indices=omit_AA)    
                    conf = conf + conf_pred


            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)
            
            aatypes_t_1 = aatypes_t_2
            prot_traj.append(aatypes_t_2.cpu().detach())

            t_1 = t_2

        #we sampled until min need to do one more step

        t_1 = ts[-1]
        batch['aatypes_t'] = aatypes_t_1
        
        with torch.no_grad():
            target_feat_t = make_one_hot(batch["aatypes_t"].to(torch.int64), 21) #temp add back recycling dim            
            model_out = model.run_blocks(batch,z, rigid_frames=frames, seq=target_feat_t, temp=temp)            

        
        pred_logits_1 = model_out["pred_logits"]
        probabilities = torch.nn.functional.softmax(pred_logits_1 / temp, dim=-1)
        pred_aatypes_1 = torch.multinomial(probabilities[0], num_samples=1).squeeze(dim=-1)[None]
        pred_aatypes_1 = _aatypes_diffuse_mask(pred_aatypes_1, aatypes_1, diffuse_mask).to(torch.int64) #only update the positions with diffuse_mask
        
        prot_traj.append(pred_aatypes_1.detach().cpu())        

        conf = torch.exp(-conf / num_res)
        # conf = torch.exp(-torch.tensor(conf, device=self._device) / num_res)
        # conf = -conf / (torch.log(torch.tensor(20, device=self._device)) * num_res)
        conf = round(conf.item(), 2)        
        
        return prot_traj, conf




