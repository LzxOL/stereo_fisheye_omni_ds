"""小尺寸独立Fusion测试；不读数据、不执行训练流程、不写结果文件。

运行：python tools/smoke_geometry_aware_fusion.py --device cpu
Warp当前无连续Quality输出，测试明确构造独立连续可信度，绝不从Mask伪造。
"""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module_cnn_only.geometry_aware_fusion import GeometryAwareFeatureFusion


def fixture(height, width, device):
    shape = (2, 8, 5, height, width)
    ms = torch.rand(2,1,5,height,width,device=device) > .3
    mc = torch.rand_like(ms.float()) > .3
    # 所有Batch和D都包含self-only、cross-only、neither、both四种情况。
    ms[...,0,:4] = torch.tensor([True,False,False,True],device=device)
    mc[...,0,:4] = torch.tensor([False,True,False,True],device=device)
    own = torch.randn(shape,device=device).masked_fill(~ms,float('nan')).requires_grad_()
    cross = torch.randn(shape,device=device).masked_fill(~mc,float('inf')).requires_grad_()
    quality = (.1+.85*torch.rand_like(ms.float())).masked_fill(~mc,float('nan'))
    return dict(own=own,cross=cross,ms=ms,mc=mc,quality=quality)


def single(model, data, quality=None):
    return model(data['own'],data['cross'],data['ms'],data['mc'],
                 data['quality'] if quality is None else quality)


def check(output, data):
    cost,valid,weights = output
    ms,mc = data['ms'],data['mc']
    assert cost.shape == data['own'].shape
    assert weights.shape == (cost.shape[0],2,*cost.shape[2:])
    assert torch.isfinite(cost).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(valid,ms|mc)
    torch.testing.assert_close(weights.sum(1,keepdim=True),valid.float(),rtol=0,atol=1e-6)
    assert not weights[:,0:1][~ms].any() and not weights[:,1:2][~mc].any()
    own_only=(ms&~mc).expand_as(cost); cross_only=(mc&~ms).expand_as(cost)
    torch.testing.assert_close(cost[own_only],data['own'][own_only],rtol=0,atol=0)
    torch.testing.assert_close(cost[cross_only],data['cross'][cross_only],rtol=0,atol=0)
    assert not cost.masked_select(~valid.expand_as(cost)).any()


def quality_check(model, data):
    high = torch.full_like(data['quality'],.9)
    low = torch.full_like(data['quality'],.1)
    _,_,wh = single(model,data,high)
    _,_,wl = single(model,data,low)
    both=data['ms']&data['mc']
    assert (wl[:,1:2][both] < wh[:,1:2][both]).all()
    # 质量不是可见性：即便Q=0，唯一有效的cross仍得到1的权重。
    zero = single(model,data,torch.zeros_like(high))
    check(zero,data)
    return float(wh[:,1:2][both].detach().mean()),float(wl[:,1:2][both].detach().mean())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    device=torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(20261116)
    model=GeometryAwareFeatureFusion().to(device)
    yin,yang=fixture(7,11,device),fixture(5,9,device)
    # 键名与现有Warp真实返回值完全一致；这里只模拟已对齐的Warp结果。
    warped=dict(yang_to_yin=yin['cross'],yang_to_yin_valid=yin['mc'],
                yin_to_yang=yang['cross'],yin_to_yang_valid=yang['mc'])
    calls=[]
    hook=model.network[0].register_forward_pre_hook(lambda m,args: calls.append((id(m),args[0].detach().clone())))
    outputs=model.forward_bidirectional(yin['own'],yang['own'],yin['ms'],yang['ms'],
        warped,yin['quality'],yang['quality'])
    hook.remove()
    assert len(calls)==2 and calls[0][0]==calls[1][0]
    for target,data,(_,features) in zip(['yin','yang'],[yin,yang],calls):
        check((outputs['fused_'+target],outputs[target+'_valid'],outputs[target+'_weights']),data)
        assert features.shape[1]==35
        both=data['ms']&data['mc']
        assert not features[:,16:24].masked_select(~both.expand(-1,8,-1,-1,-1)).any()
        assert not features[:,24:32].masked_select(~both.expand(-1,8,-1,-1,-1)).any()
        clean_s=data['own'].detach().masked_fill(~data['ms'],0.)
        clean_c=data['cross'].detach().masked_fill(~data['mc'],0.)
        torch.testing.assert_close(features[:,16:24],(clean_s*clean_c).masked_fill(~both,0.))
        torch.testing.assert_close(features[:,24:32],(clean_s-clean_c).abs().masked_fill(~both,0.))
        weights=outputs[target+'_weights']
        assert (weights[:,0:1][both] > weights[:,1:2][both]).all()
    initial_quality=[quality_check(model,d) for d in (yin,yang)]

    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    (outputs['fused_yin'].square().mean()+outputs['fused_yang'].square().mean()).backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    for data in (yin,yang):
        data['own'].grad=None;data['cross'].grad=None
    after=[single(model,d) for d in (yin,yang)]
    for out,data in zip(after,(yin,yang)):check(out,data)
    sum(out[0].square().mean() for out in after).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    for data in (yin,yang):
        for name,mask in [('own',data['ms']),('cross',data['mc'])]:
            grad=data[name].grad
            assert torch.isfinite(grad).all() and grad.abs().sum()>0
            assert not grad.masked_select(~mask.expand_as(grad)).any()
    learned_quality=[quality_check(model,d) for d in (yin,yang)]

    # 全无效：输入含NaN，softmax不产生NaN，输出/权重/梯度均为零。
    model.zero_grad(set_to_none=True)
    empty=torch.full((1,8,3,4,5),float('nan'),device=device,requires_grad=True)
    other=torch.full_like(empty,float('nan'),requires_grad=True)
    mask=torch.zeros(1,1,3,4,5,dtype=torch.bool,device=device)
    quality=torch.full_like(mask.float(),float('nan'))
    value,valid,weights=model(empty,other,mask,mask,quality)
    assert all(torch.isfinite(x).all() and not x.any() for x in (value,valid,weights))
    value.sum().backward()
    assert torch.isfinite(empty.grad).all() and not empty.grad.any()
    assert torch.isfinite(other.grad).all() and not other.grad.any()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any() for p in model.parameters())
    # 从头训练可能产生较大的有限GWC；乘积超过65504时，FP16 Conv会溢出。
    # 回归测试使用真实高动态范围输入，不以nan_to_num或裁剪掩盖问题。
    strong_model=GeometryAwareFeatureFusion().to(device)
    dtype=torch.float16 if device.type=='cuda' else torch.bfloat16
    own=torch.full((1,8,4,8,8),512.,device=device,dtype=dtype,requires_grad=True)
    cross=torch.full_like(own,384.,requires_grad=True)
    support=torch.ones(1,1,4,8,8,dtype=torch.bool,device=device)
    with torch.autocast(device.type,dtype=dtype):
        value,valid,weights=strong_model(own,cross,support,support,support.float()*.8)
    assert torch.isfinite(value).all() and torch.isfinite(weights).all()
    expected=(512*2+384*.8)/(2+.8)
    torch.testing.assert_close(value.float(),torch.full_like(value.float(),expected),rtol=0,atol=2.)
    value.float().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in strong_model.parameters())
    assert torch.isfinite(own.grad).all() and torch.isfinite(cross.grad).all()
    print(json.dumps(dict(status='PASS',device=str(device),
        output_shapes={key:list(v.shape) for key,v in outputs.items()},shared_network_calls=len(calls),
        four_mask_cases=True,weights_sum_to_one_when_valid=True,all_invalid_safe=True,
        consistency_difference_both_valid_only=True,initial_self_preference=True,
        cross_weight_high_low_quality_at_init=initial_quality,
        cross_weight_high_low_quality_after_update=learned_quality,
        all_parameters_and_costs_receive_finite_gradients=True,
        high_dynamic_range_amp_forward_backward=True,
        trainable_parameters=sum(p.numel() for p in model.parameters())),indent=2))


if __name__=='__main__':main()
