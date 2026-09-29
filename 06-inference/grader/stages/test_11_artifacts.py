import torch
import pytest

def test_portable_training_checkpoint(api,oracle,tmp_path):
    payload=dict(format='toylm-v1',config=vars(oracle.cfg),tokenizer=dict(kind='utf8-byte',bos_id=256,eos_id=257,vocab_size=258),state_dict=oracle.state_dict(),provenance={'seed':101})
    path=tmp_path/'model.pt'; torch.save(payload,path); before=torch.get_rng_state().clone()
    model=api.artifacts.load_export(path)
    torch.testing.assert_close(torch.get_rng_state(),before)
    assert not model.training
    torch.testing.assert_close(model([256,1,2])[0],oracle([256,1,2])[0])
    payload['tokenizer']['bos_id']=0; torch.save(payload,path)
    with pytest.raises(ValueError): api.artifacts.load_export(path)
