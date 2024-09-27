### INTRO

This rep is hold by MoriiHuang, which is basically based on nnUNet, nnsam and MedSAM.

This is sjtu_eiee_lab426's final solution to the MH-SEG Challenge.

The core contribution is to add TransUNet, SwinsUNet and other basic networks under the framework of nnUNet, and introduce the pipeline of semi-supervised schemes UAMT,AugSeg and iMAS.

### ENV SETTING
```shell
python -m pip install --user -U pip && python -m pip install --user pip-tools

python -m piptools sync requirements.txt
 
cd /intensity-normalization && ls -l && python setup.py install --user 

RUN cd /model/nnUNet && ls -l && pip install -e .

RUN cd /model/MedSAM && ls -l && pip install -e .

RUN cd /model/MobileSAM && ls -l && pip install -e .
```

### TRAINING

- you can train your own model the way in nnUNet/nnsam/MedSAM

- for nnUNet-based Model you can switch model by reset Model NAME like this:

```python
export MODEL_NAME=nnsam
export MODEL_NAME=nntrans
export MODEL_NAME=nnswins
```

- for semi method, we add it in entry_point, you can process like this:

The Data Struct
```
nnUNet_raw
    - imagesTr
    - imagesTs
    - labelsTr
    - imagesUns
```

put unlabeled samples in imagesUns,then run 
```shell
### 数据预处理
nnUNetv2_plan_and_preprocess -d idx  --verify_dataset_integrity  -unsupervised True
### baseline
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold

### UA-MT
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold -tr nnUNetTrainerUAMT -unsupervised True

### AugSeg
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold -tr nnUNetTrainerAugSeg -unsupervised True

### iMAS
CUDA_VISIBLE_DEVICES=0  nnUNetv2_train idx  2d fold -tr nnUNetTraineriMAS -unsupervised True
```

### inference

the model we finally use in chanllenge is nnsam(AugSeg)+nnTrans+MedSAM,make sure you have trained all of them.

After training,modify the path in test.sh to suit your own path, then run test.sh 
