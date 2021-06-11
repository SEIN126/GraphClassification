# 🖥DeepLearning Final Project - G9⭐
## Task : Graph Classification

### Contributor

[Junjyeon Kim](https://github.com/Jungyeonkim114)    
[Seungbum Lee](https://github.com/Seungbeom-Lee)    
[Chanyang Seo](https://github.com/chan8616)   
[Sein Park](https://github.com/SEIN126)   

<hr>

#### 1. Dir tree
<pre>
<code>
- GraphClassificatoin
    |- models
        |- GFN
        |- GNN(GCN, GAT, Graphsage) 
    |- data
        |- nx6.pt
        |- nx6_test.pt
        |- nx9.pt
        |- nx9_test.pt
        |- graph
        |- graph_ind
        |- train
        |- test
        |- test_sample
    |- main.py
    |- util.py
    |- ckpt
        |- best_GCN_model.pt
        |- best_GFN1_model.pt
        |- best_GFN2_model.pt
        |- best_GAT_model.pt
        |- best_GraphSage_model.pt
    |- practice
        |- toy_gcn.ipynb
        |- plotting graphs.ipynb
</code>
</pre>

#### 2. implementing
- If you want to just make the test file using our model weight file(.pt) 
```bash
python main.py
```
```bash
❗❗❗caution❗❗❗
- Because of our training dataset size(bigger than 300MB..), we can not commit ours.
- We only commit test dataset(nx9.pt , nx6.pt) in directory ./data. 
- So you can only 'test' by using our weight file(.pt) in directory ./ckpt
- If you want to train our model, please issue me on above tap. I will send you our training sets.    
```
- Else, you want to select mode   
```bash
python main.py --help
```
```bash
  -h, --help         show this help message and exit
  --epochs EPOCHS    epochs
  --train TRAIN      [True or False] train or not
  --save_pt SAVE_PT  [True or False] save weight file or not
  --save_fg SAVE_FG  [True or False] save val_acc, loss graph or not
  --printed PRINTED  [True or False] print test file or not
```

