# 处理亚马逊数据
import pickle
import pandas as pd
import torch
file_path = 'data/abr371k/All_Beauty.json'
fin = open(file_path, 'r')

df = {}
useless_col = ['helpful','reviewerName','unixReviewTime','summary','reviewTime','verified','vote','image'] # 不想要的字段
i = 0
for line in fin:
    d = eval(line, {"true":True,"false":False,"null":None})
    for s in useless_col:
        if s in d:
            d.pop(s)
    df[i] = d 
    i += 1
df = pd.DataFrame.from_dict(df, orient='index')
df.to_csv('data/abr371k/All_Beauty.csv',index=False)

data = pd.read_csv('data/abr371k/All_Beauty.csv') 
print(data.head())

rid = data['reviewerID'].unique()
ridnum = range(0, len(rid))
ridl = list(ridnum)
rdic = dict(zip(rid, ridl))
# print("值:",rdic['A1PSGLFK1NSVO'])
aid = data['asin'].unique()
aidnum = range(0, len(aid))
aidl = list(aidnum)
adic = dict(zip(aid, aidl))
print("值:",adic['0143026860'])
print('总条数|用户数|项目数',len(data['reviewerID']),len(rid), len(aid))

data['u']=data['reviewerID'].map(rdic)
data['v']=data['asin'].map(adic)
print(data.head())

w = data[['u','v','overall']]
print(w.head())
# w.to_csv('data/abr371k/rating_train.dat',sep='\t',index=False)
w = w.iloc[:10000,:]
print(w.head())
w.to_csv('data/abr371k/rating_test.dat',sep='\t',index=False,header=None)

# 分量相似度计算
dev='cpu'
uv_list = torch.load("uvinit/ml10m/em.pt",map_location=dev)
u,v= torch.tensor(uv_list['u.weight']), torch.tensor(uv_list['v.weight'])
print(u[1])

