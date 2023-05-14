# ML10M Utils
user,item = list(set(duser)), list(set(ditem)) # 项目去除重复值
print('user,item',len(user),len(item))
lens = len(duser)
# 建立索引
duserid_path = os.path.join('./uvinit/',args.dataset,'duserid.npy')
ditemid_path = os.path.join('./uvinit/',args.dataset,'ditemid.npy')
if os.access(duserid_path,os.F_OK) and os.access(ditemid_path,os.F_OK):
    duserid = np.load(duserid_path)
    ditemid = np.load(ditemid_path)
else:
    duserid,ditemid = [0] * lens, [0] * lens
    for i in range(lens):
        if(i%10000==0):
            print('进度: ',i,'/',lens)
        duserid[i] =  user.index(duser[i]) # 稠密矩阵项目列映射为其index，第1234个duserid内容元素指向去重后索引
        ditemid[i] =  item.index(ditem[i])
    np.save(duserid_path,duserid) # 保存数据，末尾可自动添加.npy
    np.save(ditemid_path,ditemid)

uv0_list = torch.load("uvinit/ml10m/em.pt",map_location=dev)
u0,v0= torch.tensor(uv0_list['u.weight']), torch.tensor(uv0_list['v.weight'])
alluv0 = torch.cat([u0, v0.add(u0.size(0))], dim=0)

print("u0.shape,v0.shape",u0.shape,v0.shape)

# 分批训练
duserid = torch.tensor(duserid, dtype=torch.long)
ditemid = torch.tensor(ditemid, dtype=torch.long)
drate =  torch.tensor(drate,dtype=torch.float)
torch_dataset = data.TensorDataset(duserid,ditemid,drate)
loader = data.DataLoader(dataset=torch_dataset,batch_size=10000,shuffle=False,num_workers=2)
for step, (suserid,sitemid,srate) in enumerate(loader):
    print('========== 批次',step," ===========")
    csruid = suserid.numpy()
    csriid = sitemid.numpy()
    csrrate = srate.numpy()
    Rmat = csr_matrix((csrrate, (csruid, csriid))) # csr稠密矩阵
    rsp = Rmat.todense()
    G = torch.from_numpy(rsp)# 稀疏矩阵，行为用户，列为项目
    Gmean = torch.mean(G.float())
    ulen,vlen = G.shape[0],len(item)
    print('len(ditem), len(item),Rmat.size,G.shape:\n', len(ditem), len(item),Rmat.size, G.shape)

    print('同构矩阵nx图')
    G = G.float()
    M,N = torch.mm(G, G.t()),torch.mm(G.t(),G)
    graphm = nx.from_numpy_array(M.numpy())
    graphn = nx.from_numpy_array(N.numpy())
    # print("graphm",graphm,"graphn",graphn)
    neiberm = NeibSampler(graphm,args.nbsz).sample().to(dev)  # 邻居采样
    neibern = NeibSampler(graphn,args.nbsz).sample().to(dev)
    # neib_savm = torch.zeros_like(neiberm) # 储存邻居
    # 向量采样
    print("suserid",suserid)
    u0select = torch.unique(suserid, dim=0).to(dev)
    v0select = torch.unique(sitemid, dim=0).to(dev)
    u0sub = torch.index_select(u0, 0, u0select)
    v0sub = torch.index_select(v0, 0, v0select)
    print("u0sub",u0sub.shape)
