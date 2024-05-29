
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from numpy.random import default_rng

# Make example data

def gen_sim_vec(myseed=12345,n=200,p=1000,D=2,rowsize=50, colsize=100, numbers=10,sigma=0.1,nbicluster=4, orthonm=False):
    #rng=default_rng(seed=myseed)
    np.random.seed(seed=myseed)
    data = []
    row_inds = []
    col_inds = []
    
    S = np.diag(np.array([27,22,18,10]))

    for i in range(numbers):
        data_inner = []
        col_ind_in = []
        row_ind_in = np.split(np.random.choice(n,size=rowsize*nbicluster,replace=False),nbicluster)
        row_inds.append(row_ind_in)
        for d in range(D):
            col_ind = np.split(np.random.choice(p,size=colsize*nbicluster,replace=False),nbicluster)
            col_ind_in.append(col_ind)
        col_inds.append(col_ind_in)
        
        U0 = np.zeros((n,nbicluster))
        for j in range(nbicluster):
            U0[row_ind_in[j],j] = np.random.uniform(0.5,1,rowsize)
        if orthonm:
            U0 = np.linalg.qr(U0)[0]
        
        
        V0s = []
        for d in range(D):
            V0 = np.zeros((p,nbicluster))
            for j in range(nbicluster):
                V0[col_ind_in[d][j],j] = np.random.uniform(0.5,1,colsize)
            V0s.append(V0)
        for d in range(D):
            if orthonm:
                V0s[d] = np.linalg.qr(V0s[d])[0]
            data_inner.append(U0 @S @ V0s[d].T + np.random.normal(0,sigma,size=(n,p)))

        data.append(data_inner)

    return data, row_inds, col_inds


# Intermediate step for original data to be used to calculate jaccard index
def gen_tmp(Rows, Cols, true_Rows, true_Cols,n,p,D):
    
    # Rows
    row1 = []
    row2 = []
    
    for k in range(len(true_Rows)):
        tmp2 = np.zeros(n)
        tmp2[true_Rows[k]] = 1
        row2.append(tmp2.reshape(-1,1))
    rowf = np.concatenate(row2,axis=1)
    
    for k in range(len(Rows)):
        tmp1 = np.zeros(n)
        tmp1[Rows[k]] = 1
        row1.append(tmp1.reshape(-1,1))
    rowg = np.concatenate(row1, axis=1)
    
    # Cols
    colg = []
    colf = []
    for d in range(D):
        col1 = []
        col2 = []
        
        for k in range(len(Cols)):
            tmp1 = np.zeros(p)
            tmp1[Cols[k][d]] = 1
            col1.append(tmp1.reshape(-1,1))
        colg.append(np.concatenate(col1,axis=1))
        
        for k in range(len(true_Rows)):
            tmp2 = np.zeros(p)
            tmp2[true_Cols[d][k]] = 1
            col2.append(tmp2.reshape(-1,1))
        colf.append(np.concatenate(col2, axis=1))
    
    return [rowg,colg], [rowf,colf]

# diagnostics function including jaccard index, false pos/neg
def issvd_diagnostics(res1,res2,true_rows,true_cols):
    
    D = len(res1[1])
    k1 = res1[0].shape[1]
    k2 = res2[0].shape[1]
    mats = []
    rel = []
    rev = []
    fscore = []
    fps = []
    fns = []
    fpmats = []
    fnmats = []
    
    for d in range(D):
        fpmat = np.zeros((k1,k2))
        fnmat = np.zeros((k1,k2))
        mat = np.zeros((k1,k2))
        for i in range(k1):
            for j in range(k2):
                A1 = res1[0][:,i].reshape(-1,1)@res1[1][d].T[i,:].reshape(1,-1)
                A = np.where(A1.flatten()>0)[0]
                B1 = res2[0][:,j].reshape(-1,1)@res2[1][d].T[j,:].reshape(1,-1)
                B = np.where(B1.flatten()>0)[0]

                C = set(A).intersection(set(B))
                mat[i,j] = len(C)/(len(A)+len(B)-len(C))
                
                rows = true_rows[j].ravel()
                cols = true_cols[d][j].ravel()
                ele = rows.shape[0]*cols.shape[0]
                fnmat[i,j] = np.sum(A1[np.ix_(rows,cols)]<1) / ele
                A1[np.ix_(rows,cols)] = 0
                fpmat[i,j] = np.sum(A1>0) / ele
        
        fps.append(np.mean(np.min(fpmat,axis=1)))
        fns.append(np.mean(np.min(fnmat,axis=1)))
        mats.append(mat)
        fpmats.append(fpmat)
        fnmats.append(fnmat)
        reld = np.mean(np.max(mat,axis=0))
        revd = np.mean(np.max(mat,axis=1))
        rel.append(reld)
        rev.append(revd)
        fscore.append(2*reld*revd/(reld+revd))
        
    return np.mean(rev), np.mean(rel), np.mean(fscore), np.mean(fps),np.mean(fns)


#main function
def issvd(X,myseed=25,standr=False,pointwise=True,steps=100,size=0.5,vthr = 0.7, ssthr=[0.6,0.8],nbicluster=10,rows_nc=True,cols_nc=True,col_overlap=False,row_overlap=False,pceru=0.1,pcerv=0.1
          ,merr=1e-4,iters=100):
    
    np.random.seed(seed=myseed)
    if not isinstance(X,list):
        print("Input should be a list of numpy arrays!")
    
    D = len(X)
    
    ##Frobeniumns norm standardization in R- overwrite this
    # # Centering and scaling
    # if standr:
    #     scaler = StandardScaler(with_std=True)
    #     X = list(map(lambda x: scaler.fit_transform(x), X))
        
    # Calculate eigenvalues
    egs = []
    for Xd in X:
        U,S,VT = np.linalg.svd(Xd)
        ss = np.cumsum(S)
        sfr = ss / np.sum(S)
        egs.append(np.where(sfr>=vthr)[0][0])
    ks = np.max(egs)+1
    nbicluster = np.min([ks,nbicluster])
    
    stop = False
    
    info = []
    Rows = []
    Cols = []
    abs_objs = []
    rel_objs = []
    ies = []
    
    rel_objs_inner = []
    abs_objs_inner = []
    uw = np.ones(X[0].shape[0]) # weights on u
    vws = list(map(lambda x: np.ones(x.shape[1]), X)) # weights on v
    
    for k in range(nbicluster):
        X0 = np.concatenate(X,axis=1)
        print("Integrative Bicluster {}...".format(k+1))
        n = X0.shape[0]
        ps = list(map(lambda s: s.shape[1], X))
        rowsin = np.repeat(True,n)
        colsins = list(map(lambda s: np.repeat(True,s), ps))
        rows = np.repeat(False,X0.shape[0])
        cols = list(map(lambda s: np.repeat(False,s.shape[1]),X))
        
        # Initialize - concatenated data
        U0,d0,V0T = np.linalg.svd(X0,full_matrices=False)
        u0 = U0[:,0]
        v0 = V0T.T[:,0]
        
        # For later convergence check
        V0cT = [np.linalg.svd(x,full_matrices=False)[2] for x in X]
        v0s = [V.T[:,0].reshape(-1,1) for V in V0cT] # these two lines also means initiate with individual v0 vectors
        
        inner_relobj = []
        inner_absobj = []
        x0s = X
        rels = 1
        
        uddd = []
        vddd = []
        
        if pointwise:
            for i in range(iters):
                uc = updateu_pw(X0,v0,uw,pceru,ssthr,steps,size,rows_nc)
                u1 = uc[0]/np.sqrt(np.sum(uc[0]**2))
                u1[np.isnan(u1)] = 0
                u1[np.isinf(u1)] = 0
                
                vcs = []
                v1s = []
                for j in range(D):
                    vc = updatev_pw(X[j],u0,vws[j],pcerv,ssthr,steps,size,cols_nc)
                    v1 = vc[0]/np.sqrt(np.sum(vc[0]**2))
                    v1[np.isnan(v1)] = 0
                    v1[np.isinf(v1)] = 0
                    v1s.append(v1.reshape(-1,1))
                    vcs.append(vc)
                if uc[2]:
                    print('Rows not stable!')
                    stop = True
                    break
                tmp = np.array(list(map(lambda s: s[2], vcs)))
                if np.any(tmp):
                    print('Cols not stable!')
                    stop = True
                    break
                
                # Convergence
                ud = np.sqrt(np.sum((u0-u1)**2))
                vds = [np.sqrt(np.sum((v1s[s]-v0s[s])**2)) for s in range(D)]
                vd = np.min(vds)
                 
                xrs = [u1.reshape(-1,1)@v1s[d].reshape(1,-1) for d in range(D)]
                xds = [np.linalg.norm((xrs[d]-x0s[d]),'f')**2 for d in range(D)]
                xdd = np.sum(xds)
                inner_absobj.append(xdd)
                
                
                if i > 0:
                    rels = xdd/inner_absobj[i-1]
                    inner_relobj.append(rels)
                
                v0s = v1s
                u0 = u1
                v0 = np.concatenate(v1s,axis=0)
                x0s = xrs
                uddd.append(ud)
                vddd.append(vds)
                
                if np.min([rels,xdd,np.max([ud,vd])]) < merr: #covergence criteria 
                    break
        else:
            for i in range(iters):
                uc = updateu(X0,v0,uw,pceru,ssthr,steps,size,rows_nc)
                u1 = uc[0]/np.sqrt(np.sum(uc[0]**2))
                u1[np.isnan(u1)] = 0
                u1[np.isinf(u1)] = 0
                vcs = []
                v1s = []
                for j in range(D):
                    vc = updatev(X[j],u0,vws[j],pcerv,ssthr,steps,size,cols_nc)
                    v1 = vc[0]/np.sqrt(np.sum(vc[0]**2))
                    v1[np.isnan(v1)] = 0
                    v1[np.isinf(v1)] = 0
                    v1s.append(v1.reshape(-1,1))
                    vcs.append(vc)
                if uc[2]:
                    print('Rows not stable!')
                    stop = True
                    break
                tmp = np.array(list(map(lambda s: s[2], vcs)))
                if np.any(tmp):
                    print('Cols not stable!')
                    stop = True
                    break
                
                # Convergence
                ud = np.sqrt(np.sum((u0-u1)**2))
                vds = [np.sqrt(np.sum((v1s[s]-v0s[s])**2)) for s in range(D)]
                vd = np.min(vds)
                
                xrs = [u1.reshape(-1,1)@v1s[d].reshape(1,-1) for d in range(D)]
                xds = [np.linalg.norm((xrs[d]-x0s[d]),'f') for d in range(D)]
                xdd = np.sum(xds)
                inner_absobj.append(xdd)
                
                
                if i > 0:
                    rels = xdd/inner_absobj[i-1]
                    inner_relobj.append(rels)
                
                v0s = v1s
                u0 = u1
                v0 = np.concatenate(v1s,axis=0)
                x0s = xrs
                uddd.append(ud)
                vddd.append(vds)
                
                if np.min([rels,xdd,np.max([ud,vd])]) < merr: #covergence criteria 
                    break
    
        rel_objs.append(inner_relobj)
        abs_objs.append(inner_absobj)
        
        stableu = uc[1] >= uc[4]
        u0[~stableu] = 0
        stableu_ind = np.where(stableu)[0]
        if not row_overlap:
            uw[stableu_ind] = 1e-6
        stablev_ind = []
        for j in range(D):
            stablev = vcs[j][1]>=vcs[j][4]
            v0s[j][~stablev] = 0
            stablev_ind.append(np.where(stablev)[0])
            if not col_overlap:
                vws[j][stablev] = 1e-6

        # Getting the residual matrix
        ds = [u0.reshape(1,-1)@X[s]@v0s[s].reshape(-1,1) for s in range(D)]
        Xre = [ds[s]*u0.reshape(-1,1)@v0s[s].reshape(1,-1)for s in range(D)]
        
        X = [X[s]-Xre[s] for s in range(D)]
        
        ies.append(i) # iterations
        
        if stop:
            number = k-1
            break
        else:
            number = k
        if i >= iters-1:
            number = k-1
            stop = True
            print('Fail to converge, increase number of iterations! \n')
            break
        if not stop:
            info.append([uc,vc,[u0,v0s]])
            Rows.append(stableu_ind)
            Cols.append(stablev_ind)
        print("\n")

    print("Integrative biclusters detected: {}\n".format(number+1))
    
    if not stop:
        number = k
    if sum(uw!=1)==n:
        print("All samples are clustered!")
        
    for d in range(D):
        if sum(vws[d]!=1)==X[d].shape[1]:
            print("All variables in view {} are clustered!".format(d))
    
    return {'N': number+1, 'Info': info,'Sample_index':Rows, 'Variable_index':Cols, 'Iterations':ies} 


# Update v function - should update for all the subsmaples at once
def updatev(X,u0,vw,pcer,ssthr,steps,size,cols_nc):
    p = X.shape[1]
    err = pcer*p
    ols = X.T @ u0
    stop = False
    lmbdas = np.sort(np.append(np.abs(ols),0))
    lmbdas = lmbdas[::-1]
    qs = np.zeros(lmbdas.shape[0])
    thrall = np.zeros(lmbdas.shape[0])
    ls = lmbdas.shape[0]-1
    
    if cols_nc:
        for l in range(lmbdas.shape[0]):
            tmp = Lasso_nc(X.T,u0,lmbdas[l],vw,steps,size)
            t = tmp != 0
            qs[l] = np.mean(np.sum(t,axis=0))  # sum columns
            sp = np.mean(t,axis=1) # the frequency of variables (row:variables, col:subsamples)
            thrall[l] = ((qs[l]**2/(err*p))+1)/2 # recovering threshold pi
            if thrall[l]>=ssthr[0]:
                ls = l
                break
    else:
        for l in range(lmbdas.shape[0]):
            tmp = Lasso(X.T,u0,lmbdas[l],vw,steps,size)
            t = tmp!=0
            qs[l] = np.mean(np.sum(t,axis=0))
            sp = np.mean(t,axis=1)
            thrall[l] = ((qs[l]**2/(err*p))+1)/2
            if thrall[l] >= ssthr[0]:
                ls=l
                break
    
    thr = thrall[ls]
    if thr >= ssthr[1]:
        while pcer <= 0.5:
            pcer = pcer + 0.01
            thrall = ((qs**2/((pcer*p)*p))+1)/2
            thr = thrall[ls]
            if thr <= ssthr[1]:
                break
    stable = np.where(sp>=thr)[0]
    if stable.shape[0] == 0:
        stop = True
    delta = lmbdas[ls]/np.abs(ols)**0
    vc = np.sign(ols) * (np.abs(ols)>=delta) *(np.abs(ols) - delta)
    return vc, sp, stop, qs, thr, ls, delta


# In[ ]:


def updatev_pw(X,u0,vw,pcer,ssthr,steps,size,cols_nc=False):
    p = X.shape[1]
    err = p*pcer
    ols = X.T @ u0
    stop = False
    lmbdas = np.sort(np.append(np.abs(ols),0))
    lmbdas = lmbdas[::-1]
    qs = np.zeros(lmbdas.shape[0])
    thrall = np.zeros(lmbdas.shape[0])
    ls = lmbdas.shape[0]-1
    l = np.where(lmbdas==np.percentile(lmbdas,50,interpolation='nearest'))[0][0]
    l_min = 0
    l_max = lmbdas.shape[0]-1
    
    if cols_nc:
        for g in range(lmbdas.shape[0]):
            tmp = Lasso_nc(X.T,u0,lmbdas[l],vw,steps,size)
            t = tmp!=0
            qs[l] = np.mean(np.sum(t,axis=0))
            sp = np.mean(t,axis=1)
            thrall[l] = ((qs[l]**2/(err*p))+1)/2
            if thrall[l] >= ssthr[0] and thrall[l] <=ssthr[1]:
                ls = l
                break
            if thrall[l] < ssthr[0]:
                l_min = l
                if l == lmbdas.shape[0]-1:
                    break
                if thrall[l+1] > ssthr[1]:
                    ls = l+1
                    if thrall[l+1] > 0:
                        ls = l
                    tmp = Lasso_nc(X.T,u0,lmbdas[ls],vw,steps,size)
                    t = tmp!=0
                    qs[ls] = np.mean(np.sum(t,axis=0))
                    sp = np.mean(t,axis=1)
                    thrall[ls] = ((qs[ls]**2/(err*p))+1)/2
                    break
                l = int(np.min([lmbdas.shape[0]-1,l_max,l-1+np.ceil((lmbdas.shape[0])/(g+2))]))
                while thrall[l] != 0:
                    l -= 1
                    if l == 0:
                        break
            if thrall[l] > ssthr[1]:
                l_max = l
                if l == 0: 
                    break
                if thrall[l-1] < ssthr[0] and thrall[l-1]!=0:
                    ls = l
                    break
                l = int(np.max([0,l_min,l-np.ceil(lmbdas.shape[0]/(g+2))+1]))
                while thrall[l] !=0:
                    l += 1
                    if l == lmbdas.shape[0]-1:
                        break
    else: 
        for g in range(lmbdas.shape[0]):
            tmp = Lasso(X.T,u0,lmbdas[l],vw,steps,size)
            t = tmp!=0
            qs[l] = np.mean(np.sum(t,axis=0))
            sp = np.mean(t,axis=1)
            thrall[l] = ((qs[l]**2/(err*p))+1)/2
            if thrall[l]>=ssthr[0] and thrall[l]<=ssthr[1]:
                ls=l
                break
            if thrall[l]<ssthr[0]:
                l_min = l
                if l == lmbdas.shape[0]-1:
                    break
                if thrall[l+1] > ssthr[1]:
                    ls = l+1
                    if thrall[l+1]>0:
                        ls = l
                    tmp = Lasso(X.T,u0,lmbdas[ls],vw,steps,size)
                    t = tmp!=0
                    qs[ls] = np.mean(np.sum(t,axis=0))
                    sp = np.mean(t,axis=1)
                    thrall[ls] = ((qs[ls]**2/(err*p))+1)/2
                    break
                l = int(np.min([lmbdas.shape[0]-1,l_max,l-1+np.ceil(lmbdas.shape[0]/(g+2))]))
                while thrall[l] != 0:
                    l -= 1
                    if l==0:
                        break
            if thrall[l]>ssthr[1]:
                l_max = l
                if l == 0:
                    break
                if thrall[l-1] < ssthr[0] and thrall[l-1]!=0:
                    ls = l
                    break
                l = int(np.max([1,l_min,l-np.ceil(lmbdas.shape[0]/(g+2))+1]))
                while thrall[l]!=0:
                    l += 1
                    if l == lmbdas.shape[0]-1:
                        break
    
    thr = ((qs[ls]**2/((pcer*p)*p))+1)/2
    stable = np.where(sp>thr)[0]
    if stable.shape[0] == 0:
        stop = True
    delta = lmbdas[l] / np.abs(ols)**0
    vc = np.sign(ols) * (np.abs(ols)>=delta) * (np.abs(ols)-delta)
    
    return vc, sp, stop, qs, thr, ls, delta


# In[ ]:


# Update u function - should update for all the subsmaples at once
def updateu(X,v0,uw,pcer,ssthr,steps,size,rows_nc):
    n = X.shape[0]
    err = pcer*n
    ols = X @ v0
    stop = False
    lmbdas = np.sort(np.append(np.abs(ols),0))
    lmbdas = lmbdas[::-1] # decreasing
    qs = np.zeros(lmbdas.shape[0])
    thrall = np.zeros(lmbdas.shape[0])
    ls = lmbdas.shape[0]-1
    
    if rows_nc:
        for l in range(lmbdas.shape[0]):
            tmp = Lasso_nc(X,v0,lmbdas[l],uw,steps,size)
            t = tmp != 0
            qs[l] = np.mean(np.sum(t,axis=0))  # sum columns
            sp = np.mean(t,axis=1) # the frequency of variables (row:variables, col:subsamples)
            thrall[l] = ((qs[l]**2/(err*n))+1)/2 # recovering threshold pi
            if thrall[l]>=ssthr[0]:
                ls = l
                break
    else:
        for l in range(lmbdas.shape[0]):
            tmp = Lasso(X,v0,lmbdas[l],uw,steps,size)
            t = tmp!=0
            qs[l] = np.mean(np.sum(t,axis=0))
            sp = np.mean(t,axis=1)
            thrall[l] = ((qs[l]**2/(err*n))+1)/2
            if thrall[l]>=ssthr[0]:
                ls = l
                break
    thr = thrall[ls]
    if thr>=ssthr[1]:
        while pcer<=0.5:
            pcer = pcer + 0.01
            thrall = ((qs**2/((pcer*n)*n))+1)/2
            thr = thrall[ls]
            if thr<=ssthr[1]:
                break
    stable = np.where(sp>=thr)[0]
    if stable.shape[0] == 0:
        stop = True
    delta = lmbdas[ls]/(np.abs(ols)**0)
    uc = np.sign(ols) * (np.abs(ols)>=delta) * (np.abs(ols) - delta)
    return uc, sp, stop, qs, thr, ls, delta


# In[ ]:


def updateu_pw(X,v0,uw,pcer,ssthr,steps,size,rows_nc=False):
    n = X.shape[0]
    err = n*pcer
    ols = X @ v0
    stop = False
    lmbdas = np.sort(np.append(np.abs(ols),0))
    lmbdas = lmbdas[::-1]
    qs = np.zeros(lmbdas.shape[0])
    thrall = np.zeros(lmbdas.shape[0])
    ls = lmbdas.shape[0]-1
    l = np.where(lmbdas==np.percentile(lmbdas,50,interpolation='nearest'))[0][0]
    l_min = 0
    l_max = lmbdas.shape[0]-1
    
    if rows_nc:
        for g in range(lmbdas.shape[0]):
            tmp = Lasso_nc(X,v0,lmbdas[l],uw,steps,size)
            t = tmp!=0
            qs[l] = np.mean(np.sum(t,axis=0))
            sp = np.mean(t,axis=1)
            thrall[l] = ((qs[l]**2/(err*n))+1)/2
            if thrall[l] >= ssthr[0] and thrall[l] <=ssthr[1]:
                ls = l
                break
            if thrall[l] < ssthr[0]:
                l_min = l
                if l == lmbdas.shape[0]-1:
                    break
                if thrall[l+1] > ssthr[1]:
                    ls = l+1
                    if thrall[l+1] > 0:
                        ls = l
                    tmp = Lasso_nc(X,v0,lmbdas[ls],uw,steps,size)
                    t = tmp!=0
                    qs[ls] = np.mean(np.sum(t,axis=0))
                    sp = np.mean(t,axis=1)
                    thrall[ls] = ((qs[ls]**2/(err*n))+1)/2
                    break
                l = int(np.min([lmbdas.shape[0]-1,l_max,l-1+np.ceil((lmbdas.shape[0])/(g+2))]))
                while thrall[l] != 0:
                    l -= 1
                    if l == 0:
                        break
            if thrall[l] > ssthr[1]:
                l_max = l
                if l == 0: 
                    break
                if thrall[l-1] < ssthr[0] and thrall[l-1]!=0:
                    ls = l
                    break
                l = int(np.max([0,l_min,l-np.ceil(lmbdas.shape[0]/(g+2))+1]))
                while thrall[l] !=0:
                    l += 1
                    if l == lmbdas.shape[0]-1:
                        break
    else: 
        for g in range(lmbdas.shape[0]):
            tmp = Lasso(X,v0,lmbdas[l],uw,steps,size)
            t = tmp!=0
            qs[l] = np.mean(np.sum(t,axis=0))
            sp = np.mean(t,axis=1)
            thrall[l] = ((qs[l]**2/(err*n))+1)/2
            if thrall[l]>=ssthr[0] and thrall[l]<=ssthr[1]:
                ls=l
                break
            if thrall[l]<ssthr[0]:
                l_min = l
                if l == lmbdas.shape[0]-1:
                    break
                if thrall[l+1] > ssthr[1]:
                    ls = l+1
                    if thrall[l+1]>0:
                        ls = l
                    tmp = Lasso(X,v0,lmbdas[ls],uw,steps,size)
                    t = tmp!=0
                    qs[ls] = np.mean(np.sum(t,axis=0))
                    sp = np.mean(t,axis=1)
                    thrall[ls] = ((qs[ls]**2/(err*n))+1)/2
                    break
                l = int(np.min([lmbdas.shape[0]-1,l_max,l-1+np.ceil(lmbdas.shape[0]/(g+2))]))
                while thrall[l] != 0:
                    l -= 1
                    if l==0:
                        break
            if thrall[l]>ssthr[1]:
                l_max = l
                if l == 0:
                    break
                if thrall[l-1] < ssthr[0] and thrall[l-1]!=0:
                    ls = l
                    break
                l = int(np.max([1,l_min,l-np.ceil(lmbdas.shape[0]/(g+2))+1]))
                while thrall[l]!=0:
                    l += 1
                    if l == lmbdas.shape[0]-1:
                        break
    
    thr = ((qs[ls]**2/((pcer*n)*n))+1)/2
    stable = np.where(sp>thr)[0]
    if stable.shape[0] == 0:
        stop = True
    delta = lmbdas[l] / np.abs(ols)**0
    uc = np.sign(ols) * (np.abs(ols)>=delta) * (np.abs(ols)-delta)
    
    return uc, sp, stop, qs, thr, ls, delta


# In[ ]:


# Inner LASSO functionality - subsample and update all subsamples
# Does not allow negative correlations
# Results from each subsample is on the column
def Lasso(X,b,lmbda,weights,steps,size):
    subsets = [np.random.choice(b.shape[0], size=int(b.shape[0]*size), replace=False) for s in range(steps)]
    res = [LassoSteps(s,subsets,X,b,lmbda,weights) for s in range(steps)]
    return np.concatenate(res,axis=1)


# Update one subsample
# Does not allow negative correlations
def LassoSteps(index,subsets,X,b,lmbda,weights):
    ols = X[:,subsets[index]] @ b[subsets[index]]
    ols = ols.reshape(-1,1)*weights.reshape(-1,1)
    delta = lmbda/(np.abs(ols)**0)
    ols = np.sign(ols)*(np.abs(ols)>=delta)*(np.abs(ols)-delta)
    ols[np.isnan(ols)] = 0
    mostof = np.sign(np.sum(np.sign(ols)))
    if mostof == 0:
        mostof = 1
    ols[np.where(np.sign(ols)!=mostof)[0]] = 0
    ols[np.isnan(ols)] = 0
    return ols.reshape(-1,1)


# In[ ]:


# Inner LASSO functionality - subsample and update all subsamples
# Allow negative correlations
def Lasso_nc(X,b,lmbda,weights,steps,size):
    subsets = [np.random.choice(b.shape[0], size=int(b.shape[0]*size), replace=False) for s in range(steps)]
    res = [LassoSteps_nc(s,subsets,X,b,lmbda,weights) for s in range(steps)]
    return np.concatenate(res,axis=1)


# Update one subsample
# Allow negative correlations
def LassoSteps_nc(index,subsets,X,b,lmbda,weights):
    ols = X[:,subsets[index]] @ b[subsets[index]] 
    ols = ols.reshape(-1,1)*weights.reshape(-1,1)
    delta = lmbda/(np.abs(ols)**0)
    ols = np.sign(ols)*(np.abs(ols)>=delta)*(np.abs(ols)-delta)
    ols[np.isnan(ols)] = 0
    return ols.reshape(-1,1)


#visualize clusters
# After Clustered
import seaborn as sns
import matplotlib.pyplot as plt
def plotHeatMap(X,Rows,Cols,D=2,nbicluster=4):
  new_df = []
  for d in range(D):
      cs = []
      d=0
      col = np.array([],dtype=int)
      for i in range(nbicluster):
          r1 = X[d][Rows[i],:]
          col = np.append(col,Cols[i][d])
          c1 = r1[:,col]
          c2 = np.delete(r1, col, axis=1)
          c3 = np.concatenate([c1,c2], axis=1)
          cs.append(c3)
      new_df.append(np.vstack(cs))
      
  fign, axesn = plt.subplots(nrows=1, ncols=2, figsize=(15,3))
  sns.heatmap(new_df[0],ax=axesn[0])
  sns.heatmap(new_df[0],ax=axesn[1])
  plt.show()
  return sns.heatmap