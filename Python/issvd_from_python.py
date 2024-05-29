#%%
import issvd_functions as func
mydata=func.gen_sim_vec(myseed=25,n=200,p=1000,D=2,rowsize=50, colsize=100, numbers=1,sigma=0.1,nbicluster=4, orthonm=False)
X1 = mydata[0][0][0]
X2 = mydata[0][0][1]
Xdata = [X1, X2]
res = func.issvd(X=Xdata, standr=False, pointwise=True, steps=100,size=0.5,ssthr=[0.6,0.8],
                                nbicluster=4,rows_nc=False,cols_nc=False,col_overlap=False,row_overlap=False,
                                pceru=0.1,pcerv=0.1,merr=0.0001,iters=100)
myRows = res['Sample_index']
myCols = res['Variable_index']

#print(len(Xdata))
#print(Xdata[0].shape)
#print(Xdata[1].shape)
#print(Xdata[0][myRows[0],:])
#print(Xdata[0][myRows[1],:])
#print(Xdata[0][myRows[2],:])
#print(Xdata[0][myRows[3],:])
#func.plotHeatMap(X=Xdata,Rows=myRows,Cols=myCols,D=2,nbicluster=4)

func.plotHeatMap(X=Xdata,Rows=myRows,Cols=myCols,D=2,nbicluster=3)
plt.show()
# %%
