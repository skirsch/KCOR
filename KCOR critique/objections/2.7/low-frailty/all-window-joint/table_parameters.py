"""Record fitted parameters on the same Table 8 bootstrap draws."""
from experiment import *
def init(pop):
    table_init(pop,"profile")
    global base
    base=table.compare.joint_fit
def task(pair):
    i,j=pair
    pars=[]
    def recorded(*args,**kwargs):
        p,d=base(*args,**kwargs)
        pars.append([p[0],p[1],float(d["success"])])
        return p,d
    table.compare.joint_fit=recorded
    table.estimate(i,j)
    return i,j,np.array(pars)
def main():
    book=io.BytesIO(subprocess.check_output(["git","cat-file","blob",table.INPUT_BLOB],cwd=ROOT))
    pop=[table.read_population(book,s) for s in table.SHEETS]
    data=np.empty((2,1000,9,3))
    with ProcessPoolExecutor(5,initializer=init,initargs=(pop,)) as pool:
        for i,j,v in pool.map(task,[(i,j) for j in range(1000) for i in range(2)],chunksize=5):
            data[i,j]=v
    np.savez_compressed(HERE/"profile-table-parameters.npz",parameters=data)
if __name__=="__main__": main()
