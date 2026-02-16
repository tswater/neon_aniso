# Fix scalar L2 and uvw L2
# -- add ANID to L2 for scalars
# -- add u to L2 for uvw
from neonutil.l2tools import add_from_l1


l1base='/home/tswater/tyche/data/neon/L1/'
scalar_base = '/home/tswater/tyche/data/neon/L2/L2_scalar/'
scalar_files= ['q_stable.h5','q_unstable.h5','t_unstable.h5','t_stable.h5','c_unstable.h5','c_stable.h5']
sl1dir=['neon_5m/','neon_30m/','neon_30m/','neon_5m/','neon_30m/','neon_5m/']
ul1dir=['neon_5m/','neon_30m/','neon_30m/']
uvw_base ='/home/tswater/tyche/data/neon/L2/L2_ybd/'
uvw_files= ['ybd_stable5.h5','ybd_stable30.h5','ybd_unstable.h5']

for i in range(len(scalar_files)):
    file=scalar_files[i]
    l1=l1base+sl1dir[i]
    fpath=scalar_base+file
    print(fpath)
    add_from_l1(fpath,'main',['ANID_YBs','ANID_XBs'],l1dir=l1)

for i in range(len(uvw_files)):
    file=uvw_files[i]
    l1=l1base+ul1dir[i]
    fpath=uvw_base+file
    print(fpath)
    add_from_l1(fpath,'main',['Us'],l1dir=l1)


