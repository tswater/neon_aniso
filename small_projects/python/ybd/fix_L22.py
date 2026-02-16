# Fix scalar L2 and uvw L2
# -- add ANID to L2 for scalars
# -- add u to L2 for uvw
from neonutil.l2tools import add_from_l1


l1base='/home/tswater/tyche/data/neon/L1/'
ul1dir=['neon_5m/','neon_30m/','neon_30m/']
uvw_base ='/home/tswater/tyche/data/neon/L2/L2_revision/'
uvw_files= ['uvw_stable.h5','uvw_unstable.h5']

for i in range(len(uvw_files)):
    file=uvw_files[i]
    l1=l1base+ul1dir[i]
    fpath=uvw_base+file
    print(fpath)
    add_from_l1(fpath,'main',['ANID_YBs','ANID_XBs'],l1dir=l1)

