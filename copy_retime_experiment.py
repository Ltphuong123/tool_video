from pathlib import Path
import subprocess
import tempfile
from apps.video_fast_export import _ffmpeg_binary

ff = _ffmpeg_binary()
root = Path(tempfile.mkdtemp(prefix='copy_retime_'))
print(root)
src = root/'source.mp4'
subprocess.run([ff,'-hide_banner','-loglevel','error','-f','lavfi','-i','testsrc2=s=160x90:r=30000/1001:d=3','-c:v','libx264','-bf','3','-g','90',str(src)],check=True)
script=root/'copy.bsf'
script.write_text("setts=pts='st(0,PTS*TB);if(lt(ld(0),1),ld(0)*0.5,0.5+(ld(0)-1)*2)/TB_OUT':dts='st(0,DTS*TB);if(lt(ld(0),1),ld(0)*0.5,0.5+(ld(0)-1)*2)/TB_OUT':duration='st(0,(PTS+DURATION)*TB);st(1,if(lt(ld(0),1),ld(0)*0.5,0.5+(ld(0)-1)*2));st(0,PTS*TB);(ld(1)-if(lt(ld(0),1),ld(0)*0.5,0.5+(ld(0)-1)*2))/TB_OUT':time_base=1/1000000",encoding='utf-8')
dst=root/'output.mp4'
p=subprocess.run([ff,'-hide_banner','-loglevel','warning','-copyts','-start_at_zero','-i',str(src),'-map','0:v:0','-c:v','copy','-/bsf:v',str(script),'-an','-avoid_negative_ts','disabled','-movflags','+faststart',str(dst)],capture_output=True)
print(p.returncode,p.stderr.decode(errors='replace'))
for file in (src,dst):
 p=subprocess.run([ff,'-hide_banner','-loglevel','error','-i',str(file),'-map','0:v:0','-c:v','copy','-f','framehash','-hash','sha256','-'],capture_output=True)
 out=p.stdout.decode(); (root/(file.stem+'.framehash')).write_text(out)
 data=[line for line in out.splitlines() if line and not line.startswith('#')]
 print(file,len(data),'\n'.join(data[:6]),'\nlast', '\n'.join(data[-4:]))
 p=subprocess.run([ff,'-hide_banner','-loglevel','error','-i',str(file),'-map','0:v:0','-fps_mode','passthrough','-f','framemd5','-'],capture_output=True)
 hashes=[line.rsplit(',',1)[-1].strip() for line in p.stdout.decode().splitlines() if line and not line.startswith('#')]
 print('decoded',len(hashes),'unique',len(set(hashes)))
 if file==src: first_hashes=hashes
 else: print('pixelhash equal',hashes==first_hashes)
