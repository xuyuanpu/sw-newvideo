<callout icon="🛠️" color="gray_bg">
配合 SW Notion AI 视频生成 Skill 使用。电脑工作区重置后，按本页从零恢复工程；/data/sw-video-kit 若仍在，可直接复制使用。
</callout>

## 1. 环境准备

- 工作目录：<code>/data/&lt;片名&gt;/proj</code>（Remotion 工程）、<code>/data/&lt;片名&gt;/qa</code>（检查帧）。
- Remotion 依赖在 <code>/vercel/sandbox/node_modules</code>：在工程目录执行 <code>ln -sfn /vercel/sandbox/node_modules node_modules</code>，否则报 <code>Cannot find package '@remotion/bundler'</code>。
- 语音识别：<code>pip install faster-whisper</code>（模型 medium，CPU int8）。不可用时走第 3 节的静音检测方案。
- BGM：<code>轻松专业 BGM.mp3</code>（= [sw-brand-bgm.mp](http://sw-brand-bgm.mp)3，SHA256 <code>0be39882…</code>）；Logo：官方 <code>logo.png</code>（SHA256 <code>0fee5573…</code>）放进 <code>proj/public/</code>。工作区没有时向用户索取原件。
- 后台渲染与停止：用 <code>nohup node render.mjs &gt; ../render.log 2&gt;&amp;1 &amp;</code> 启动；停止用 <code>pkill -f "[n]ode render.mjs"</code>（方括号写法避免误杀当前 shell）。<b>同一输出文件同一时间只能有一个渲染进程</b>，否则成片损坏（“Invalid NAL unit size”）。渲染完用 <code>ffprobe -count_frames</code> 核对帧数。

## 2. 音频 → 逐字时间轴

1. 用户音频转 16k 单声道：<code>ffmpeg -i voice.mp3 -ar 16000 -ac 1 v16.wav</code>；另转 48k 的 voice.wav 用于混音。
2. 识别：<code>python3 asr.py v16.wav asr.json medium</code>
3. 对齐：配音稿（画面写法，MBTI 写作 MBTI®，每行一句）存为 script.txt，运行 <code>python3 align.py</code> → timing.json（lines 句级、chunks ≤16 字字幕块、full + ct 逐字时间）。匹配率低于 95% 时检查漏读/错读。

```py
from faster_whisper import WhisperModel
import json,sys,wave,numpy as np
m=WhisperModel(sys.argv[3] if len(sys.argv)>3 else "small",device="cpu",compute_type="int8")
w=wave.open(sys.argv[1]);y=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
segs,info=m.transcribe(y,language="zh",word_timestamps=True,beam_size=5,initial_prompt="以下是普通话的句子。MBTI，ISTJ，Skill and Will。")
out=[{"start":s.start,"end":s.end,"text":s.text,"words":[{"word":w.word,"start":w.start,"end":w.end} for w in s.words]} for s in segs]
json.dump({"duration":info.duration,"segments":out},open(sys.argv[2],"w"),ensure_ascii=False,indent=1)
print("".join(s["text"] for s in out))
```

```py
import json,re,difflib
asr=json.load(open('asr.json'))
# ASR chars with times
ac=[];at=[]
for s in asr['segments']:
    for w in s['words']:
        t=re.sub(r'[^\w]','',w['word'].lower())
        n=len(t)
        for i,c in enumerate(t):
            ac.append(c); at.append((w['start']+(w['end']-w['start'])*i/n, w['start']+(w['end']-w['start'])*(i+1)/n))
lines=[l.strip() for l in open('script.txt') if l.strip()]
# script chars
def norm(c):
    c=c.lower()
    return c if re.match(r'\w',c) and c!='_' else None
full=''.join(lines)
sc=[];si=[]
spoken=full.replace('&','and')
# keep mapping indices on 'full' via a parallel string
mapidx=[]
for i,ch in enumerate(full):
    if ch=='&':
        for x in 'and': sc.append(x); mapidx.append(i)
        continue
    n=norm(ch)
    if n and ch!='®': sc.append(n); mapidx.append(i)
sm=difflib.SequenceMatcher(None,sc,ac,autojunk=False)
T=[None]*len(sc); matched=0
for a,b,n in sm.get_matching_blocks():
    for k in range(n): T[a+k]=at[b+k]; matched+=1
print('matched',matched,'/',len(sc))
# interpolate
known=[i for i,t in enumerate(T) if t]
for i in range(len(T)):
    if T[i] is None:
        p=max([k for k in known if k<i],default=None); q=min([k for k in known if k>i],default=None)
        if p is None: T[i]=(T[q][0],T[q][0])
        elif q is None: T[i]=(T[p][1],T[p][1])
        else:
            f=(i-p)/(q-p); x=T[p][1]+(T[q][0]-T[p][1])*f; T[i]=(x,x)
# char time on full string
ft={}
for k,i in enumerate(mapidx):
    a,b=T[k]; 
    if i in ft: ft[i]=(ft[i][0],b)
    else: ft[i]=(a,b)
def span(i0,i1):
    ks=[i for i in range(i0,i1) if i in ft]
    return ft[ks[0]][0], ft[ks[-1]][1]
# sentences/lines
out={'lines':[],'chunks':[]}
pos=0
for l in lines:
    i0=full.index(l,pos); i1=i0+len(l); pos=i1
    a,b=span(i0,i1); out['lines'].append({'text':l,'start':a,'end':b,'i0':i0})
    # chunks: split at punctuation
    parts=re.split(r'(?<=[，。：？、；！])',l)
    j=i0
    for p in parts:
        if not p: continue
        txt=p.rstrip('，。：；、')
        if not txt.strip(): j+=len(p); continue
        a,b=span(j,j+len(p)); out['chunks'].append({'text':txt,'start':round(a,3),'end':round(b,3),'i0':j})
        j+=len(p)
# split long chunks
def vis(t): return len(t.replace('®',''))
nc=[]
for c in out['chunks']:
    t=c['text']
    if vis(t)<=16: nc.append(c); continue
    k=len(t)//2
    # prefer split after 的/了/和/在
    best=min(range(5,len(t)-4),key=lambda m: abs(m-k)-(3 if t[m-1] in '的了和在，' else 0))
    a1,b1=span(c['i0'],c['i0']+best); a2,b2=span(c['i0']+best,c['i0']+len(t))
    nc.append({'text':t[:best],'start':round(a1,3),'end':round(b1,3),'i0':c['i0']})
    nc.append({'text':t[best:],'start':round(a2,3),'end':round(b2,3),'i0':c['i0']+best})
out['chunks']=nc
ct=[]
last=0
for i in range(len(full)):
    if i in ft: last=ft[i][0]
    ct.append(round(last,3))
out['full']=full; out['ct']=ct
json.dump(out,open('timing.json','w'),ensure_ascii=False,indent=1)
for l in out['lines']: print(f"{l['start']:7.2f}-{l['end']:7.2f} {l['text'][:30]}")
print(max(vis(c['text']) for c in out['chunks']))
for c in out['chunks']:
  if vis(c['text'])>13: print(c['text'])
```

## 3. 识别不可用时的备用方案

<code>ffmpeg -i hook.wav -af silencedetect=n=-30dB:d=0.06 -f null -</code> 找句间停顿，按字数在每段内插值；关键字（如强调词）再用 20ms 音量包络找音节间的低谷校准。结果写成与 timing.json 相同的 <code>{full, ct}</code> 结构。

## 4. 工程结构

- <code>Film.jsx</code>：正片。约定：<code>tt(行号, '关键词')</code> 取关键词出现的帧，所有动作挂在真实语音时间上；镜头起点 = 行起点 − 6 帧；共用组件 Span（区间淡入淡出）、KB（插画慢推）、In/Pop/Write（入场）、Caption（字幕层，y≈1548，深色圆角底）、Chrome（常驻商标小字 + 右上角 Logo，片尾卡时隐藏）。
- <code>Hook.jsx</code>：开场钩子，读取 hook.json；字幕在末尾 12 帧前收起；抖动只加在元素上。
- <code>Cover.jsx</code>：封面，按画布高度自适应 9:16 / 3:4。
- <code>index.jsx</code>：Main = 钩子 + 正片（重叠 4 帧）+ 首帧封面（前 6 帧硬切）；另注册 Cover916 / Cover34 两个静帧组合。

```javascript
import React from 'react';
import {registerRoot, Composition, Sequence, AbsoluteFill, useCurrentFrame, interpolate} from 'remotion';
import {Film, TOTAL} from './Film.jsx';
import {Hook, HOOK} from './Hook.jsx';
import {Cover} from './Cover.jsx';
const OV=4;
export const ALL=HOOK-OV+TOTAL;
const CoverIn=()=>{const f=useCurrentFrame();if(f>=6)return null;return <AbsoluteFill><Cover/></AbsoluteFill>;};
const Main=()=><AbsoluteFill style={{background:'#1C2433'}}><Sequence from={HOOK-OV}><Film/></Sequence><Sequence durationInFrames={HOOK}><Hook/></Sequence><CoverIn/></AbsoluteFill>;
const Root=()=><>
 <Composition id="Film" component={Main} durationInFrames={ALL} fps={30} width={1080} height={1920}/>
 <Composition id="Hook" component={Hook} durationInFrames={HOOK} fps={30} width={1080} height={1920}/>
<Composition id="Cover916" component={Cover} durationInFrames={1} fps={30} width={1080} height={1920}/>
 <Composition id="Cover34" component={Cover} durationInFrames={1} fps={30} width={1080} height={1440}/>
</>;
registerRoot(Root);
```

```javascript
import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame, interpolate, Easing} from 'remotion';
import HK from './hook.json';
const FPS=30,F=s=>Math.round(s*FPS);
const cl={extrapolateLeft:'clamp',extrapolateRight:'clamp'};
const font='"Noto Sans SC", sans-serif';
const ink='#1C2433',cobalt='#2D5AAE',cobaltD='#1E3F80',coral='#E9785C',cream='#F6E7C9',white='#fff',green='#3FB27F';
const at=sub=>{const i=HK.full.indexOf(sub);if(i<0)throw new Error(sub);return F(HK.ct[i]);};
export const HOOK=F(HK.end)+18; // hold after last word
const Txt=({size,color=white,w=900,children,style})=><div style={{fontFamily:font,fontWeight:w,fontSize:size,color,lineHeight:1.2,whiteSpace:'nowrap',...style}}>{children}</div>;
const Row=({y,children,style})=><div style={{position:'absolute',left:0,top:y,width:1080,display:'flex',justifyContent:'center',alignItems:'center',gap:30,...style}}>{children}</div>;
const back=(f,a,d=12)=>interpolate(f,[a,a+d],[0,1],{...cl,easing:Easing.out(Easing.back(1.8))});
const Check=({p})=><svg width="130" height="130" viewBox="0 0 56 56" style={{transform:`scale(${0.5+0.5*p})`,opacity:Math.min(1,p*2)}}><circle cx="28" cy="28" r="25" fill={green}/><path d="M15 29 L25 39 L42 18" stroke={white} strokeWidth="6" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="44" strokeDashoffset={44*(1-p)}/></svg>;
const Word=({a,children})=>{const f=useCurrentFrame();const p=back(f,a);return <div style={{opacity:Math.min(1,p*1.6),transform:`scale(${1.35-0.35*p})`}}>{children}</div>;};
export const Hook=()=>{const f=useCurrentFrame();
  const t1=at('零延期'),t2=at('零返工'),tw=at('为什么'),tl=at('丢了客户');
  const dim=interpolate(f,[tw,tw+8],[1,0.45],cl);
  const sp=interpolate(f,[tl,tl+7],[0,1],{...cl,easing:Easing.in(Easing.quad)});
  const shake=f>=tl+7&&f<tl+13?Math.sin((f-tl)*2.6)*8:0;
  const out=interpolate(f,[HOOK-4,HOOK],[1,0],cl);
  // caption chunks
  const cap=f<tw-2?'零延期、零返工的团队':'为什么还是丢了客户？';
  return <AbsoluteFill style={{opacity:out}}>
   <AbsoluteFill style={{background:cobaltD}}><div style={{position:'absolute',left:-260,top:1180,width:950,height:950,borderRadius:'50%',background:cobalt}}/><div style={{position:'absolute',left:720,top:-200,width:620,height:620,borderRadius:'50%',background:cobalt}}/></AbsoluteFill>
   <div style={{opacity:dim}}>
    <Row y={300}><Word a={t1-8}><Txt size={170}>零延期</Txt></Word><Check p={interpolate(f,[t1+6,t1+16],[0,1],cl)}/></Row>
    <Row y={540}><Word a={t2}><Txt size={170}>零返工</Txt></Word><Check p={interpolate(f,[t2+6,t2+16],[0,1],cl)}/></Row>
   </div>
   <Row y={850}><div style={{opacity:interpolate(f,[tw,tw+8],[0,1],cl),transform:`translateY(${interpolate(f,[tw,tw+10],[30,0],{...cl,easing:Easing.out(Easing.cubic)})}px)`}}><Txt size={76} color={cream}>为什么还是</Txt></div></Row>
   <Row y={1000}><div style={{opacity:f<tl?0:1,transform:`translateX(${shake}px) rotate(-6deg) scale(${1.45-0.45*sp})`}}><div style={{background:coral,border:`8px solid ${white}`,borderRadius:28,padding:'14px 54px',boxShadow:'0 20px 40px rgba(0,0,0,.3)'}}><Txt size={140}>丢了客户？</Txt></div></div></Row>
   {f<HOOK-12&&<Row y={1548}><div style={{background:'rgba(20,24,35,.78)',borderRadius:22,padding:'14px 34px'}}><Txt size={48} w={700}>{cap}</Txt></div></Row>}
   <AbsoluteFill><div style={{position:'absolute',left:0,top:1760,width:1080,height:160,background:'linear-gradient(rgba(0,0,0,0),rgba(0,0,0,.42))'}}/>
    <Row y={1862}><Txt size={22} w={500} color="rgba(255,255,255,.8)">MBTI® 是 MBTI 基金会在美国以及世界各国的注册商标</Txt></Row>
    <div style={{position:'absolute',left:1080-48-150,top:52,width:150,background:white,borderRadius:18,padding:'8px 12px',boxShadow:'0 6px 16px rgba(0,0,0,.18)',boxSizing:'border-box'}}><Img src={staticFile('logo.png')} style={{width:126,display:'block'}}/></div></AbsoluteFill>
  </AbsoluteFill>;};
```

```javascript
import React from 'react';
import {AbsoluteFill, Img, staticFile, useVideoConfig} from 'remotion';
const font='"Noto Sans SC", sans-serif';
const cobalt='#2D5AAE',cobaltD='#1E3F80',coral='#E9785C',cream='#F6E7C9',white='#fff',green='#3FB27F',ink='#1C2433';
const Txt=({size,color=white,w=900,children,style})=><div style={{fontFamily:font,fontWeight:w,fontSize:size,color,lineHeight:1.15,whiteSpace:'nowrap',...style}}>{children}</div>;
const Row=({y,children,gap=26})=><div style={{position:'absolute',left:0,top:y,width:1080,display:'flex',justifyContent:'center',alignItems:'center',gap}}>{children}</div>;
const Check=({s=96})=><svg width={s} height={s} viewBox="0 0 56 56"><circle cx="28" cy="28" r="25" fill={green}/><path d="M15 29 L25 39 L42 18" stroke={white} strokeWidth="6" fill="none" strokeLinecap="round" strokeLinejoin="round"/></svg>;
export const Cover=()=>{const {height:H}=useVideoConfig();const tall=H>1600;
  const panel=tall?1010:800, imgY=tall?430:205, y0=tall?250:172;
  return <AbsoluteFill style={{background:cream,overflow:'hidden'}}>
   <div style={{position:'absolute',left:0,top:imgY,width:1080,height:1935}}><Img src={staticFile('lineup.jpg')} style={{width:1080,height:1935}}/></div>
   <svg style={{position:'absolute',left:0,top:0}} width="1080" height={panel+80}><path d={`M0 0 H1080 V${panel-40} Q540 ${panel+70} 0 ${panel-40} Z`} fill={cobaltD}/></svg>
   <div style={{position:'absolute',left:720,top:-200,width:620,height:620,borderRadius:'50%',background:cobalt,opacity:.8}}/>
   <div style={{position:'absolute',left:48,top:56}}><div style={{background:cream,color:ink,fontFamily:font,fontWeight:800,fontSize:34,padding:'10px 26px',borderRadius:999}}>MBTI® 团队偏好</div></div>
   <div style={{position:'absolute',left:1080-48-170,top:44,width:170,background:white,borderRadius:20,padding:'10px 14px',boxSizing:'border-box',boxShadow:'0 6px 16px rgba(0,0,0,.2)'}}><Img src={staticFile('logo.png')} style={{width:142,display:'block'}}/></div>
   <Row y={y0}><Txt size={128}>零延期</Txt><Check/></Row>
   <Row y={y0+165}><Txt size={128}>零返工</Txt><Check/></Row>
   <Row y={y0+350}><Txt size={70} color={cream}>为什么还是</Txt></Row>
   <Row y={y0+460}><div style={{transform:'rotate(-5deg)',background:coral,border:`8px solid ${white}`,borderRadius:28,padding:'8px 50px',boxShadow:'0 18px 36px rgba(0,0,0,.3)'}}><Txt size={124}>丢了客户？</Txt></div></Row>
   <div style={{position:'absolute',left:0,top:H-150,width:1080,height:150,background:'linear-gradient(rgba(0,0,0,0),rgba(0,0,0,.45))'}}/>
   <Row y={H-48}><Txt size={22} w={500} color="rgba(255,255,255,.85)">MBTI® 是 MBTI 基金会在美国以及世界各国的注册商标</Txt></Row>
  </AbsoluteFill>;};
```

## 5. 渲染

- 检查帧：<code>node render.mjs stills 0,120,480</code> → <code>../qa/fNNNNN.png</code>，逐张用看图工具打开。
- 全片：<code>node render.mjs</code> → <code>../film.mp4</code>（无声 H.264，crf 18，并发 2，约 6～8 分钟/2 分钟成片）。
- 封面：<code>node cover.mjs</code> → <code>../Cover916.png</code>、<code>../Cover34.png</code>。

```javascript
import {createRequire} from 'module';import fs from 'fs';
const _ts=createRequire('/vercel/sandbox/node_modules/@remotion/bundler/dist/index.js')('typescript');
if(!_ts.sys){_ts.sys={readFile:p=>fs.existsSync(p)?fs.readFileSync(p,'utf8'):undefined};}
if(!_ts.readConfigFile){_ts.readConfigFile=()=>({config:{}});}
import {bundle} from '@remotion/bundler';
import {renderMedia, renderStill, selectComposition} from '@remotion/renderer';
import path from 'path';
const mode=process.argv[2]||'video';
const serveUrl=await bundle({entryPoint:path.resolve('index.jsx'),publicDir:path.resolve('public')});
const comp=await selectComposition({serveUrl,id:'Film'});
if(mode==='stills'){
  const frames=process.argv[3].split(',').map(Number);
  for(const f of frames){await renderStill({composition:comp,serveUrl,output:`../qa/f${String(f).padStart(5,'0')}.png`,frame:f});}
}else{
  await renderMedia({composition:comp,serveUrl,codec:'h264',crf:18,muted:true,concurrency:2,outputLocation:'../film.mp4',onProgress:({progress})=>{if(Math.round(progress*100)%10===0)process.stdout.write(Math.round(progress*100)+' ')}});
}
process.exit(0);
```

```javascript
import {createRequire} from 'module';import fs from 'fs';
const _ts=createRequire('/vercel/sandbox/node_modules/@remotion/bundler/dist/index.js')('typescript');
if(!_ts.sys){_ts.sys={readFile:p=>fs.existsSync(p)?fs.readFileSync(p,'utf8'):undefined};}
if(!_ts.readConfigFile){_ts.readConfigFile=()=>({config:{}});}
import {bundle} from '@remotion/bundler';
import {renderStill, selectComposition} from '@remotion/renderer';
import path from 'path';
const serveUrl=await bundle({entryPoint:path.resolve('index.jsx'),publicDir:path.resolve('public')});
for(const id of ['Cover916','Cover34']){const c=await selectComposition({serveUrl,id});await renderStill({composition:c,serveUrl,output:`../${id}.png`,frame:0});}
process.exit(0);
```

## 6. 拼接人声、混音、合成

1. 钩子与正片人声拼接（正片在钩子起始后 X 秒开始，X = 正片 Sequence 起始帧 / 30）：<br /><code>ffmpeg -i hook.wav -i voice.wav -filter_complex "[0:a]aresample=48000,aformat=channel_layouts=mono,apad=whole_dur=X[h];[1:a]aresample=48000,aformat=channel_layouts=mono[v];[h][v]concat=n=2:v=0:a=1[a]" -map "[a]" voice2.wav</code>
2. 混音：<code>./mix.sh voice2.wav 总时长 mix.wav</code>（人声 -18 LUFS，BGM 从 60s 起、sidechain 压低、首尾淡入淡出、-1 dBFS 限幅）。
3. 合成：<code>ffmpeg -i film.mp4 -i mix.wav -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -movflags +faststart 成片.mp4</code>
4. 核对：<code>ffprobe</code> 时长；<code>ffmpeg -i 成片.mp4 -af ebur128 -f null -</code> 整体约 -17 LUFS；从成片抽首帧和若干中段帧确认音画同步。

```bash
#!/bin/bash
# 用法: mix.sh voice.wav 总时长秒 输出.wav   （voice.wav 已含钩子+正片，时间轴与画面一致）
V=$1; D=$2; OUT=$3; BGM=${BGM:-/data/sw/assets/templates/video/assets/bgm/sw-brand-bgm.mp3}
FO=$(python3 -c "print($D-1)")
ffmpeg -y -loglevel error -i "$V" -ss 60 -i "$BGM" -filter_complex "[0:a]aresample=48000,loudnorm=I=-18:TP=-2,asplit[v1][v2];[1:a]aresample=48000,volume=-2dB,afade=t=in:d=0.6,afade=t=out:st=$FO:d=1,atrim=0:$D[b];[b][v1]sidechaincompress=threshold=0.03:ratio=2.2:attack=18:release=320[bd];[bd]volume=0.6[bg];[v2][bg]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.891,apad,atrim=0:$D[a]" -map "[a]" -c:a pcm_s16le "$OUT"
```

## 7. 字幕 SRT

由 timing.json 的 chunks 生成，正片时间整体加上钩子偏移 X 秒；钩子句手动写两条；文本中 MBTI 统一为 MBTI®。

```py
import json
c=json.load(open('timing.json'))['chunks'];off=X
def f(s):
    ms=int(round(s*1000));return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"
items=[("钩子前半句",0.08,1.92),("钩子后半句",2.18,3.45)]+[(x['text'].replace('MBTI®','MBTI').replace('MBTI','MBTI®'),x['start']+off,x['end']+off) for x in c]
open('成片.srt','w').write("\n".join(f"{i+1}\n{f(a)} --> {f(b)}\n{t}\n" for i,(t,a,b) in enumerate(items)))
```

## 8. 交付

用 download\_file\_from\_sandbox 导出成片 mp4、srt、9:16 与 3:4 封面；在对话中用视频块 + 文件块展示，并更新内容单元页面的确认版本（含 sha256 前 12 位、时长）与审核记录。
