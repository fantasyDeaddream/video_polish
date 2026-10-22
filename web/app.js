'use strict';
const $ = s => document.querySelector(s);
const state = {current: null, jobs: [], preset: '1080p', planVersion: 0, validPlan: false, uploading: false, previews: new Set(), pollBusy: false};
const statusText = {uploaded:'待设置', queued:'排队中', running:'处理中', done:'已完成', partial:'部分完成', failed:'处理失败', cancelled:'已取消'};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const bytes = n => n >= 1024**3 ? `${(n/1024**3).toFixed(2)} GB` : `${(n/1024**2).toFixed(1)} MB`;
const duration = seconds => { if(seconds>0&&seconds<1)return `${seconds.toFixed(2)} 秒`; const s = Math.max(0, Math.round(seconds || 0)); return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`; };
let toastTimer;
function toast(message, error=false) { const el=$('#toast'); el.textContent=message; el.classList.toggle('error',error); el.hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>el.hidden=true,error?9000:4500); }
async function api(path, options={}) {
  const response = await fetch(path, options);
  if (!response.ok) { let data; try { data=await response.json(); } catch { data={detail:`HTTP ${response.status}`}; } throw new Error(typeof data.detail==='string'?data.detail:Array.isArray(data.detail)?data.detail.map(e=>e.msg?.replace(/^Value error, /,'')||'参数无效').join('；'):JSON.stringify(data.detail)); }
  return response.json();
}
const post = (path, body) => api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
function target() {
  if ($('#workflow').value==='interpolate') return {preset:'1080p'};
  const pixels=Number($('#custom-pixels').value);
  if (state.preset==='custom' && (!Number.isInteger(pixels)||pixels<16||pixels>7680)) throw new Error('自定义像素需为 16–7680 的整数');
  return {preset:state.preset, edge:$('#custom-edge').value, pixels:state.preset==='custom'?pixels:1080};
}
function interpolation() { return {enabled:$('#workflow').value!=='upscale',model:$('#rife-model').value,target_fps:$('#target-fps').value==='custom'?$('#custom-fps').value.trim():$('#target-fps').value,backend:$('#rife-backend').value,precision:$('#rife-precision').value,scale:Number($('#rife-scale').value),scene_cut_detection:$('#scene-cuts').checked,order:'sr_then_vfi'}; }
function options() { return {target:target(),workflow:$('#workflow').value,interpolation:interpolation(),model:$('#model').value,strength:Number($('#strength').value)/100,temporal:Number($('#temporal').value)/100,crf:Number($('#crf').value),tile:Number($('#tile').value),tensorrt:$('#tensorrt').checked,tensorrt_precision:$('#trt-precision').value,pipeline:$('#pipeline').checked}; }
function canStart() { const needsRife=$('#workflow').value!=='upscale';const ready=!needsRife||state.rifeModels?.some(m=>m.id===$('#rife-model').value&&m.available);$('#start').disabled=!state.current || !state.validPlan || state.uploading || !ready; }
function renderSource(plans=[]) {
  const items=state.current?.files||[];
  $('#source-empty').hidden=items.length>0;
  $('#source-footer').hidden=!items.length;
  $('#source-count').textContent=`${items.filter(f=>f.metadata).length} 份有效视频 · ${bytes(items.reduce((n,f)=>n+(f.metadata?.size_bytes||0),0))}`;
  $('#source-list').innerHTML=items.map(f=>{
    const entry=plans.find(p=>p.index===f.index),p=entry?.target,ip=entry?.interpolation;
    if (!f.metadata) return `<div class="source-row"><div class="file-icon">!</div><div><div class="file-name">${esc(f.name)}</div><div class="error">${esc(f.error)}</div></div></div>`;
    const m=f.metadata;
    return `<div class="source-row"><div class="file-icon">▶</div><div><div class="file-name">${esc(f.name)}</div><div class="file-meta">${duration(m.duration)} · ${m.fps} fps · ${bytes(m.size_bytes)} · ${m.audio_tracks?'有音轨':'无音轨'}${m.rotation?' · 已校正旋转':''}</div><div class="resolution"><span>${m.width} × ${m.height}${m.sar!==1?'（非方形像素）':''}</span><span class="arrow">→</span><span>${p?`${p.width} × ${p.height}`:'计算输出方案…'}</span></div>${ip?.enabled?`<div class="resolution-note">${m.fps} → ${esc(ip.rate)} fps · 约 ${ip.estimated_frames??'待解码确认'} 帧 · 保持时长</div>`:''}${p&&p.mode!=='upscale'?`<div class="resolution-note">${$('#workflow').value==='interpolate'?'仅插帧：保留显示尺寸，跳过超分。':'目标不大于原片：使用高质量缩放，跳过神经超分。'}</div>`:''}</div></div>`;
  }).join('');
  canStart();
}
async function updatePlan() {
  const version=++state.planVersion;
  state.validPlan=false; canStart();
  const hints={'720p':'短边 720 px','1080p':'短边 1080 px','2k':'长边 2560 px','4k':'长边 3840 px','custom':'自定义基准边'};
  $('#target-hint').textContent=`${hints[state.preset]} · 自动计算另一条边`;
  $('#custom-target').hidden=state.preset!=='custom';
  $('#custom-fps-label').hidden=$('#target-fps').value!=='custom';
  let t;
  try { t=target(); } catch(e) { $('#target-hint').textContent=e.message; return; }
  if (!state.current) return;
  try {
    const plan=await post('/api/plan',{upload_id:state.current.id,target:t,workflow:$('#workflow').value,interpolation:interpolation()});
    if (version!==state.planVersion) return;
    state.validPlan=true;$('#rife-hint').textContent='保持时长和音轨。超分并插帧时，先增强原始帧，再生成中间帧。快速运动、遮挡和字幕可能出现伪影。';renderSource(plan.files);
  } catch(e) { if(version===state.planVersion) { $('#target-hint').textContent=e.message;$('#rife-hint').textContent=e.message; canStart(); } }
}
async function uploadFiles(files) {
  if(state.uploading||!files.length) return;
  if(files.length>32) return toast('每批最多选择 32 份视频',true);
  const old=state.current;
  state.uploading=true;canStart();$('#dropzone').classList.add('busy');$('#upload-progress').hidden=false;
  const form=new FormData();for(const file of files) form.append('files',file);
  try {
    const result=await new Promise((resolve,reject)=>{
      const xhr=new XMLHttpRequest();xhr.open('POST','/api/uploads');
      xhr.upload.onprogress=e=>{if(e.lengthComputable){const p=Math.round(e.loaded/e.total*100);$('#upload-progress progress').value=p;$('#upload-progress span').textContent=p===100?'正在读取视频信息…':`上传 ${p}%`;}};
      xhr.onerror=()=>reject(new Error('无法连接本地服务'));
      xhr.onabort=()=>reject(new Error('上传已取消'));
      xhr.onload=()=>{let data;try{data=JSON.parse(xhr.responseText);}catch{return reject(new Error(`上传失败 HTTP ${xhr.status}`));}if(xhr.status>=200&&xhr.status<300)resolve(data);else reject(new Error(data.detail||'上传失败'));};
      xhr.send(form);
    });
    state.current=result;renderSource();await updatePlan();
    if(old) await api(`/api/jobs/${old.id}`,{method:'DELETE'}).catch(()=>{});
    toast(`已读取 ${result.files.filter(f=>f.metadata).length} 份视频`);
  } catch(e) {toast(e.message,true);} finally {state.uploading=false;$('#files').value='';$('#dropzone').classList.remove('busy');$('#upload-progress').hidden=true;canStart();}
}
$('#files').addEventListener('change',e=>uploadFiles([...e.target.files]));
$('#dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#files').click();}});
['dragenter','dragover'].forEach(type=>$('#dropzone').addEventListener(type,e=>{e.preventDefault();$('#dropzone').classList.add('dragover');}));
['dragleave','drop'].forEach(type=>$('#dropzone').addEventListener(type,e=>{e.preventDefault();$('#dropzone').classList.remove('dragover');}));
$('#dropzone').addEventListener('drop',e=>uploadFiles([...e.dataTransfer.files]));
$('#presets').addEventListener('click',e=>{const button=e.target.closest('[data-preset]');if(!button)return;state.preset=button.dataset.preset;document.querySelectorAll('[data-preset]').forEach(el=>{el.classList.toggle('selected',el===button);el.setAttribute('aria-checked',String(el===button));});updatePlan();});
$('#custom-edge').addEventListener('change',updatePlan);$('#custom-pixels').addEventListener('input',updatePlan);
$('#workflow').addEventListener('change',()=>{const workflow=$('#workflow').value;$('#sr-settings').hidden=workflow==='interpolate';$('#interpolation-settings').hidden=workflow==='upscale';$('#tile').disabled=workflow==='interpolate';updatePlan();});
for(const id of ['target-fps','rife-model','rife-backend','rife-precision','rife-scale','scene-cuts'])$(`#${id}`).addEventListener('change',updatePlan);
$('#custom-fps').addEventListener('input',updatePlan);
['strength','temporal'].forEach(id=>$(`#${id}`).addEventListener('input',()=>{$(`#${id}-value`).textContent=$(`#${id}`).value+'%';}));
$('#model').addEventListener('change',()=>{const model=$('#model').value,disabled=model==='lanczos';$('#strength').disabled=disabled;$('#temporal').disabled=disabled;$('#tensorrt').disabled=disabled||$('#tensorrt').dataset.available!=='true';if(disabled)$('#tensorrt').checked=false;$('#trt-precision-label').hidden=!$('#tensorrt').checked;const hints={'auto':'按原生倍率超分后再缩放到目标尺寸，保留原始输入细节。','quality':'不超过 2 倍使用 x2plus，否则使用 x4plus。','general-fast':'轻量通用模型，原生 4 倍后缩至目标尺寸；纹理表现与 RRDB 不同。','general-clean':'轻量弱降噪权重，保留更多噪点与纹理；并非更强降噪。','anime':'动漫 RRDB 模型，原生 4 倍。','anime-fast':'动漫视频轻量模型，原生 4 倍；更适合线条、动画画面。','lanczos':'纯插值缩放，不执行神经超分、光流稳定或 TensorRT。'};$('#model-hint').textContent=hints[model];});
$('#tensorrt').addEventListener('change',()=>{$('#trt-precision-label').hidden=!$('#tensorrt').checked;});
$('#clear-source').addEventListener('click',async()=>{if(!state.current)return;try{await api(`/api/jobs/${state.current.id}`,{method:'DELETE'});state.current=null;state.validPlan=false;renderSource();}catch(e){toast(e.message,true);}});
$('#start').addEventListener('click',async()=>{
  if(!state.current||!state.validPlan)return;
  $('#start').disabled=true;
  try{await post('/api/jobs',{upload_id:state.current.id,options:options()});state.current=null;state.validPlan=false;renderSource();await refreshJobs();$('#jobs').scrollIntoView({behavior:'smooth',block:'nearest'});toast('已加入处理队列，关闭页面后服务仍会继续处理。');}catch(e){toast(e.message,true);canStart();}
});
function renderJobs() {
  const jobs=state.jobs.filter(j=>j.status!=='uploaded');
  $('#jobs-count').textContent=jobs.length;$('#jobs-empty').hidden=jobs.length>0;
  const done=jobs.flatMap(j=>j.files.filter(f=>f.status==='done').map(f=>({...f,jobId:j.id})));
  $('#save-all').hidden=!done.length;
  // Preserve playing video elements across polling updates.
  const videos=new Map([...document.querySelectorAll('video[data-key]')].map(v=>[v.dataset.key,v]));
  const markup=jobs.map(job=>{
    const active=['running','queued'].includes(job.status);
    return `<article class="job-card" data-job-card="${job.id}"><div class="job-header"><div><span class="status ${esc(job.status)}">● ${esc(statusText[job.status]||job.status)}</span><small>${job.files.length} 份 · ${new Date(job.created_at*1000).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}</small></div><button class="text-button" data-action="${active?'cancel':'delete'}" data-job="${job.id}">${active?'取消并清理':'删除缓存'}</button></div>${job.error?`<div class="error" style="padding:10px 20px">${esc(job.error)}</div>`:''}<div class="job-files">${job.files.map(f=>{
      const key=`${job.id}-${f.index}`,m=f.metadata,t=f.target;
      const ip=f.interpolation_plan;
      const subtitle=m&&t?`${m.width} × ${m.height} → ${t.width} × ${t.height} · ${m.fps}${ip?.enabled?` → ${ip.rate}`:''} fps`:'';
      const engine=f.evidence?.engine||f.engine;
      const validations=engine?.tensorrt?.validations||[];
      const qualityLabel=validations.length?` · 与 PyTorch 校验 MAE ≤ ${Math.max(...validations.map(v=>v.mae)).toFixed(3)}/255`:'';
      const vfi=f.evidence?.interpolation;
      const srLabel=engine?.backend==='realesrgan'?`AI · ${engine.model.replace('.pth','')} · ${engine.inference_engine==='tensorrt'?`TensorRT ${engine.tensorrt?.precision?.toUpperCase()||''}`:engine.device?.toUpperCase()}${engine.tile?` · 分块 ${engine.tile}`:''}`:engine?.backend==='lanczos'?'Lanczos 缩放':'原始尺寸';
      const label=srLabel+(vfi?` · ${vfi.model} / ${vfi.backend} · ${vfi.output_frames} 帧`: '');
      const detail=f.status==='running'?`${f.stage||'处理视频'} · 原始帧 ${f.processed_frames||0}${ip?.enabled?` · 输出 ${f.output_frames||0} 帧`:''} · 处理 ${f.processing_fps||0} 原帧/秒${f.eta_seconds?` · 预计剩余 ${duration(f.eta_seconds)}`:''}`:f.status==='done'?`${label} · ${bytes(f.result_bytes)} · 耗时 ${duration(f.evidence?.elapsed_seconds)}${qualityLabel}`:statusText[f.status]||f.status;
      return `<div class="job-file"><div class="job-line"><div><div class="file-name">${esc(f.name)}</div><div class="file-meta">${esc(subtitle)}</div></div><div class="file-actions">${f.status==='done'?`<button data-action="preview" data-job="${job.id}" data-index="${f.index}">${state.previews.has(key)?'收起预览':'预览'}</button><button data-action="save" data-job="${job.id}" data-index="${f.index}">另存为 ↗</button>`:''}</div></div><div class="job-detail"><span>${esc(detail)}</span>${f.status==='running'?`<span>${Math.round((f.progress||0)*100)}%</span>`:''}</div>${['queued','running'].includes(f.status)?`<progress max="1" value="${f.progress||0}" aria-label="处理进度"></progress>`:''}${f.error?`<div class="error">${esc(f.error)}</div>`:''}${state.previews.has(key)&&f.status==='done'?`<div data-video-slot="${key}"></div>`:''}</div>`;
    }).join('')}</div></article>`;
  }).join('');
  const root=$('#jobs'),template=document.createElement('template');template.innerHTML=markup;
  const desired=jobs.map((job,index)=>{const signature=JSON.stringify([job,[...state.previews].filter(k=>k.startsWith(job.id))]);const old=root.querySelector(`[data-job-card="${job.id}"]`);if(old?.dataset.signature===signature)return old;const card=template.content.children[index];card.dataset.signature=signature;return card;});
  for(const child of [...root.children])if(!desired.includes(child))child.remove();
  desired.forEach((card,index)=>{if(root.children[index]!==card)root.insertBefore(card,root.children[index]||null);});
  for(const job of jobs)for(const f of job.files){const key=`${job.id}-${f.index}`,slot=document.querySelector(`[data-video-slot="${key}"]`);if(!slot)continue;const video=videos.get(key)||document.createElement('video');video.className='video-preview';video.dataset.key=key;video.controls=true;video.preload='metadata';if(!videos.has(key))video.src=f.preview_url;if(video.parentElement!==slot)slot.append(video);}
}
async function refreshJobs() {
  if(state.pollBusy)return;state.pollBusy=true;
  try{const data=await api('/api/jobs');state.jobs=data.jobs;if(!state.current&&!state.uploading){const pending=data.jobs.find(j=>j.status==='uploaded');if(pending){state.current=pending;renderSource();updatePlan();}}renderJobs();}catch(e){$('#health').textContent='本地服务已断开';$('#health').classList.remove('ready');}finally{state.pollBusy=false;}
}
function findFile(jobId,index){return state.jobs.find(j=>j.id===jobId)?.files.find(f=>f.index===Number(index));}
async function saveFile(file) {
  if(!file?.download_url)return;
  if(typeof window.showSaveFilePicker === 'function'){
    let handle;
    try{handle=await window.showSaveFilePicker({suggestedName:file.output_name,types:[{description:'MP4 视频',accept:{'video/mp4':['.mp4']}}]});}catch(e){if(e.name!=='AbortError')toast(e.message,true);return;}
    let writable;
    try{const response=await fetch(file.download_url);if(!response.ok)throw new Error('结果已过期或被清理');writable=await handle.createWritable();await response.body.pipeTo(writable);toast('视频已保存');}catch(e){if(writable)await writable.abort().catch(()=>{});toast(`保存失败：${e.message}`,true);}
  }else{const a=document.createElement('a');a.href=file.download_url;a.download=file.output_name;a.click();toast('已交给浏览器下载；可在浏览器设置中开启每次询问保存位置。');}
}
$('#jobs').addEventListener('click',async e=>{
  const button=e.target.closest('button[data-action]');if(!button)return;
  const {action,job,index}=button.dataset;const file=findFile(job,index);
  try{
    if(action==='save')return await saveFile(file);
    if(action==='preview'){const key=`${job}-${index}`;if(state.previews.has(key))state.previews.delete(key);else state.previews.add(key);renderJobs();return;}
    if(action==='cancel')await post(`/api/jobs/${job}/cancel`,{});
    if(action==='delete')await api(`/api/jobs/${job}`,{method:'DELETE'});
    await refreshJobs();
  }catch(e){toast(e.message,true);}
});
$('#save-all').addEventListener('click',async()=>{
  const files=state.jobs.flatMap(j=>j.files.filter(f=>f.status==='done').map(f=>({...f,jobId:j.id})));
  if(typeof window.showDirectoryPicker !== 'function'){for(const file of files){const a=document.createElement('a');a.href=file.download_url;a.download=file.output_name;a.click();}return toast('已请求批量下载；若浏览器提示，请允许多个文件下载。');}
  let folder;try{folder=await window.showDirectoryPicker({mode:'readwrite'});}catch(e){if(e.name!=='AbortError')toast(e.message,true);return;}
  const button=$('#save-all');button.disabled=true;let count=0;
  try{for(const file of files){const name=`${file.jobId.slice(0,6)}_${file.index+1}_${file.output_name}`;const handle=await folder.getFileHandle(name,{create:true});const response=await fetch(file.download_url);if(!response.ok)throw new Error('部分结果已清除');const writable=await handle.createWritable();try{await response.body.pipeTo(writable);}catch(e){await writable.abort().catch(()=>{});throw e;}count++;button.textContent=`正在保存 ${count} / ${files.length}`;}toast(`已另存 ${count} 份视频`);}catch(e){toast(`已保存 ${count} 份；${e.message}`,true);}finally{button.disabled=false;button.textContent='批量另存到文件夹 ↗';}
});
async function init(){
  try{const health=await api('/api/health');$('#health').innerHTML='<i></i>'+ (health.models.some(m=>m.available)?'AI 模型就绪 · 本地处理':'本地服务就绪 · 缺少 AI 权重');$('#health').classList.add('ready');const ready=!!health.acceleration?.available;$('#tensorrt').dataset.available=String(ready);$('#tensorrt').disabled=!ready;$('#trt-status').textContent=ready?'已安装':'未安装';$('#trt-hint').textContent=ready?'首次启用需编译，缓存复用；精度差异将写入任务验证记录。':'未检测到组件；运行 scripts/setup_acceleration.py 后重启服务。';state.rifeModels=health.interpolation?.models||[];$('#rife-status').textContent=state.rifeModels.every(m=>m.available)&&state.rifeModels.length?'模型就绪':'缺少权重';for(const m of state.rifeModels){const option=$(`#rife-model option[value="${m.id}"]`);if(option)option.disabled=!m.available;}$('#rife-backend option[value="tensorrt"]').disabled=!ready;for(const id of ['general-fast','general-clean','anime-fast']){const entry=health.models.find(m=>m.id===id);const option=$(`#model option[value="${id}"]`);if(option&&!entry?.available){option.disabled=true;option.textContent+='（缺少权重）';}}await refreshJobs();}catch(e){$('#health').textContent='服务未连接';toast(e.message,true);}
  setInterval(refreshJobs,1800);
}
init();
