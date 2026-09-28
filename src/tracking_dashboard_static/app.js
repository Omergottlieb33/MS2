const report=JSON.parse(document.getElementById('report-data').textContent), $=id=>document.getElementById(id);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=2)=>v==null?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:n});
const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
function quantile(values,p){if(!values.length)return null;const a=[...values].sort((a,b)=>a-b),x=(a.length-1)*p,i=Math.floor(x);return a[i]+(a[Math.ceil(x)]-a[i])*(x-i);}
let filtered=[],selected=null,limit=20,roiIds=null;
$('title').textContent=report.title;document.title=report.title;
$('subtitle').textContent=report.frames+' frames · '+report.tracks.length.toLocaleString()+' tracklets · '+report.paths.tracklets.split('/').pop();
$('minFrames').max=report.frames;
$('paths').textContent='Tracklets: '+report.paths.tracklets+' · Masks: '+report.paths.masks;
$('definitions').textContent='Volume CV = standard deviation / mean (at least two observations). Movement uses consecutive frames only; dashed line = '+report.settings.max_step_um+' µm. Neighbors = up to '+report.settings.neighbor_k+' closest cells within '+report.settings.neighbor_radius_um+' µm. Retention counts the same tracked neighbor IDs in the next frame; untracked neighbors stay in the denominator. Both movement and neighborhood tests exclude gaps.';
function refresh(){
 const query=$('search').value.trim(),minimum=Math.max(1,Number($('minFrames').value)||1),key=$('sort').value;
 filtered=report.tracks.filter(t=>(roiIds===null||roiIds.has(t.id))&&t.id.includes(query)&&t.active_frames>=minimum);
 const fields={length:'length_fraction',volume:'volume_cv',movement:'step_p95',neighbors:'neighbor_retention'};
 filtered.sort((a,b)=>{const x=a[fields[key]],y=b[fields[key]];if(x==null)return y==null?0:1;if(y==null)return -1;return (key==='neighbors'?x-y:y-x)||a.id.localeCompare(b.id,undefined,{numeric:true});});
 if(!filtered.some(t=>t.id===selected))selected=filtered[0]?.id??null;
 limit=20;$('count').textContent=filtered.length.toLocaleString()+' of '+report.tracks.length.toLocaleString()+' tracks';
 const cvs=filtered.map(t=>t.volume_cv).filter(x=>x!=null),steps=filtered.flatMap(t=>t.steps.filter(x=>x!=null));
 const kept=filtered.reduce((a,t)=>a+t.neighbor_retained,0),opportunities=filtered.reduce((a,t)=>a+t.neighbor_total,0);
 const large=steps.filter(x=>x>report.settings.max_step_um).length;
 const cards=[
 ['Track length / recording',pct(quantile(filtered.map(t=>t.length_fraction),.5)),'Median detected duration · '+filtered.filter(t=>t.active_frames===report.frames).length+' full-length tracks'],
 ['Volume consistency',pct(quantile(cvs,.5)),'Median volume CV · '+cvs.length.toLocaleString()+' measurable tracks'],
 ['Frame-to-frame movement',fmt(quantile(steps,.95))+(steps.length?' µm':''),'95th percentile · '+large.toLocaleString()+'/'+steps.length.toLocaleString()+' steps above '+report.settings.max_step_um+' µm'],
 ['Nearby-cell consistency',pct(opportunities?kept/opportunities:null),kept.toLocaleString()+'/'+opportunities.toLocaleString()+' neighboring identities retained']
 ];
 $('cards').innerHTML=cards.map(c=>'<div class="card"><h2>'+esc(c[0])+'</h2><div class="value">'+esc(c[1])+'</div><small>'+esc(c[2])+'</small></div>').join('');
 drawRows();drawDetail();
}
function drawRows(){
 $('rows').innerHTML=filtered.slice(0,limit).map(t=>'<tr class="'+(t.id===selected?'selected':'')+'"><td><button data-track="'+esc(t.id)+'">'+esc(t.id)+'</button></td><td>'+pct(t.length_fraction)+'<span class="bar" style="width:'+90*t.length_fraction+'px"></span><small>'+t.active_frames+'/'+report.frames+' frames · '+t.gap_frames+' internal gaps</small></td><td>'+pct(t.volume_cv)+'</td><td>'+fmt(t.step_p95)+(t.step_p95==null?'':' µm')+'<small>'+t.step_tests+' consecutive steps</small></td><td>'+pct(t.neighbor_retention)+'<small>'+t.neighbor_retained+'/'+t.neighbor_total+' comparisons</small></td></tr>').join('')||'<tr><td colspan="5">No tracks match this filter.</td></tr>';
 $('more').hidden=limit>=filtered.length;
}
function chart(values,{max=null,line=null,unit='',color='#087f8c'}={}){
 const finite=values.filter(v=>v!=null);if(!finite.length)return '<div class="empty">No eligible observations</div>';
 const W=550,H=145,L=42,R=12,T=12,B=25,hi=max??(Math.max(...finite,line??0)*1.15||1);
 const xx=t=>L+t/Math.max(1,report.frames-1)*(W-L-R),yy=v=>H-B-v/hi*(H-T-B);
 let svg='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Per-frame '+esc(unit)+'">';
 for(const value of [0,hi/2,hi])svg+='<line class="grid" x1="'+L+'" x2="'+(W-R)+'" y1="'+yy(value)+'" y2="'+yy(value)+'"/><text class="axis" text-anchor="end" x="'+(L-5)+'" y="'+(yy(value)+3)+'">'+fmt(value,2)+'</text>';
 for(const t of [0,Math.floor((report.frames-1)/2),report.frames-1])svg+='<text class="axis" text-anchor="middle" x="'+xx(t)+'" y="'+(H-5)+'">'+t+'</text>';
 if(line!=null)svg+='<line stroke="#bd8552" stroke-dasharray="4 3" x1="'+L+'" x2="'+(W-R)+'" y1="'+yy(line)+'" y2="'+yy(line)+'"/>';
 let segment=[];function flush(){if(segment.length)svg+='<polyline fill="none" stroke="'+color+'" stroke-width="1.7" points="'+segment.join(' ')+'"/>';segment=[];}
 values.forEach((v,t)=>{if(v==null){flush();return;}segment.push(xx(t)+','+yy(v));svg+='<circle r="2.5" fill="'+color+'" cx="'+xx(t)+'" cy="'+yy(v)+'"><title>Frame '+t+': '+fmt(v,3)+' '+esc(unit)+'</title></circle>';});flush();
 return svg+'</svg>';
}
function timeline(t){
 const W=550,H=145,L=12,R=12,w=(W-L-R)/report.frames;
 let svg='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Observed and missing frames">';
 t.volume.forEach((v,i)=>svg+='<rect x="'+(L+i*w)+'" y="38" width="'+Math.max(.3,w-1)+'" height="32" rx="1" fill="'+(v==null?'#dce5e8':'#188f8d')+'"><title>Frame '+i+': '+(v==null?'missing':'observed')+'</title></rect>');
 return svg+'<text class="axis" x="'+L+'" y="95">0</text><text class="axis" text-anchor="end" x="'+(W-R)+'" y="95">'+(report.frames-1)+'</text><text class="axis" x="'+L+'" y="123">Teal: observed · Gray: absent · No gap interpolation</text></svg>';
}
function drawDetail(){
 const t=report.tracks.find(x=>x.id===selected);
 if(!t){$('selectedTitle').textContent='No track selected';$('selectedSummary').textContent='';$('plots').innerHTML='';return;}
 $('selectedTitle').textContent='Track '+t.id;
 const owned=t.neighbor_observations?t.neighbor_tracked/t.neighbor_observations:null;
 $('selectedSummary').textContent='Frames '+t.start+'–'+t.end+' · '+t.active_frames+' detected / '+report.frames+' recording frames · '+t.gap_frames+' internal gaps · Nearby detections with track IDs: '+pct(owned);
 const plots=[
 ['Duration · '+pct(t.length_fraction)+' of recording',timeline(t)],
 ['Volume · CV '+pct(t.volume_cv)+' · median '+fmt(t.median_volume)+' µm³',chart(t.volume.map(v=>v==null?null:v/t.median_volume),{line:1,unit:'× median volume'})],
 ['Movement · p95 '+fmt(t.step_p95)+(t.step_p95==null?'':' µm'),chart(t.steps,{line:report.settings.max_step_um,unit:'µm',color:'#b67543'})],
 ['Nearby identities retained · '+pct(t.neighbor_retention),chart(t.neighbors,{max:1,unit:'fraction',color:'#7655a5'})]
 ];
 $('plots').innerHTML=plots.map(p=>'<div class="plot"><h3>'+esc(p[0])+'</h3>'+p[1]+'</div>').join('');
}
$('rows').onclick=e=>{const b=e.target.closest('button[data-track]');if(!b)return;selected=b.dataset.track;drawRows();drawDetail();$('detail').scrollIntoView({behavior:'smooth',block:'start'});};
for(const id of ['search','minFrames','sort'])$(id).addEventListener('input',refresh);
$('more').onclick=()=>{limit+=30;drawRows();};
$('csv').onclick=()=>{
 const keys=['id','active_frames','length_fraction','start','end','gap_frames','volume_cv','median_volume','step_p95','step_tests','large_steps','neighbor_retention','neighbor_retained','neighbor_total'];
 const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';
 const csv=[keys.join(','),...filtered.map(t=>keys.map(k=>quote(t[k])).join(','))].join('\n');
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv'})),a=document.createElement('a');a.href=url;a.download='tracklet_metrics.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
refresh();

// ROI polygons use pixel (x,y), independent of physical voxel spacing.
function insidePolygon(point, polygon){
 if(!point || polygon.length<3)return false;
 const [x,y]=point;let inside=false;
 for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
  const [xi,yi]=polygon[i],[xj,yj]=polygon[j];
  if(((yi>y)!==(yj>y)) && x<(xj-xi)*(y-yi)/(yj-yi)+xi)inside=!inside;
 }
 return inside;
}
function applyRoi(data,name){
 if(!data || typeof data!=='object')throw Error('Expected an ROI JSON object.');
 if(report.tracks.some(t=>!t.centers_xy))throw Error('Regenerate this report to enable ROI filtering.');
 const first=data.t_first??0,last=data.t_last==null||data.t_last<0?report.frames-1:data.t_last;
 const boundaries=[[data.roi_first??[],first],[data.roi_last??[],last]];
 for(const [polygon,frame] of boundaries){
  if(!Array.isArray(polygon))throw Error('ROI polygons must be arrays of [x, y] vertices.');
  if(!polygon.length)continue;
  if(polygon.length<3||polygon.some(p=>!Array.isArray(p)||p.length!==2||p.some(v=>typeof v!=='number'||!Number.isFinite(v))))throw Error('Each polygon needs at least three finite [x, y] vertices.');
  if(!Number.isInteger(frame)||frame<0||frame>=report.frames)throw Error('ROI frame is outside this recording.');
 }
 if(!boundaries.some(([p])=>p.length))throw Error('The ROI contains no polygons.');
 // Catch accidentally loading an ROI from another recording, while permitting renamed copies.
 const source=data.tracklets;
 if(source && source.startsWith('/') && source.split('/').slice(0,-1).join('/')!==report.paths.tracklets.split('/').slice(0,-1).join('/'))
  throw Error('This ROI belongs to another tracklet folder. Open the corresponding recording first.');
 const ids=new Set(report.tracks.filter(t=>boundaries.some(([p,f])=>insidePolygon(t.centers_xy[f],p))).map(t=>t.id));
 roiIds=ids;
 $('roiStatus').textContent=name+' · '+ids.size+' of '+report.tracks.length+' tracks · inside either endpoint polygon; whole-track metrics';
 $('clearRoi').hidden=false;refresh();
}
$('roiFile').onchange=async e=>{
 const file=e.target.files[0];if(!file)return;
 try{applyRoi(JSON.parse(await file.text()),file.name);}
 catch(err){$('roiStatus').textContent='ROI not applied: '+err.message+(roiIds===null?' All tracks remain selected.':' Previous ROI remains selected.');}
};
$('clearRoi').onclick=()=>{roiIds=null;$('roiFile').value='';$('clearRoi').hidden=true;$('roiStatus').textContent='All tracks · no ROI selected';refresh();};
if(report.roi){try{applyRoi(report.roi.data,report.roi.name);}catch(err){$('roiStatus').textContent='ROI not applied: '+err.message;}}
