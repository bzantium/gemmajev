let data, tick=0, playing=false, last=0, maximum=0;
const panels=document.querySelector('#panels'), timeline=document.querySelector('#timeline');
const directions=['north','east','south','west'];
const layers=[];
async function start(){
  data=await(await fetch('demo.json')).json();
  maximum=1000;timeline.max=maximum;
  for(const maze of data.mazes){
    const panel=document.createElement('article');panel.className='panel';
    panel.innerHTML='<div class="panel-head"><h2></h2><span class="size"></span></div><canvas class="map" width="640" height="640"></canvas><div class="status"><span class="outcome"></span><span class="metric"></span></div><p class="prob-heading">Probability the next cell is open</p><div class="bars"></div>';
    panel.querySelector('h2').textContent=maze.label;panel.querySelector('.size').textContent=`${maze.size} × ${maze.size}`;
    directions.forEach(key=>{const bar=document.createElement('div');bar.innerHTML=`<div class="bar-label"><span>${key}</span><span class="value"></span></div><div class="track"><div class="fill"></div></div>`;panel.querySelector('.bars').append(bar)});
    panels.append(panel);
    const layer=document.createElement('canvas');layer.width=640;layer.height=640;const ctx=layer.getContext('2d'),scale=620/maze.size;
    ctx.fillStyle='#fafbfc';ctx.fillRect(0,0,640,640);ctx.fillStyle='#c8ced8';
    maze.initial.walls.forEach(([row,col])=>ctx.fillRect(10+col*scale,10+row*scale,scale+.3,scale+.3));layers.push(layer);
  }
  await document.fonts.ready;draw();window.demoReady=true;requestAnimationFrame(animate);
}
function draw(){
  const current=Math.floor(tick);timeline.value=current;
  data.mazes.forEach((maze,i)=>{
    const panel=panels.children[i],frames=maze.system.frames,k=Math.min(Math.floor(current/maximum*(frames.length-1)),frames.length-1),frame=frames[k];
    const ctx=panel.querySelector('canvas').getContext('2d'),scale=620/maze.size,point=([r,c])=>[10+(c+.5)*scale,10+(r+.5)*scale];
    ctx.drawImage(layers[i],0,0);ctx.strokeStyle='#2879ff';ctx.lineWidth=3;ctx.lineJoin='round';ctx.lineCap='round';ctx.globalAlpha=.5;ctx.beginPath();
    frames.slice(0,k+1).forEach((f,n)=>{const [x,y]=point(f.position);n?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.globalAlpha=1;
    const [gx,gy]=point(maze.initial.goal);ctx.fillStyle='#ffe100';ctx.fillRect(gx-scale/2,gy-scale/2,scale,scale);ctx.strokeStyle='#b49d00';ctx.lineWidth=1;ctx.strokeRect(gx-scale/2,gy-scale/2,scale,scale);
    const [x,y]=point(frame.position);ctx.beginPath();ctx.arc(x,y,5.5,0,Math.PI*2);ctx.fillStyle='#2879ff';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.5;ctx.stroke();
    if(frame.collision){ctx.beginPath();ctx.arc(x,y,9,0,Math.PI*2);ctx.strokeStyle='#e45b53';ctx.lineWidth=2;ctx.stroke()}
    const finished=k===frames.length-1,collisions=frames.slice(1,k+1).reduce((n,f)=>n+Number(f.collision),0);
    panel.querySelector('.outcome').textContent=finished?(maze.system.summary.outcome==='goal'?'Exit reached':'Step limit reached'):(k===0?'Ready':frame.collision?'Wall encountered':'Exploring');
    panel.querySelector('.metric').textContent=`${k} attempts · ${collisions} wall hits`;
    directions.forEach((key,n)=>{const value=frame.probabilities[key],bar=panel.querySelector('.bars').children[n];bar.querySelector('.value').textContent=value===undefined?'':`${(value*100).toFixed(1)}%`;bar.querySelector('.fill').style.width=`${(value||0)*100}%`});
  });
  document.querySelector('#clock').textContent=`Replay ${Math.round(current/maximum*100)}%`;
}
function animate(now){if(playing){tick=Math.min(maximum,tick+(now-last)/1000*(maximum/24));draw();if(tick>=maximum){playing=false;document.querySelector('#play').textContent='Play'}}last=now;requestAnimationFrame(animate)}
document.querySelector('#play').onclick=()=>{if(tick>=maximum)tick=0;playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};
document.querySelector('#restart').onclick=()=>{tick=0;playing=false;document.querySelector('#play').textContent='Play';draw()};
timeline.oninput=()=>{tick=Number(timeline.value);draw()};
window.seekReplay=value=>{playing=false;tick=Math.max(0,Math.min(maximum,value));draw()};
start().catch(e=>{document.querySelector('header p').textContent=`Unable to load replay: ${e.message}`;console.error(e)});
