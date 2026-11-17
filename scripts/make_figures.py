"""Regenerate the four figures in docs/figures/ (English labels)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle

BG="#fcfcfb"; T1="#0b0b0b"; T2="#52514e"; MUTE="#b8b6b0"
BLUE="#2a78d6"; ORANGE="#eb6834"; RED="#e34948"
OUT=os.path.join(os.path.dirname(__file__),"..","docs","figures")+"/"

def node(ax,x,y,label,r=0.42):
    ax.add_patch(Circle((x,y),r,fc="#e8f0fb",ec=BLUE,lw=2,zorder=3))
    ax.text(x,y,label,ha="center",va="center",fontsize=11,color=T1,zorder=4)
def link(ax,a,b,dashed=False,color=BLUE):
    ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=2 if dashed else 2.5,ls="--" if dashed else "-",zorder=1)
def save(fig,name,**kw):
    for ext in ("png","svg"):
        fig.savefig(OUT+f"{name}.{ext}",dpi=150,facecolor=BG,**kw)
    plt.close(fig)

# Fig 1: topologies
fig,axs=plt.subplots(1,4,figsize=(14,4.4),facecolor=BG)
for ax in axs: ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-2,2); ax.set_ylim(-2.1,2)
ax=axs[0]; p={1:(-1,0),2:(1,0)}
link(ax,p[1],p[2]); [node(ax,*p[k],f"Spark{k}") for k in p]
ax.set_title("2 nodes: direct link",fontsize=13,color=T1)
ax=axs[1]; p={1:(0,1.1),2:(-1.1,-0.8),3:(1.1,-0.8)}
for a,b in [(1,2),(2,3),(3,1)]: link(ax,p[a],p[b])
[node(ax,*p[k],f"Spark{k}") for k in p]
ax.set_title("3 nodes: triangle (official)",fontsize=13,color=T1)
ax.text(0,-1.75,"every node is a neighbor",ha="center",fontsize=11,color=T2)
ax=axs[2]; p={1:(0,0.9),2:(-1.1,-0.8),3:(1.1,-0.8)}
for a,b in [(1,2),(1,3)]: link(ax,p[a],p[b])
[node(ax,*p[k],f"Spark{k}") for k in p]
ax.text(0,1.55,"hub",ha="center",fontsize=11,color=T2)
ax.set_title("3 nodes: star (my July setup)",fontsize=13,color=T1)
ax.text(0,-1.75,"Spark2 and Spark3 are not linked",ha="center",fontsize=11,color=T2)
ax=axs[3]; p={1:(-1,1),2:(1,1),3:(1,-1),4:(-1,-1)}
for a,b in [(1,2),(2,3),(3,4),(4,1)]: link(ax,p[a],p[b])
link(ax,p[1],p[3],True,RED); link(ax,p[2],p[4],True,RED)
[node(ax,*p[k],f"Spark{k}") for k in p]
ax.set_title("4 nodes: ring",fontsize=13,color=T1)
ax.text(0,-1.75,"red dashed: no direct link (diagonals)",ha="center",fontsize=11,color=RED)
fig.tight_layout(); save(fig,"fig1_topology")

# Fig 2: CX7 block diagram
fig,ax=plt.subplots(figsize=(11,4.2),facecolor=BG); ax.set_facecolor(BG); ax.axis("off")
ax.set_xlim(0,11); ax.set_ylim(0,4.2)
def box(x,y,w,h,txt,fc,fs=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.15",fc=fc,ec=T2,lw=1.5))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=T1)
box(0.3,0.9,2.4,2.4,"GB10\nGrace CPU +\nBlackwell GPU","#eef2f7")
box(5.4,0.9,2.0,2.4,"ConnectX-7","#e8f0fb")
for y in [2.6,1.6]:
    ax.annotate("",xy=(5.4,y),xytext=(2.7,y),arrowprops=dict(arrowstyle="<->",color=BLUE,lw=2.2))
    ax.text(4.05,y+0.15,"PCIe Gen5 x4 (~100 Gbps)",ha="center",va="bottom",fontsize=10,color=T1)
box(9.0,2.3,1.7,0.8,"QSFP port 0","#f5f5f3",fs=11)
box(9.0,1.1,1.7,0.8,"QSFP port 1","#f5f5f3",fs=11)
for y in [2.7,1.5]:
    ax.annotate("",xy=(9.0,y),xytext=(7.4,y),arrowprops=dict(arrowstyle="<->",color=T2,lw=1.6))
ax.text(9.85,3.35,"200GbE x 2 on the outside",ha="center",fontsize=10.5,color=T2)
ax.text(5.5,0.35,"One node can move at most ~200 Gbps in total over its two PCIe links (July measurement: 185.1 Gbps RDMA, both ports)",
        ha="center",fontsize=10.5,color=T1)
save(fig,"fig2_cx7_block",bbox_inches="tight")

# Fig 3: pipeline vs tensor parallel (schematic)
fig,axs=plt.subplots(2,1,figsize=(11,5.6),facecolor=BG)
rows=["Node 1","Node 2","Node 3"]
for ax,title in zip(axs,["Pipeline (llama.cpp RPC): nodes compute one after another",
                         "Tensor parallel (vLLM / SGLang TP): all nodes compute at once, all-reduce every layer"]):
    ax.set_facecolor(BG); ax.set_xlim(0,12); ax.set_ylim(-0.8,3.0)
    ax.set_yticks([2,1,0]); ax.set_yticklabels(rows,fontsize=11,color=T1)
    ax.set_xticks([]); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
    for y in [0,1,2]: ax.axhline(y,color="#e6e5e1",lw=1,zorder=0)
    ax.set_title(title,fontsize=12.5,color=T1,loc="left")
ax=axs[0]; t=0.2
for r in [2,1,0]:
    ax.barh(r,3.6,left=t,height=0.5,color=BLUE,edgecolor=BG,lw=2)
    ax.text(t+1.8,r,"its own layers",ha="center",va="center",fontsize=9.5,color="white"); t+=3.6
ax.annotate("",xy=(11.0,2.55),xytext=(0.2,2.55),arrowprops=dict(arrowstyle="<->",color=T2,lw=1.2))
ax.text(5.6,2.62,"one token",ha="center",va="bottom",fontsize=10,color=T2)
ax.text(11.6,-0.62,"time →",ha="right",fontsize=10,color=T2)
ax=axs[1]; t=0.2
for L in range(6):
    for r in [2,1,0]: ax.barh(r,0.6,left=t,height=0.5,color=BLUE,edgecolor=BG,lw=1.5)
    t+=0.6
    ax.barh(1,0.25,left=t,height=2.4,color=ORANGE,edgecolor=BG,lw=1); t+=0.25
ax.annotate("",xy=(t,2.55),xytext=(0.2,2.55),arrowprops=dict(arrowstyle="<->",color=T2,lw=1.2))
ax.text((0.2+t)/2,2.62,"one token",ha="center",va="bottom",fontsize=10,color=T2)
ax.text(0.2,-0.62,"layer 1 → layer 2 → …   blue: compute (1/3 per node), orange: all-reduce",fontsize=10,color=T2)
ax.text(11.6,-0.62,"time →",ha="right",fontsize=10,color=T2)
fig.tight_layout(); save(fig,"fig3_pp_vs_tp")

# Fig 4: NVIDIA's ring vs switch numbers
fig,ax=plt.subplots(figsize=(8,4.8),facecolor=BG); ax.set_facecolor(BG)
x=np.array([0,1]); w=0.34; ring=[32.82,40.11]; sw=[36.34,52.58]
b1=ax.bar(x-w/2-0.01,ring,w,color=BLUE,label="Ring (no switch)")
b2=ax.bar(x+w/2+0.01,sw,w,color=ORANGE,label="Switch")
for bars,vals,mult in [(b1,ring,["2.77x","3.38x"]),(b2,sw,["3.06x","4.43x"])]:
    for b,v,m in zip(bars,vals,mult):
        ax.text(b.get_x()+b.get_width()/2,v+1,f"{v:.2f}\n({m})",ha="center",va="bottom",fontsize=10,color=T1)
ax.set_xticks(x); ax.set_xticklabels(["4 nodes","8 nodes"],fontsize=12,color=T1); ax.set_ylim(0,65)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.spines["left"].set_color(MUTE); ax.spines["bottom"].set_color(MUTE)
ax.tick_params(colors=T2); ax.yaxis.grid(True,color="#ecebe7"); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=10.5,loc="upper left")
ax.set_title("Decode throughput (TPS) by node count — Qwen3.8-27B NVFP4, concurrency 1",fontsize=12,color=T1,loc="left")
fig.text(0.01,0.01,"Speedup vs. one node. Source: NVIDIA Technical Blog",fontsize=9,color=T2)
fig.tight_layout(rect=(0,0.03,1,1)); save(fig,"fig4_nodes_decode")
