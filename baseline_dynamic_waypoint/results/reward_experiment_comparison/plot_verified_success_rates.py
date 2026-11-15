"""Show evaluation and training separately so rare training successes remain visible."""
import csv
import json
from pathlib import Path
import shutil

from plot_success_rates import CASES, GOALS, ROOT, OUT
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42, "svg.fonttype":"none"})
    evaluations, training, records = [], [], []
    for label, name, filename, steps, _, _ in CASES:
        source = ROOT/name/filename
        stats = json.loads(source.read_text())["ppo"]
        e_rows = [r for r in read_rows(source.with_name(source.stem+"_episodes.csv")) if r["policy"]=="ppo"]
        t_rows = read_rows(ROOT/name/"training_episodes.csv")
        assert len(e_rows)==100
        assert [int(r["seed"]) for r in e_rows]==list(range(10000,10100))
        one_eval, one_train = [], []
        for goal in GOALS:
            values=[]
            for kind, rows in [("evaluation",e_rows),("training",t_rows)]:
                subset=[r for r in rows if r["slot"]==goal]
                n=sum(int(r["success"]) for r in subset)
                rate=100*n/len(subset)
                values.append((n,len(subset),rate))
                records.append(dict(experiment=label,run=name,metric=kind,goal=goal,
                                    successes=n,episodes=len(subset),percent=rate,
                                    source=str(source if kind=="evaluation" else ROOT/name/"training_episodes.csv")))
                if kind=="evaluation":assert abs(rate/100-stats["success_rate_by_slot"][goal])<1e-12
            one_eval.append(values[0]);one_train.append(values[1])
        evaluations.append(one_eval);training.append(one_train)
    assert training[1][1][:2]==(1,1775) and training[1][3][:2]==(1,1757)
    assert evaluations[1][1][0]==evaluations[1][3][0]==0

    fig,axes=plt.subplots(1,2,figsize=(16,8.3),facecolor="white")
    fig.subplots_adjust(left=.255,right=.97,top=.77,bottom=.30,wspace=.12)
    cmap=LinearSegmentedColormap.from_list("success",["#F0F5F6","#77B8BF","#087E8B"])
    labels=["Simple reward\npretrained + 200k", "Timeout + dropout\n53 inputs; stopped run",
            "Route progress + tracking\nfinal, 2M steps", "No tracking + FRONT side bonus\nfinal, 1.5M steps"]
    for panel,(ax,data,title) in enumerate(zip(axes,[evaluations,training],
          ["DETERMINISTIC EVALUATION\nFinal/latest recorded checkpoint",
           "TRAINING HISTORY\nAll completed episodes in each run"])):
        matrix=np.array([[item[2] for item in row] for row in data])
        ax.imshow(matrix,cmap=cmap,vmin=0,vmax=100,aspect="auto")
        ax.set_title(title,fontsize=13,weight="bold",color="#233643",pad=21,linespacing=1.6)
        ax.set_xticks(range(4),[g.upper() for g in GOALS],fontsize=11,weight="bold",color="#233643")
        ax.set_yticks(range(4),labels if panel==0 else [""]*4,fontsize=11,color="#233643")
        ax.tick_params(axis="both",length=0,pad=13)
        for spine in ax.spines.values():spine.set_visible(False)
        for row in range(4):
            for col in range(4):
                n,total,percent=data[row][col]
                shown="100%" if percent==100 else "0%" if percent==0 else f"{percent:.3f}%" if percent<.1 else f"{percent:.1f}%"
                color="white" if percent>=65 else "#233643"
                ax.text(col,row-.105,shown,ha="center",va="center",fontsize=13.5,weight="bold",color=color)
                ax.text(col,row+.16,f"{n:,} / {total:,}",ha="center",va="center",fontsize=9.8,color=color)
        ax.set_xticks(np.arange(-.5,4,1),minor=True)
        ax.set_yticks(np.arange(-.5,4,1),minor=True)
        ax.grid(which="minor",color="white",linewidth=3)
        ax.tick_params(which="minor",length=0)
    for column in (1,3):axes[1].add_patch(Rectangle((column-.5,.5),1,1,fill=False,edgecolor="#D28D24",linewidth=3))
    fig.text(.055,.958,"Successes during training and evaluation",fontsize=23,weight="bold",color="#233643",va="top")
    fig.text(.055,.902,"Same four experiments; percentages and success counts are shown separately for the two measurements.",
             fontsize=12,color="#536A78")
    cax=fig.add_axes([.735,.216,.23,.015])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,100),cmap=cmap),cax=cax,orientation="horizontal",ticks=[0,50,100])
    cb.outline.set_visible(False);cb.ax.set_xticklabels(["0%","50%","100%"],fontsize=9)
    fig.text(.055,.225,"The timeout/dropout run did have rare FRONT and RIGHT successes during training.",fontsize=12,weight="bold",color="#233643")
    fig.text(.055,.181,"FRONT: 1/1,757 (0.057%). RIGHT: 1/1,775 (0.056%). Neither succeeded in its deterministic evaluation cases.",
             fontsize=10.5,color="#536A78")
    fig.text(.055,.137,"Evaluation uses monitoring seeds 10000–10099. The interrupted run's latest recorded evaluation is at 1.25M steps.",
             fontsize=10,color="#536A78")
    fig.text(.055,.104,"Training pools sampled-action episodes across learning; it does not measure the final policy. Budgets and initialization differ.",
             fontsize=10,color="#536A78")
    fig.text(.055,.071,"Other 57-input timeout/dropout variants reached 4/22 RIGHT evaluation successes (18.2%); their FRONT evaluation remained 0/18.",
             fontsize=10,color="#536A78")
    fig.text(.055,.038,"At matched 1.5M steps, both route variants evaluated at LEFT/RIGHT/BACK 100% and FRONT 77.8%; collisions were 0 vs 4/100.",
             fontsize=10,color="#536A78")
    for ext in ("png","pdf","svg"):
        old=OUT/f"success_by_goal.{ext}"
        archive=OUT/f"success_by_goal_evaluation_only.{ext}"
        if old.exists() and not archive.exists():shutil.copy2(old,archive)
        target=OUT/f"success_training_vs_evaluation.{ext}"
        fig.savefig(target,dpi=180,facecolor="white")
        shutil.copy2(target,old)
    plt.close(fig)
    with (OUT/"success_training_vs_evaluation.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    audit=[]
    for directory in ROOT.iterdir():
        try:config=json.loads((directory/"run.json").read_text())
        except (OSError,ValueError):continue
        weights=config.get("environment",{}).get("reward_weights",{})
        dropout=config.get("dropout_rate",config.get("ppo_config",{}).get("policy_kwargs",{}).get("dropout_rate",0))
        if not dropout or not weights.get("timeout") or weights.get("route_progress"):continue
        train_rows=read_rows(directory/"training_episodes.csv")
        eval_files=[f for f in directory.glob("evaluation*.json") if f.name!="evaluation_before.json"]
        for goal in ["right","front"]:
            t=[r for r in train_rows if r["slot"]==goal]
            measurements=[]
            for path in eval_files:
                result=json.loads(path.read_text())
                if not isinstance(result,dict) or "ppo" not in result:continue
                stats=result["ppo"]
                if goal in stats["success_rate_by_slot"]:
                    measurements.append((stats["success_rate_by_slot"][goal],path.name))
            best=max(measurements,default=(0,"none"))
            audit.append(dict(run=directory.name,dropout=dropout,timeout=weights["timeout"],goal=goal,
                              training_successes=sum(int(r["success"]) for r in t),training_episodes=len(t),
                              maximum_recorded_eval_percent=100*best[0],maximum_eval_source=best[1]))
    with (OUT/"timeout_dropout_audit.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    (OUT/"VERIFIED_COMPARISON.md").write_text(
        "# Training versus evaluation\n\n"
        "The selected 53-input timeout/dropout run had 1 FRONT and 1 RIGHT training success. "
        "Its deterministic evaluations had zero successes for both goals. "
        "The updated graph separates these measurements; the evaluation-only graph is preserved.\n\n"
        "The separate older 57-input variants had up to 18.2% RIGHT evaluation success, "
        "but no FRONT evaluation success. See timeout_dropout_audit.csv for the broader audit.\n\n"
        "Regenerate the updated graph with plot_verified_success_rates.py. "
        "plot_success_rates.py regenerates the original evaluation-only view.\n",encoding="utf-8")
    print(OUT/"success_training_vs_evaluation.png")


if __name__=="__main__":main()
