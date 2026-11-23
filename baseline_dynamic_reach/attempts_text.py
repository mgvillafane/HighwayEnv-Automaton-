"""The written part of every attempt page (attempts.py): its idea, approach, status and analysis.

B0-B5 and P1-P6 are written out by hand from EXPERIMENT_NOTES.md (each prediction there was written before its
run). Step 8 pipelines are composed from what their order and reward terms were for, the stage they belonged to,
and numbers computed from the saved evaluations, with hand-written notes for the finalists.
"""
import math
import statistics

# --- Steps 1-7 ----------------------------------------------------------------------------------------------------
B_P = {
    "B0": {
        "title": "success only (the root)",
        "status": "Root of the chain. Learns to reach by driving at full speed; crashes in about 44% of episodes.",
        "status_short": "root",
        "idea": """
The simplest reward that equals what is scored: +50 for reaching the slot, nothing else. Every later arm changes
exactly one setting of its parent, so every effect can be traced to one change. Marcos's reward has more terms; each
can be added alone later.

*Prediction (written before the run):* it learns, with 35–60% success.""",
        "approach": "Fresh weights, the full task, 1M transitions, seeds 7, 8, 9 on the Windows PC.",
        "analysis": """
- **Prediction held:** 55% success.
- **Why it rams.** With only a success reward, a crash costs exactly what a timeout costs (0), and the discount pays
  more for arriving early. So the best policy is full speed toward the slot. That is what B0 learned: move to the
  slot's lane, accelerate to 25 m/s, never brake.
- **Every crash is a rear-end:** on FRONT into the anchor itself (the slot is 35 m past it, in its own lane),
  elsewhere into the 12 m/s lead car about 2.6 s in.
- FRONT is reached in 0–1 of 62 episodes.""",
    },
    "B1": {
        "title": "crash penalty, from fresh weights",
        "status": "Collapsed: 1% success, 98% timeouts. Kept as the evidence for learning to reach before adding a crash cost.",
        "status_short": "collapse",
        "idea": """
B0 fails only by crashing, and its reward scores a crash like a timeout. One change: a crash costs as much as the
goal (−50).

*Prediction:* crashes under 20%, success at least B0's.""",
        "approach": "B0's reward plus the crash cost, from fresh weights; full task, 1M transitions, seeds 7, 8, 9.",
        "analysis": """
- **Collapse.** From a near-random start a crash is about 7 times likelier than success, so trying is worth about
  0.115 × 50 − 0.79 × 50 ≈ −34, while slowing to 10 m/s and timing out is worth 0.
- Training success stays at 0 for more than 900k of the 1M transitions. The policy keeps exploring (entropy 0.84–1.04
  of 1.61 nats), but random actions from 10 m/s never reach a slot.
- **The lesson used ever since** (P4, and round 1 of every Step 8 pipeline): learn to reach with success as the only
  reward first, and add the crash cost from that policy.""",
    },
    "B2": {
        "title": "crash penalty, from B0's trained policy",
        "status": "No collapse; crashes fall from 44% to 16%, but only by giving up FRONT.",
        "status_short": "gives up FRONT",
        "idea": """
The same reward as B1, but starting from B0's trained policy (55% success). From there trying is worth about +5 from
the first update, so the crash cost should not make giving up the best option. One change from B1: the starting
weights.

*Prediction:* no collapse; crashes under 20%; BACK/LEFT/RIGHT at least B0's.""",
        "approach": "B1's reward, starting from B0's trained policy (same seed); full task, 1M transitions.",
        "analysis": """
- **No collapse**, and crashes fall from 44% to 16%.
- **The whole drop comes from FRONT:** there every B0 attempt cost −50, so the policy stopped pursuing FRONT. All 62
  FRONT episodes time out.
- **BACK/LEFT/RIGHT crashes are unchanged.** At ±50, a gamble that wins about half the time is worth about as much as
  giving up, so nothing pushes the policy to brake behind a slower car.""",
    },
    "B3": {
        "title": "potential-based progress shaping",
        "status": "FRONT still 0; BACK/LEFT/RIGHT equal to the scripted driver's successes, with many more crashes.",
        "status_short": "no FRONT",
        "idea": """
Per-decision credit for closing the distance to the slot, as a potential: 0.1 × (0.99 Φ′ − Φ), Φ = −distance. It
telescopes, so it changes how fast credit arrives but cannot change which policy is best. It was the cheapest test of
whether FRONT lacked only a learning signal.

*Prediction:* FRONT reached in at least 5 of 62 episodes on two seeds.""",
        "approach": "B2 plus the progress term, starting from B0's trained policy; full task, 1M transitions.",
        "analysis": """
- **FRONT stays at 0.** FRONT does not lack credit; it lacks a policy that passes the anchor.
- BACK/LEFT/RIGHT: 112–113 of 138 reached, the scripted driver's 113. But B3 crashes in all the others, where the
  scripted driver waits.""",
    },
    "B3m": {
        "title": "B3 trained again on the Mac (the noise floor)",
        "status": "Within 1 point of B3; the same 24 episodes fail on all seeds. Sets the noise floor.",
        "status_short": "noise floor",
        "idea": """
Work moved to the Mac, and `ivcheck` refuses a Mac child of a PC parent, so B3 was trained again there. It is also a
measurement of how much a result moves when nothing but the machine changes.

*Prediction:* within 5 points of B3.""",
        "approach": "B3's settings exactly, trained on the Mac (declared IV `platform`).",
        "analysis": """
- **Within 1 point of B3.** The same 24 episodes fail on all three seeds.
- **The noise floor:** repeating B3 changes 3–4 of the 138 BACK/LEFT/RIGHT episodes per seed. An effect is credible
  only if it is larger than about 5 episodes per seed and has the same sign on all three seeds.
- Evaluation is identical on both machines; training differs within this floor.""",
    },
    "B4": {
        "title": "crash penalty ×3 (50 → 150)",
        "status": "Failed: crashes unchanged. The critic learned the price; the policy did not brake.",
        "status_short": "no effect",
        "idea": """
Step 5's diagnosis: every BACK/LEFT/RIGHT failure is a crash into a slower car in the slot's lane. At ±50, ramming
(which wins about half the time) and waiting are worth about the same. At −150, ramming is worth about −40 against 0
for waiting, so the policy should learn to hold back.

*Prediction:* BACK/LEFT/RIGHT crashes at most 10 per seed, success 105–116.""",
        "approach": "B3m with the crash cost tripled, starting from B0's trained policy; full task, 1M transitions, Mac.",
        "analysis": """
- **Failed.** Crashes on BACK/LEFT/RIGHT are 25, 28 and 29 per seed (B3m: 27, 25, 28). Seed 9 lost 21 RIGHT successes to
  timeouts without crashing less.
- **The critic learned the price**: its value in blocked episodes fell from about 0 to −5, −50 and −27, and lower just
  before ramming. But the policy's probability of SLOWER behind the blocking car did not rise (7%, 0.5%, 7%).
- **The better outcome is almost never tried:** about 30 of the ~6,300 BACK/LEFT/RIGHT training episodes per seed end
  with ego waiting, against about 1,460 crashes.
- Found later (Step 7a): the shared gradient clip throttled every update, and tripling the penalty tripled the value
  gradient that caused it.""",
    },
    "B5": {
        "title": "reward scaled by 1/50 (fixes a training throttle)",
        "status": "The throttle is fixed, outcomes unchanged. Every later run uses this scale.",
        "status_short": "fixes training",
        "idea": """
The Step 7a audit found that one gradient-norm clip (0.5) covers the policy and value networks together. With rewards
of ±50, the value gradient is 30–120 times the policy gradient, so every update was scaled down to 1–4% (median KL
≈ 0.001, clip fraction ≈ 0). Multiplying the whole reward by a positive constant cannot change the best policy, so
scale 0.02 (goal = +1) removes the throttle without changing the task. B0's critic output is multiplied by 0.02 at
load, so the starting policy is identical.

*Prediction:* normal-sized updates (KL 0.005–0.02); BACK/LEFT/RIGHT crashes below B3m's on two seeds.""",
        "approach": "B3m with the reward multiplied by 0.02, starting from B0's trained policy; full task, 1M transitions, Mac.",
        "analysis": """
- **The mechanism is fixed:** the value network's share of the gradient falls from 100% to 3–19%, and updates are 5
  times larger (still below the predicted range).
- **Outcomes are unchanged:** 113 / 25 / 0 on BACK/LEFT/RIGHT on every seed. The throttle was real but was not what
  held the policy in place.
- **Reading:** every arm from B0 to B5 reaches essentially the same episodes. About 25 BACK/LEFT/RIGHT episodes need a
  pass, and no arm learned one. That led to Step 7's passing practice.""",
    },
    "P1": {
        "title": "passing practice, from B0's policy",
        "status": "Ends by waiting. One seed learned to pass, then gave it up.",
        "status_short": "gives up",
        "idea": """
The user's hypothesis: the training mix is too diverse. "Go to the slot's lane and accelerate" is right in most
episodes, and that habit traps PPO in the few that need a pass. So train only on passing situations: blocked
BACK/LEFT/RIGHT episodes, with the blocking car held in its lane, so waiting never clears the way.

*Prediction:* it stops ramming first (crashes from 99% to under 30%, mostly into timeouts); success 0–15%.""",
        "approach": "B5's reward on the practice task, starting from B0's trained policy; 1M transitions, Mac.",
        "analysis": """
- **It ends by waiting** behind the blocking car (about 0 / 7 / 193 of the 200 practice episodes).
- **But seed 8 learned to pass, then gave it up.** Its training success rose to 41% at 60k transitions (its 50k
  checkpoint really passes in 58 of 200 practice episodes), then switched to waiting within 30k transitions.
- **Why:** with +1 / −1, "30% reached, 70% crash" is worth about −0.4, and waiting is worth 0. Discounting makes it
  worse: a failed pass crashes at about 2–3 s (weight 0.74), a pass succeeds at about 8 s (0.45), so trying beats waiting
  only above about 62% success. B1's trap again.
- The hypothesis held in part: on passing-only data PPO *can* find the pass. What stops it is the reward.""",
    },
    "P1s": {
        "title": "passing practice, from fresh weights",
        "status": "Collapsed to waiting, as predicted.",
        "status_short": "collapse",
        "idea": """
The same practice as P1, from fresh weights, as the user asked to compare. From random weights trying is worth about
0.04 − 0.87 ≈ −0.8 against 0 for waiting: B1's situation.

*Prediction:* collapse, with at least 80% timeouts and at most 5% success on two seeds.""",
        "approach": "P1 with fresh weights; practice task, 1M transitions, Mac.",
        "analysis": """
- **Collapse, as predicted:** 0 reached, 0–1 crashes, 199–200 timeouts of 200 practice episodes.
- It found at most 10 successes in training, all in the first 100k transitions.
- Starting from B0 helped P1 explore (one seed found the pass); fresh weights never did.""",
    },
    "P2": {
        "title": "practice without progress shaping",
        "status": "Collapsed like P1s; progress shaping is dropped from here (no measured effect).",
        "status_short": "collapse",
        "idea": """
The user asked to test one hypothesis at a time, to stop starting from B0's weights, and to check first whether the
progress reward is needed. One change from P1s: progress 0.1 → 0, so the reward is only +1 reach / −1 crash.

*Prediction:* it also ends by waiting, which would be a floor effect: it could not show whether progress matters.""",
        "approach": "P1s without the progress term; practice task, fresh weights, 1M transitions, Mac.",
        "analysis": """
- **Same as P1s:** 0 reached on every seed.
- **The collapse is fast:** within about 20k transitions (25 PPO updates) crashes go from 80% to 0% and timeouts to 100%.
  From random weights a crash is 15–20 times likelier than a success, so stopping is learned long before a pass.
- Progress shaping neither prevents nor causes this collapse, so it is left out from here (simpler).""",
    },
    "P3": {
        "title": "practice with easy starts",
        "status": "Failed: not even the easy starts are reached; gives up from the start.",
        "status_short": "gives up",
        "idea": """
Change where practice starts, not the reward: ego starts in the next lane 10 m past, level with, or 15 m behind the
blocking car, or at its natural start (one of four at random). From the easy starts, early attempts should succeed
more often than the ~62% break-even.

*Prediction:* "past" and "beside" starts reached in at least 70%; natural start at least 15% on two seeds.""",
        "approach": "P2 with mixed starts; practice task, fresh weights, +1 / −1, 1M transitions, Mac.",
        "analysis": """
- **All three predictions failed:** 0 reached at the end, even from the easy starts. Timeouts reach 82–87% by 100k.
- **Easy for a sensible driver, not for a random one.** On the same episodes the scripted driver reaches "past" 33 of 41
  and "beside" 45 of 55; random actions reach 9 and 12, and crash in about 70% (random lane changes into traffic).
- **Reading:** giving up is set by the start of learning, not by where episodes start. With crash = −goal from fresh
  weights, stopping is learned first, everywhere. The way out was already known: B0 → B2.""",
    },
    "P4": {
        "title": "practice with easy starts, success only",
        "status": "Passes: 34–50% of natural-start practice episodes (scripted driver 2.5%), but crashes in most others.",
        "status_short": "learns to pass",
        "idea": """
Repeat B0 → B2 on the practice task: first learn to pass with success as the only reward (a crash simply ends the
episode with 0), then add the crash cost from that policy (P5). One change from P3: crash cost 1 → 0.

*Prediction:* no collapse; easy starts reached in at least 50%; natural start at least 10% on two seeds; crashes at
least 30%.""",
        "approach": "P3 without the crash cost; practice with easy starts, fresh weights, 1M transitions, Mac.",
        "analysis": """
- **All four predictions held.** No collapse: training success rose steadily, with 0–1% timeouts.
- **Natural-start practice episodes: 34–50.5% reached on every seed**, against 0 for P1–P3 and 2.5% for the scripted
  driver. All but one success per seed is a real pass (ego more than 5 m ahead of every blocking car).
- Crashes 49.5–66%, mostly into the blocking car.
- **The deterministic policy swings** between ~0 and ~110 reached from one 50k checkpoint to the next while training is
  smooth: the arg-max action flips between "pass" and "ram" in many episodes at once.
- **Reading:** without a crash cost, trying is never worse than waiting, so attempts continue and passes are learned.""",
    },
    "P5": {
        "title": "practice with the crash cost, from P4's policy",
        "status": "Keeps passing (62–64% on two seeds); on the full task it beats the scripted driver on BACK/LEFT/RIGHT.",
        "status_short": "passes, safer",
        "idea": """
B1 → B2 on the practice task: P3's reward (+1 / −1), starting from P4's passing policy. One change from P3: the
starting weights.

*Prediction:* no collapse (at least 20% on two seeds); crashes fall below P4's; it may wait in some natural starts.""",
        "approach": "P3's reward, starting from P4's trained policy (same seed); practice with easy starts, 1M transitions, Mac.",
        "analysis": """
- **No collapse:** seeds 7 and 8 reach 62–64% of the natural-start practice episodes (P4: 44–50.5%). The crash cost
  turned into more success, not waiting.
- Seed 9's final model is a dip: its checkpoints swing between 6 and 104 reached.
- **On the full task, never trained on:** seeds 7 and 8 reach 119 and 121 of 138 BACK/LEFT/RIGHT, the first arms above
  the scripted driver's 113.
- **FRONT opens** (21–27 of 62 on seeds 7 and 8): a FRONT slot is a pass of the anchor itself, so the skill carries over.
- Remaining failure: ramming the blocking car.""",
    },
    "P6": {
        "title": "the full task from P5's passing policy",
        "status": "First policies above the scripted driver on BACK/LEFT/RIGHT on every seed; its order became Step 8's order O.",
        "status_short": "beats scripted on B/L/R",
        "idea": """
Fine-tune the passing policy on the normal mix of episodes, so it learns when a pass is needed and when to drive
straight to the slot, with FRONT back in the data. Parent B5 (the full task, +1 / −1, progress 0.1); one change: start
from P5's policy instead of B0's.

*Prediction:* it keeps the pass (blocked episodes at least 27 of 45); BACK/LEFT/RIGHT at least 115 of 138; FRONT at
least 20 of 62 on two seeds.""",
        "approach": "B5's reward on the full task, starting from P5's trained policy (P4 → P5 → P6, 1M each), Mac.",
        "analysis": """
- **All five predictions held.** Blocked episodes 42, 33, 33 of 45; BACK/LEFT/RIGHT 134, 125, 123 of 138 (scripted 113);
  FRONT 48, 41, 20 of 62 (scripted 40).
- **On held-out episodes (Step 7h)** its own level held (within ±1.6 points), but only seed 7 beat the scripted driver
  beyond noise; seeds 8 and 9 crash 5–6 times as often.
- **Under Step 8's score** (reached − crash), P6 is level with the scripted driver at best: −0.097 on the 8d final test
  (p 0.40), because it crashes in 16% of episodes against 2%.
- Its pipeline (practice, practice with the crash cost, full task) became Step 8's order O.""",
    },
}

# --- Step 8 --------------------------------------------------------------------------------------------------------
ORDER_IDEA = {
    "N": "**Order N, no practice.** All three rounds are on the full task. Round 1 is success only (B0's lesson: learn to "
         "reach before any crash cost); rounds 2 and 3 use the reward under test. It asks whether practice is needed "
         "at all.",
    "H": "**Order H, practice from the natural start.** Round 1 is success only on practice episodes (blocked "
         "BACK/LEFT/RIGHT, blocker held), with no easy starts; rounds 2 and 3 are the full task.",
    "E": "**Order E, practice with easy starts.** Round 1 is success only on P4's practice task (easy starts); rounds 2 "
         "and 3 are the full task. P4 → P6 without P5's second practice round.",
    "O": "**Order O, the original order.** P4 → P5 → P6 at 2M per round: round 1 practice with easy starts, success only "
         "(P4); round 2 the same practice with the reward under test (P5); round 3 the full task (P6).",
}
CRASH_IDEA = {
    0.0: "**Crash cost 0:** success only in every round, the control. Discounting alone pays for arriving early, so it "
         "predicts fast driving and many crashes.",
    0.5: "**Crash cost 0.5 × goal.** Discounting already makes crash 1 more cautious than the scored reached − crash (a "
         "crash at ~2 s weighs 0.74, a success at ~8 s 0.45), so half the cost may score better.",
    1.0: "**Crash cost 1 × goal**, P6's level: a crash costs as much as a goal.",
    2.0: "**Crash cost 2 × goal**, the safety direction.",
}
PROGRESS_IDEA = ("**Progress shaping 0.1 per m** (potential-based; it cannot change the best policy). P6 had it by "
                 "inheritance and it was never tested on this path.")
TIME_IDEA = {
    0.001: "**Time cost 0.001 per decision** (Marcos's time term at the 1/50 scale): a whole timed-out episode costs "
           "0.25 goals. Added in 8c2 because every crash cost made runs give up on FRONT.",
    0.004: "**Time cost 0.004 per decision:** a whole timed-out episode costs 1 goal, the largest level that keeps "
           "waiting cheaper than the smallest crash cost tested (0.5). Aimed at the FRONT give-up.",
}
WIDTH_IDEA = ("**256-256 networks** instead of 64-64 (about 160k weights instead of 16k). The user's rule: a larger "
              "model only for a large measured gain.")
STAGE_IDEA = {
    "Step 8a": "Step 8a tested network width at P6's reward on order E (3 seeds). The bar, written before: 256 is adopted "
               "only if it adds +0.10 on all 3 seeds with no more crashes.",
    "Step 8b": "Step 8b compared the four training orders at P6's reward (crash 1, progress 0.1), 64-64, 3 seeds.",
    "Step 8c": "Step 8c was the reward grid on orders N and O: crash 0 / 0.5 / 1 / 2 × progress 0 / 0.1, 3 seeds. The "
               "rule picks the simplest reward that the best one does not beat beyond noise.",
    "Step 8c2": "Step 8c2 added a time cost, because 8c measured that every crash cost makes runs give up on FRONT "
                "(timeouts of 10% or more).",
    "Step 8e": "Step 8e trained three finalists 256 wide on all 8 seeds, with a fourth round of 3M; Step 8f then tested "
               "the best on 500 fresh episodes.",
}
STAGE_READING = {
    "Step 8a": "Across 8a: 256 − 64 = +0.067 (p 0.45, ahead on 2 of 3 seeds), so 64-64 stayed. The 64-wide pipeline's "
               "weak seed (10) never learned to pass in round 1; 256-wide gave up on no seed.",
    "Step 8b": "Across 8b: all four orders lie within 0.12 of each other and no one-factor difference is beyond noise "
               "with 3 seeds. The rule chose N (simplest); O scored highest. What practice changed was FRONT: reached "
               "54–55% with practice against 40% without.",
    "Step 8c": "Across 8c: any crash cost brings the FRONT give-up, and success only never does. The score swaps crashes "
               "for timeouts almost one for one. Nothing beat success only by the rule with 3 seeds; P6's reward scored "
               "highest in both orders.",
    "Step 8c2": "Across 8c2: a time cost moves the give-up but does not remove the trade. Where t4 ends it (crash 0.5), "
                "part of the timeouts become reaches and part crashes. No time-cost cell is beyond noise of its "
                "no-time-cost cell.",
    "Step 8e": "Across 8e: by the bars written before, nothing changed on average (+0.06 to +0.13 for 256, not beyond "
               "noise). But 256 removed the FRONT give-up (13 → 2 of 24 runs) and made practice work on every seed.",
}
FINALIST_NOTES = {
    "N_c05w0t1": """
**The 8d winner (until 8f).** On the 5 confirmation seeds it scored highest (+0.633), beat success only (+0.162,
p 0.002, 5 of 5), and the tie-break among equally simple finalists went to it. On the 8d final test (500 fresh
episodes) it scored +0.628, below the scripted driver beyond noise (−0.146, p 0.029). The whole gap was FRONT: it
reaches 39% of FRONT episodes and gives up on 43%. On the 8f fresh episodes it scored +0.614 and lost to the 256-wide
original order by −0.177.""",
    "O_c1w0t4": """
**Best of the original order at 64 wide.** It tied the winner on the confirmation seeds (+0.632 against +0.633) and
scored highest on the 8d final test (+0.694, level with the scripted driver: −0.081, p 0.31). Seeds 10, 16 and 28
never learned to pass in practice and give up on FRONT. It is the recipe that became the adopted one at 256 wide.""",
    "O_c1w01": """
**The P4 → P5 → P6 storyline retrained on the PC** at 2M per round, with P6's reward. Final test +0.672 (level with
the scripted driver, −0.102, p 0.22), above P6 itself (+0.677 on the same episodes, from 1M per round on the Mac).""",
    "N_c1w01": "**P6's reward without practice.** An 8d finalist; +0.561 on the confirmation seeds.",
    "N_c1w0t4": "**The winner's order with O_c1w0t4's reward:** it links the two by one factor each. An 8d finalist.",
    "N_c0w0": """
**Success only, no practice.** The rule's choice after the 3-seed screening (nothing beat it beyond noise), but on
the 5 confirmation seeds every crash-cost reward beat it in order N (+0.09 to +0.16, p 0.002–0.049). It never gives
up and crashes in about 25% of episodes.""",
    "O_c0w0": "**Success only, original order.** Never gives up, crashes in about 25%; +0.507 on the confirmation seeds.",
    "E_c1w01_n256": """
**The first 256-wide pipeline (8a).** +0.752 against +0.685 at 64-64, but not beyond noise (p 0.45) and not ahead on
every seed, so 64 stayed. Every seed landed between +0.71 and +0.80, and no seed gave up on FRONT: the first sign of
what 8e confirmed.""",
    "N_c05w0t1_n256": """
**The winner's recipe at 256 wide (8e).** 256 removes the FRONT give-up here too (FRONT timeouts 34% → 5% at 9M), but
without practice it turns it into crashes: on the 8f fresh episodes it scores +0.573, below the same recipe at 64
(−0.041), with crashes +7 points.""",
    "N_c1w0t4_n256": "**No practice, crash 1 + time cost, 256 wide (8e).** Links the winner and the adopted recipe by one factor each.",
    "O_c1w0t4_n256": """
**The adopted recipe (Step 8f).** On 500 fresh episodes (60,000–60,499) it beat the 64-wide winner by +0.177 (p 0.006,
ahead on 8 of 8 seeds), crashed no more (−1.8 points), and no seed gave up on FRONT: all three conditions of the bar
written before the test. It is level with the scripted driver (+0.033, p 0.16): it reaches far more (89.5% against
78.2%) but crashes about four times as often (10.4% against 2.4%), mostly on FRONT.

**Why it works.** 256 wide helps only together with practice: within order O it fixes the three seeds whose 64-wide
practice never learned to pass (+0.35 each) and changes little on the others. At 256 the practice rounds alone already
reach +0.795 on the full task at 4M; the full-task round adds nothing more, and a fourth round (9M) neither (−0.008).

**What is left (Step 8g).** 91% of its FRONT crashes are rear-ends at full speed (25 m/s, 12 m/s faster than the car
hit), with SLOWER chosen in the last 2 s in only 2%. It passes, but does not brake.

**Consecutive goals (Step 9).** When the goal in the observation is replaced by the next one after each reached slot,
later goals score about 0.4 below the first (+0.78 → about +0.37), in every condition and on every seed. Part of
that is not the model's: with a new car the scripted driver drops too (+0.73 → +0.48), by timing out more, so the
model's own extra drop is about 0.16, and it is all crashes (11% → 31%; scripted 2.5%). Around one car the scripted
driver does better on later goals, so there the whole drop is the model's. It reaches slots at 25 m/s, 11 m/s faster
than the anchor, so the next goal starts in a rear-end course: 63% of later-goal crashes come within 3 s of the
switch. Around one car it also cannot drop back to a slot behind it. On single goals it is unchanged: goal 1 of
every chain matches the ordinary episode exactly. Details:
[`results/chains/summary.md`](../results/chains/summary.md).""",
}


def _compare_lines(attempt):
    """One-factor comparisons on the 400 selection episodes, paired over the seeds both pipelines have."""
    from baseline_dynamic_reach.config import run_name, step8_name
    from baseline_dynamic_reach.grid import compare, load_stats, tags
    order, reward, width = attempt["order"], attempt["reward"], attempt["width"]
    me = attempt["final"]
    pairs = []
    if width:
        pairs.append((step8_name(order, 3, reward), "width", "the same pipeline at 64-64"))
    else:
        if reward != "c0w0" and order in "NO":
            pairs.append((step8_name(order, 3, "c0w0"), "reward", "success only, same order"))
        if reward != "c1w01" and order in "NO":
            pairs.append((step8_name(order, 3, "c1w01"), "reward", "P6's reward, same order"))
        other = {"N": "O", "O": "N"}.get(order)
        if other:
            pairs.append((step8_name(other, 3, reward), "order", f"order {other}, same reward"))
        if order in "HE":
            pairs.append((step8_name("N", 3, reward), "practice", "order N (no practice), same reward"))
    lines = []
    seeds = attempt["seeds"]
    for other_arm, factor, label in pairs:
        stats = load_stats([me, other_arm], seeds, tags())
        try:
            result = compare(me, other_arm, factor, seeds, stats)
        except Exception as error:  # noqa: BLE001 - a refused pair is reported, not hidden
            lines.append(f"- Against {label} (`{other_arm}`): not compared ({error.__class__.__name__}).")
            continue
        if result is None:
            continue
        lines.append(f"- Against {label} (`{other_arm}`): {result['mean']:+.3f} (95% interval {result['ci95'][0]:+.3f} to "
                     f"{result['ci95'][1]:+.3f}, p {result['p']:.2f}, ahead on {result['ahead']} of {result['n']} seeds; "
                     f"crash rate {100 * result['crash_change']:+.1f} points).")
    return lines


def _rank_line(attempt):
    from baseline_dynamic_reach.attempts import attempts as all_attempts, selection_stats
    from baseline_dynamic_reach.config import STEP8_SCREEN_SEEDS, run_name
    peers = [a for a in all_attempts() if a["kind"] == "S8" and a["width"] == attempt["width"]]

    def screen_score(a):
        values = [selection_stats(run_name(a["final"], s)) for s in STEP8_SCREEN_SEEDS]
        values = [v["all"]["score"] for v in values if v]
        return statistics.fmean(values) if values else -math.inf

    ranked = sorted(peers, key=screen_score, reverse=True)
    position = [a["id"] for a in ranked].index(attempt["id"]) + 1
    width = "256-256" if attempt["width"] else "64-64"
    best = "" if position == 1 else f"; the best is {ranked[0]['id']}, {screen_score(ranked[0]):+.3f}"
    return (f"- On the screening seeds (7, 10, 13) it ranks {position} of the {len(ranked)} {width} pipelines "
            f"(score {screen_score(attempt):+.3f}{best}).")


def _step8_text(attempt):
    from baseline_dynamic_reach.config import CRASH_LEVELS, TIME_LEVELS, reward_parts
    order, reward, width = attempt["order"], attempt["reward"], attempt["width"]
    crash_code, progress_code, time_code = reward_parts(reward)
    idea = [ORDER_IDEA[order], "", CRASH_IDEA[CRASH_LEVELS[crash_code]], ""]
    if progress_code == "w01":
        idea += [PROGRESS_IDEA, ""]
    if time_code:
        idea += [TIME_IDEA[TIME_LEVELS[time_code]], ""]
    if width:
        idea += [WIDTH_IDEA, ""]
    idea += [STAGE_IDEA[attempt["step"]]]
    finalist = len(attempt["seeds"]) >= 8
    adopted = attempt["id"] == "O_c1w0t4_n256"
    if adopted:
        status, short = "**Adopted in Step 8f** as the Step 8 recipe.", "**adopted**"
    elif attempt["id"] == "N_c05w0t1":
        status, short = "8d winner (64-64); replaced in Step 8f by the 256-wide original order.", "8d winner"
    elif finalist and width:
        status, short = "Trained on all 8 seeds in Step 8e; not adopted.", "8e"
    elif finalist:
        status, short = "8d finalist: confirmed on 5 new seeds (8 in all); not chosen.", "8d finalist"
    else:
        status, short = f"Screened in {attempt['step']} on 3 seeds; not a finalist.", "screened"
    analysis = [_rank_line(attempt), *_compare_lines(attempt), "", STAGE_READING[attempt["step"]]]
    if attempt["id"] in FINALIST_NOTES:
        analysis += ["", FINALIST_NOTES[attempt["id"]].strip()]
    names = {"N": "no practice", "H": "practice (natural start)", "E": "practice with easy starts", "O": "original order"}
    title = f"{names[order]}, {reward}" + (", 256 wide" if width else "")
    approach = (f"Three rounds of 2M transitions on the Windows PC, each starting from the previous round's trained "
                f"policy (same seed); round 1 from fresh weights, success only."
                + (" A fourth round of 3M (9M in all) continued round 3 in Step 8e." if attempt.get("long") else ""))
    return {"title": title, "status": status, "status_short": short, "idea": "\n".join(idea), "approach": approach,
            "analysis": "\n".join(analysis)}


def text_for(attempt):
    if attempt["kind"] == "BP":
        return B_P[attempt["id"]]
    return _step8_text(attempt)
