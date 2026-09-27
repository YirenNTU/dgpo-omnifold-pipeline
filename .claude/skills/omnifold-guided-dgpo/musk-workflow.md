# Musk operating system for this research project

This file is mandatory whenever the `omnifold-guided-dgpo` workflow is opened.
Part 1 records what Musk is documented to do, with sources. Part 2 turns it
into rules for the OmniFold-guided DGPO study. Part 3 is the checklist that
every session and every protocol must pass. Rules in Part 2 and 3 override
convenience; they never override the research contract in `CLAUDE.md`.

---

## Part 1 — What Musk actually does (sourced)

### 1.1 The Algorithm (Isaacson, *Elon Musk*, ch. 46; also stated by Musk in the 2021 Starbase interview)

Five steps, in this order. Order is the point: the most common mistake is to
optimize or automate something that should have been deleted.

1. **Question every requirement.** "Each should come with the name of the
   person who made it." A requirement from a department cannot be
   interrogated; a requirement from a person can. Requirements from "smart
   people" are the most dangerous because nobody questions them.
2. **Delete any part or process you can.** "You may have to add them back
   later." If you do not end up adding back at least 10% of what you deleted,
   you did not delete enough.
3. **Simplify and optimize.** Only after step 2. "Possibly the most common
   error of a smart engineer is to optimize something that should not exist."
4. **Accelerate cycle time.** "Every process can be speeded up." But only
   after steps 1–3.
5. **Automate.** "That comes last."

Corollaries recorded in the same passage:

- "All technical managers must have hands-on experience."
- "Comradery is dangerous. It makes it hard for people to challenge each
  other's work."
- "It's OK to be wrong. Just don't be confident and wrong."
- "Never ask your troops to do something you're not willing to do."
- "Whenever there are problems to solve, don't just meet with your managers.
  Do a skip level, where you meet with the level right below your managers."
- "When hiring, look for people with the right attitude. Skills can be taught."
- "A maniacal sense of urgency is our operating principle."
- "The only rules are the ones dictated by the laws of physics. Everything else
  is a recommendation."

### 1.2 First-principles reasoning

Boil a problem down to the truths you are sure of, then reason up; do not
reason by analogy ("this is how it is usually done"). The canonical example is
battery cost: the market said $600/kWh, the raw materials on the metals
exchange cost about $80/kWh, so the gap was convention, not physics.

### 1.3 Idiot index

Ratio of the cost of a finished thing to the cost of its raw materials. A high
ratio means the process, not physics, is where the cost lives. Musk used it to
decide what to build in-house at SpaceX.

### 1.4 The limiting factor

Musk works on exactly one bottleneck at a time: the single thing that most
limits the whole system right now. He goes to it personally, fixes it, then
moves to the next. Marc Andreessen describes it as fixing one critical
bottleneck per week.

### 1.5 Iteration rate and failure

"If a schedule is long, it's wrong. If it's tight, it's right." "The best
part is no part; the best process is no process." Starship went through many
full-vehicle iterations in a year; explosions are treated as data, not shame.
The metric that matters is rate of learning per unit time.

### 1.6 The 2018 Tesla productivity email

- Excessive meetings are the blight of big companies; avoid large meetings
  unless everyone gets value, and keep them short.
- Walk out of a meeting or drop off a call as soon as you are not adding
  value. It is not rude to leave; it is rude to waste someone's time.
- Do not use acronyms or nonsense words. Anything that requires an
  explanation inhibits communication.
- Communication should travel by the shortest path needed to get the job
  done, not through the chain of command.
- A "company rule" that is obviously ridiculous in a specific situation should
  be changed.

### 1.7 Time-boxing and the feedback loop

Musk plans his day in five-minute blocks. "Constantly think about how you
could be doing things better and questioning yourself" is the advice he calls
the single most important.

### 1.8 Documented cost of the method

Isaacson also records that physically impossible schedules demoralize
engineers ("engineers aren't stupid"). Urgency applies to process waste, not
to the laws of physics or to audit budgets. That caveat is part of the method.

---

## Part 2 — Translation into rules for this project

Physics here means: the DGPO objective, the density-ratio math, the audit
validity threshold (>= 1,000 classifier updates, saturated), the GPU budget,
and the wall-clock of a 16-GPU Ray allocation. Everything else is a
recommendation and can be questioned or deleted.

### R1. First principles on the bottleneck

Every session states the problem from first principles before any action:
which link of the chain
`discrimination -> ratio -> candidate ordering -> gradient -> optimizer
displacement -> distribution change -> fresh audit`
is the limiting factor today, and what must be physically true for the
H4 signal to cross it. No proposal may be justified by analogy to another
field ("RLHF does X", "GANs do Y") without stating the corresponding fact
about this chain.

### R2. Every requirement has a name

Each item in a protocol's frozen contract, each logged metric, and each arm
carries an owner: the user, or the run ID that established the need. Items
whose only justification is "we always log it" or "the previous protocol had
it" are deleted (R3). Requirements the user set are still questioned once,
in one sentence, then respected.

### R3. Delete before adding

Before a protocol adds an arm, a metric, a panel, a script, or a document,
it lists what it deletes. Targets, in order of cost:

- arms that cannot change a hypothesis status;
- secondary metrics that no decision rule reads;
- audit points that do not enter the primary statistic;
- protocol documents superseded by a later one (mark, do not erase history);
- scripts duplicated by a `--mode` flag on an existing runner.

If a quarter's deletions never had to be reinstated, deletion was too timid.

### R4. Optimize only what should exist

Before refining a diagnostic (a corrected replay, a v2/v3 rerun, a finer
metric), ask whether its answer can change the next decision. If the decision
rule in `current-diagnosis.md` does not read that answer, the diagnostic
should not exist. `h4b55phys1 -> h4b55phys2` is the standing example to test
against this rule.

### R5. Accelerate cycle time; the schedule test

- Prefer the fastest experiment that can falsify the hypothesis: read-only
  saved-direction replay (minutes) before a one-step trainer run (about an
  hour) before a 50-step trajectory (many hours). A long experiment is only
  allowed after the fast one has shown the effect exists.
- Independent arms run in parallel allocations, never serially.
- Every protocol states expected time-to-first-valid-point. If it exceeds one
  allocation, the protocol must name a cheaper proxy that runs first.
- A run gets a predeclared **kill criterion** (a step at which, if the
  primary statistic is already unrecoverable, the run stops and frees GPUs).

### R6. The project's idiot index

```text
idiot index = GPU-hours spent / number of hypothesis rows whose status changed
```

Track it per experiment in the RESULT note. A finished run that changed no
row in `current-diagnosis.md` has an infinite idiot index and must be named
as such. Crashed runs count their GPU-hours in the numerator.

### R7. Automate last, but do automate

Manual steps that have been repeated three times unchanged get a script:
today that is the W&B snapshot (`wandb_run_snapshot.py`, done), the RESULT
note skeleton, and the registry row. Never automate a protocol step that is
still being questioned under R2.

### R8. One limiting factor per session

The session works on the one bottleneck named in `current-diagnosis.md`
"Highest-leverage unresolved test", or on the record-keeping debt that
blocks knowing it. Anything else is written down as a candidate and dropped
for the day. The user may override this, in writing, per session.

### R9. Wrong is allowed; confident-and-wrong is not

Every conclusion carries one of **supports**, **rules out as a sufficient
explanation**, **unresolved**, plus a limitation. Numbers travel with their
audit budget and panel. An undertrained audit is reported as undertrained
before its AUC is spoken. A crashed run's partial history is evidence; its
missing endpoint is not.

### R10. Comradery is dangerous: challenge the plan

When the registry contradicts a proposal from the user or from Claude, say
so first, in one sentence, with the run ID. Agreement without a cited run is
not agreement. The user is expected to do the same to Claude.

### R11. Hands-on and skip-level

Decisions are made from the raw W&B history aligned by `global_step`, not
from a summary, a registry paragraph, or a memory note. The registry is the
manager; the history is the level below. Read it.

### R12. Shortest-path communication

- Findings go straight into the registry, diagnosis, and RESULT note, then
  one short message to the user: limiting factor, decision, evidence, next
  step.
- No new acronym without expansion on first use in each file.
- No status narration, no meeting-style recap, no restating the plan.
- A convention in this skill that is obviously wrong for a specific case is
  changed in the file, with a dated note, rather than followed.

### R13. Time-box and close the loop the same day

- Each session declares a time box and one deliverable at the start.
- Every finished or crashed run is registered on the day it ends. Unregistered
  runs are the clearest sign the feedback loop is broken; on 2026-09-17 there
  were nine.
- End of session: one line on what was done better than last time and what
  will be done better next time (the feedback-loop question).

### R14. Urgency within physics

No schedule may assume an audit under 1,000 updates, a skipped saturation
check, a skipped `--check-only` preflight, or a skipped contract test. Those
are physics for this project. Urgency is spent on deleting arms, running arms
in parallel, and registering results, never on cutting validity.

---

## Part 3 — Checklists (enforced)

### 3.1 Session opening (fill before any tool call other than reading)

```text
Limiting factor today  : <one link of the chain, one sentence>
First-principles claim : <what must be true for H4 signal to cross it>
Deliverable            : <one artifact: run launched / registry current / protocol written>
Time box               : <hours>
Deleted or declined    : <what this session will NOT do, and why>
Registry debt          : <unregistered finished/crashed runs, count>
```

If registry debt is nonzero and the deliverable is not "registry current",
justify the order in one sentence (R8, R13).

### 3.2 Protocol gate (appended to every `config/<PROTOCOL>.md` before launch)

```markdown
## Musk gate
- Link tested: <chain link>; decision row it can change: <row in current-diagnosis.md>
- Requirement owners: <item -> name or run ID> for every frozen-contract line
- Deleted from the nearest previous protocol: <list, or "nothing" with a reason>
- Fastest falsifier run first: <replay / one-step / trajectory> — why not cheaper
- Time to first valid point: <h>; parallel arms: <n allocations>
- Kill criterion: <step, statistic, threshold>
- Expected idiot index: <GPU-h> / <rows that can change>
```

A protocol without this section is not launched.

### 3.3 Result gate (in every `<ID>_RESULT_<date>.md`)

```text
Status label      : supports | rules out as sufficient | unresolved
Rows changed      : <current-diagnosis.md rows>, or "none" (then say the idiot index is infinite)
GPU-hours         : <n>  ->  idiot index: <n / rows>
Kill criterion    : hit / not hit / not defined
Better next time  : <one line>
```

### 3.4 Session close

```text
Done better than last session : <one line>
Do better next session        : <one line>
Registry debt now             : <count>
```

---

## Sources

- Isaacson's five steps and corollaries, quoted: https://fs.blog/elon-musk-the-algorithm/
- Chapter attribution and corollary wording: https://medium.com/@1marketcapital/elon-musks-algorithm-for-making-decisions-bcc1dbe94107
- Starbase interview, "optimize something that should not exist", requirements need a name: https://everydayastronaut.com/starbase-tour-and-interview-with-elon-musk/ and https://oodaloop.com/analysis/archive/the-everyday-astronaut-elon-musk-and-his-five-step-engineering-process/
- 2018 Tesla productivity email: https://www.cnbc.com/2018/04/18/elon-musks-productivity-rules-according-to-tesla-email.html
- First principles and the battery example: https://www.startuparchive.org/p/elon-musk-explains-first-principles-thinking-and-uses-it-to-predict-80-decline-in-battery-prices
- Idiot index: https://www.jaakkoj.com/concepts/idiot-index and https://founderchapters.com/chapters/elon-musk/the-operating-system/the-idiot-index/
- Limiting factor, one bottleneck at a time: https://wisdomoftheweek.substack.com/p/the-limiting-factor and https://stormy.ai/blog/scaling-via-limiting-factor-elon-musk-strategy
- Iteration rate, "schedule is long, it's wrong", "best part is no part": https://founderchapters.com/chapters/elon-musk/the-operating-system/rapid-iteration/
- Time-boxing and the feedback-loop quote: https://www.mayooshin.com/time-blocking-elon-musk-manage-time
- Cost of the method (demoralizing schedules): https://fs.blog/elon-musk-the-algorithm/ (Isaacson passages)
