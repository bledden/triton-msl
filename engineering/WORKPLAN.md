# 0.4.0 working plan

## November 7 dense-transpose route improvement

2026-11-07 later: Promoted narrow dense full-storage zero-offset transpose admission and two durable tests after installed numerical/route checks and independent review. Related transposed-B public mean falls1.08–1.26ms to0.21–0.26ms(4.5–5.4x),3200samples retained; descriptive finiteM4 result,notformalreleasequalification. FullCPU1318native/1226pure pass;11GPUroute/alias/fallback tests per mode plus durable reruns;64numerical cases have16outputs byte-equal across4installs. Current artifact receipt1f79f3c2,58pending. No default experimental flags weakened. Evidence transposed-dense-direct-20261107/FINDINGS.txt. Earlier mixed host-stack timings preserved. Next: broader public/reporter workload performance and remaining declared release scope, reusing installed harness; no new design-only review cycle or publication.

## November 7 installed integration and training execution

2026-11-07: Four host optimizations integrated into candidate and native/pure wheels rebuilt. Installed CPU1306native/1214pure pass,21/15 independent fastpath controls plus durable wrappers pass. Public9case panel and related transposed-B4case panel pass in before/after native/pure installs, output hashes equal. 7200-sample installed comparison mixed and descriptive, no general speedup or relabel. Fresh emitted kernels pass complete opt-in Reactant external training,41gradient arrays(max1.192e-7),six SGD updates each orientation. Artifact/native_and_pure receipt independently reviewed/source-bound a5d3e459;58pending,HOLD. Evidence host-stack-integration-20261107,host-stack-installed-20261107-harness,training-integration-20261107. Next: resolve actual performance costs/transition behavior and remaining user workload scope; do not discard composed contributions solely from noisy aggregate. No publication.


## October 28 Fable review accepted; native correctness layer built

Fable review `6b1d2bb8` accepts the composition at CPU scope. Fresh preparation
`fa512-rowlocal-native-correctness-preparation-20261028` (handoff `fb5b2a40`,
seal `83a3807d`) implements 23 pinned fixtures and a 230-command correctness-only
schedule across C/S/B/SB/F. It pins 256-thread dispatch, width32, prefix/suffix
guards, two finite output poisons, source-specific 32-key padded-tail semantics,
and independent finite/nonfinite/zero/subnormal references. Twenty tests pass
under Python3.9/3.14, including the full synthetic acceptance sequence; both
integrity checks pass. No compiler, native framework, device, ps or lease ran.

Fable's scalar write/read accounting note is addressed additively. The prior F
hardware precedent is retained with the metallib/runtime compiler distinction.
The original worker-window results review remains open; an exact offline request
is ready in WORKER-WINDOW-REVIEW-REQUEST.txt, not sent externally.

Next concrete implementation: adapt the inherited native adapter/controller from
its fixed two-fixture/875-command contract to these cases, offsets and counts;
bind the actual provider closure, then qualify all failure paths. This preparation
is execution-disabled and is not being presented as a finished native instrument.
Keep numerical qualification before timing, combined-arm testing, all historical
outcomes and all 59 obligations. No repeated approval request for this CPU work.


## October 28 row-local shader composition implemented on CPU

The successful GPU bisection is retained unchanged. Its R/RE leads motivate a
safe source investigation, not a diagonal-MMA product revert. New evidence lane
`fa512-rowlocal-composition-cpu-20261028` contains C/S/B/SB and historical F:

- S: each SIMD group privately broadcasts row coefficients, then scales matching
  fragment storage. It trades more coefficient writes for narrower publication
  compared with the historically unresolved/adverse shared-coefficient F.
- B: final scratch communication uses SIMD-group barriers, with unchanged row
  arithmetic and output indices. SB tests both mechanisms together.
- Ten offline compiler jobs pass. Fourteen CPU checks and six AIR mutation
  controls pass on Python 3.9 and 3.14. All five arms retain 21,888 scratch bytes;
  register pressure, GPU arithmetic and performance are not established.

Handoff `6d85c21d`, seal `24bca576`. No production, installed or prior evidence
changes. Next is independent scratch/API assessment and native numerical
qualification before a new predeclared performance comparison. A combined gain
may be useful even when individual contrasts remain unresolved. Keep F and each
required workload visible; do not relax the completed bisection's noise floor.

COMPOSITION.json maps the backend partners: FA zero-copy and parallel cold
inventory already exist in the baseline, so cannot be credited twice. Establish
actual small-descriptor reach before calling it an FA launch improvement. Keep
the reciprocal-hoist factor and separate cooperative/triangle populations in the
portfolio. All 59 obligations remain open; CPU compiler evidence is not release
or speedup credit.


## October 28 successful GPU diagnostic; rescale and combined leads retained

User explicitly authorized proceeding with other processes subject to capacity,
then directed troubleshoot/resolve/retry without repeated review back-and-forth.
This permits routine bounded repairs in this scope;it does not permit unsafe
shader replay,erasing failures,relaxing correctness/ownership/resource guards,
or pretending new deltas were independently reviewed.

After Opusbb9fe38b,the repaired-import GPU16f653a7 reached864accepted commands
before its300s lifetime. We retained that run and changed only worker lifetime
to420s within unchanged600s controller,60s cleanup,900s authority and4GiB limit.
Targeted73tests per2independent environments pass;new binding07348616 and fresh
CPU preflight/authority. Parent review and user-authorized delta are distinguished.

SUCCESS:311.746s,5accepted builds,875verifiedcommands,clean exit0,empty owned
census,lease released. Exactsameoutputs across5variants on bothfixtures;maxabs
0.000139764 against0.004 tolerance. Population276modules/557images stable atclose;
all additions fromCPU267/552 accountedfor. Independent native/system reductions
match;results handoffebadff7d,sealc469f656,in fa512-shader-bisection-worker-window-
results-20261028. This is completed hardware diagnostic,no end-to-end promotion.

Retain compatible leads:
- R rescale-revert and RE combined-revert:pairedrescale8.063us/8.521us positive
  32/32blocks,including allocation-matched E-vs-RE. Do not discard because formal
  floor21.500us makes this screen UNRESOLVED,or because end-to-endgain may be small.
- E epilogue~1.438us positive26/32;combinedRE median221.250us vsC231.125us.
  Interaction remains unresolved;do not assert independence or add medians asproof.
- O oldendpoint218.562us vsC231.125us is directional only;anchor formallyunresolved.

Next:owned null-pair outlier account is retained(oneRopeningpair10.750us residual),
without blaming background processes or changing thresholds. Design lower-noise
confirmation before a further measurement;inspect rescale codegeneration and
preserve combination tests with other backend/launch optimizations. Keep all
publicpath,end-to-end,shape,dtype,architecture and safety obligations in view.
No extra GPUrun merely to get a preferred answer. All59,N1release,RG-02,CP-01 open.

## October 28 guarded-import repair complete; successor ready for review

The failed attempt is preserved; the concrete preparation gap is repaired.
Candidate `fa512-shader-bisection-guarded-import-candidate-20261028`: handoff
b5bef517,seal80d0eb98,GPU binding16f653a7. Companion provider seal7976a0a3,
CPU bindingf95ed761. All128 copied files unchanged. Full-byte manifest adds
same-byte recording.py and preparation checker;61212files/7487links.

Actual serve and Snapshots import/fixture composition reproduces old KeyError,
then reaches before NativeAPI body with successor. Synthetic bootstrap/authority
and inert ctypes are explicit; no real Gate/Guard/freeze/native qualification.
Missing/wrong bytes refuse. Final90jobs/320tests per environment pass;6complete
verifiers pass. Retained harness failures are disclosed,not counted as passes.
No shader,oracle,Pin,limit,cleanup or875command schedule is weakened.

Cleanup account retains nested signal error and complete=false alongside reap
and empty census. Fakes show successfinish needs no signal,not hostEPERMcause.
Old spend and authority untouched;newspent empty,no newdecision or native run.

Next is one encompassing stop+repair review through Blake,not multiple small
reviews;then currentcoordination and new provider/GPU authority under unchanged
freshness/lease rules. Do not reuse a4d276f6 or its expired receipt. The target is
an actual valid GPU diagnostic,then kernel/backend optimization including the
compatible-gains retrospective. CPU pass is not GPU success or speedup credit.
N1release,RG-02,CP-01 and all59 obligations remain OPEN/HOLD.

## October 28 CPU preflight passed; GPU attempt spent on missing dynamic dependency

Separate direct decisions preceded CPU and GPU execution under Fable443993a2.
CPU662e651f completed51.156s;both provider children passed,353samples,0.900GiB
peak. Populations match prior accepted names/images with path-only relocation.
Original completion time preserved;receipt/clearance139.176s later,controller
entry24.276s after issuance. This progress is retained,not native GPU qualification.

GPUa4d276f6 spent once,UNAVAILABLE9.628s. The actual guarded adapter import reads
recording.py dynamically;that file exists in the seal but not the manifest.
It raised KeyError before adapter.open_native by retained traceback/source.
No command/prepare receipt or timing result. Import-AST closure and unguarded
CPU checks missed it;the provider preflight does not exercise this entry path.
The prior blanket dependency-completeness claim is withdrawn.

Worker reaped exit1,final owned census empty,lease release recorded. Nested
signal PermissionError and complete=false remain;empty top-level cleanup_errors
is not proof every cleanup action succeeded. Signal error cause unestablished.
No retry,source change or in-place binding repair.

Results review packet `fa512-shader-bisection-final-attempt001-analysis-20261028`
handoff04961de3/seal5170a7d2:9offline/probe replays and3fullverifiers pass.
Native-denied actual adapter import fails with sealed map,passes with only the
recording.py hash added in memory,and refuses the wrong hash. Not full startup.
Next after result review:CPU dependency/entry-composition repair that binds
dynamic loads and tests the real guarded import path;retain provider qualification
separately;review cleanup diagnostics. Any successor needs new binding/review/go.
FA512 and compatible-gains retrospective remain;all59obligations open,no gain.

## October 27 shared lock restored; final paired bindings ready for review

Blake's direct proceed answered the scoped restoration request. The absent lock
was created exclusively, dev16777229/inode189249476, without replacement or
flock. No lease/window acquired; no independent census of other sessions.

Fresh final candidate `fa512-shader-bisection-final-candidate-20261027`: handoff
`db83c952`, seal `55fc194e`, GPU binding `a4d276f6`. Companion provider-final
candidate: seal `06bddaf4`, CPU binding `662e651f`. All88GPU+40CPU copied files
are byte-identical to accepted parents; final header paths/flags/lock identity
are the only changes. Headers enabled for authentication, no authority supplied.
Full-byte inventory61210files/2006266763bytes/7487links;84jobs/311tests per
environment pass,75inherited outputs match,six full verifiers pass. Parent seals
unchanged. Private spent root empty;provider attempt remains absent.

Next: final encompassing binding review, current coordination/manual assessment,
separate concrete CPU go,one bounded provider run,offline report preserving UTC,
then device decision and linked receipt/clearance within300seconds. Native
behavior and performance remain unqualified. No provider,device,ps or GPU run
occurred in preparation. No historical authority or retry carries over. Retain
actual population review,own GPU idle-gap analysis and compatible-gains queue.
All59 obligations,N1release,RG-02,CP-01 remain OPEN/HOLD.

## October 27 activation protocol accepted; spent-root prerequisite closed

Opus review `c085fc89` accepts the activation code without defects. Its R1 is
closed: the bound private spent directory now exists empty (0700, owned, no
symlink). Protocol supplement `fa512-shader-bisection-activation-protocol-
supplement-20261027` has handoff `46c4f34e`, seal `f011cbff`; three read-only
verifiers pass. Parent packets and code remain unchanged. Final review and the
pre-window setup must recheck emptiness; never clear spent records.

Next requires current cross-session confirmation that other GPU sessions are
stopped and retain no old lock descriptor, then exclusive absent-path lock
restoration. The question is pending with Blake; no lock access or native work
was performed. Then prepare fresh runtime folders, full-byte inventory and final
paired bindings for review. Coordinate/manual assessment first, separately
approved CPU run next, device decision and receipt/clearance within300seconds
of original completion. Include preflight/math in actual population review.
No authority, performance gain or release closure; all59 obligations stay open.

## October 27 complete-provider preflight and authenticated entry preparation

Accepted Opus review `0114bd8e`. The CPU activation packet is ready for one
encompassing source review: `fa512-shader-bisection-activation-preparation-20261027`,
handoff `1286d4ef`, seal `58a5c409`, GPU draft `5222b5b8`; companion provider
packet seal `d419da49`. It reuses the successful bounded full-provider runner,
adds exact binding/decision/target linkage and original UTC times, then assembles
a GPU preflight offline. Old scope-only reports, partial runs, changed evidence
and stale timestamps refuse. Native entries are now authenticated, but both
bindings are disabled and the lease identity remains null.

75 synthetic jobs pass with 308 tests in each of three environments; native/pure
share a binary, system3.9 is independent. Six source audits and six full-byte
verifiers pass. Shared inventory:61,210files,2,006,266,763bytes,7,487links across
complete selected SDK/site/stdlib roots. Initial smoke and stopped first-wave
fixture failures are retained; fixes changed tests, not runtime. Full final wave
is qualification002. No native execution, ps, framework load or lock operation.

Next: review source integration, coordinate the existing-inode shared lock,
produce/review exact final paired bindings, separately authorize and perform the
fresh CPU provider preflight, then fresh receipt/clearance/quiet device decision.
No historical decision transfers and no authority is issued by this packet.
Keep the PyObjC-mediated static witness with its native uncertainty; explain
actual population additions and measure idle gaps only from our GPU timestamps.
The five-arm875-command diagnostic and compatible-gains retrospective remain;
no production/default change or correctness/performance/release credit. All59
obligations and N1release,RG-02,CP-01 stay OPEN/HOLD.

## October 27 ordinary-dispatch repair and fresh binding draft

Opus review `79dbafae` found ordinary dispatch could diverge from the namespace
pin. That R1 is now repaired: normal lookup must return a matching instance
selector; exact namespace identity remains required. An optional static witness
also catches both reviewer class overrides, including same-metadata replacements.
Static absence remains valid. Hidden bridge rerouting is not attested.

Packet `fa512-shader-bisection-binding-preparation-20261027` has handoff
`cd0a05ec`, seal `dbad07f7`, disabled binding draft `22478b0f` under the evidence
root.57jobs/164tests per environment and 6reviewer red-green probes pass;48inherited
outputs match. Three full verifiers pass. One stale smoke-count assertion failure
is retained; runtime was unchanged by its correction. The fresh inventory hashes
61,181files/2.006GB and7,487links across the full selected SDK/site/stdlib roots.

No native work or lock query ran. The lease identity is null, execution is false,
and native entry refusals remain. Review the repair/draft, then prepare complete
provider-aware preflight and authenticated entry activation. Do not just remove
historical preflight's refusal: it lacks the later whole-provider checks. Final
binding needs cross-session lock identity, review, fresh preflight/authority,
coordination and Blake's device decision. No earlier approval transfers. Preserve
late-population review, full SDK bytes, the five-arm875command finite diagnostic,
and the compatible-gains retrospective. All59release/workload obligations open.


## October 27 GPU selector/bootstrap propagation — CPU packet ready

Opus accepted the successful provider run (`48ee96fb`). Its G1-G3 findings now
have a concrete CPU successor: `fa512-device-selector-pinning-preparation-20261027`,
handoff `5d9334ac`, seal `5ab34200`, under the evidence root.

The GPU Pin uses the explicit instance-method namespace and retains exact unbound
descriptor identity. This removes the static-visibility dependency; it does not
accept a new wrapper based on matching metadata. Stable hidden descriptors pass,
transient or replaced descriptors refuse. The actual NativeAPI is unchanged.
Bootstrap Fix A is copied exactly from the successful CPU worker. The environment
allowlist, numerical oracle, five arms and 875-call schedule remain unchanged.

51 synthetic qualification jobs pass,145 tests per environment, with42 inherited
outputs equal the parent. Three final audits and three sealed verifications pass.
No device, framework import, live selector lookup, census or lock operation ran.
PyObjC12.1 source supports cached and fresh-wrapper paths; live stability on Metal
classes remains unmeasured. Late device/compile images remain unqualified. The
bound setuptools startup hook is expected and explained, not attributed to Metal.

Next: independent delta review using the verification lane's REVIEW-REQUEST.txt,
then prepare the fresh complete GPU execution binding/activation. This packet has
no executable binding and keeps native entries disabled. Preserve full SDK byte
reads, actual frozen-population retention/review and separately coordinated GPU
approval. Do not add another native probe or reuse a spent decision. Keep the
compatible-gains retrospective queue; no kernel/end-to-end gain claimed here.


## October 27 real provider qualification — both stages complete

One approved seven-scope CPU attempt under binding `3c6cb4e7` and Fable review
`5dfe96e0` completed in50.564s, entry0, both stages VERIFIED. Decision `cb7c6c64`
preceded launch. The real class alloc/instance init lookups succeeded without
invoking them; the framework worker completed all31 composition records and its
freeze/closing. Populations80/359 and267/552 (modules/images) are retained and
parent-accepted. Both children exited0/reaped. The348 samples show0.891GiB peak;
final owned census empty, incomplete discovery and sampled-cap limits preserved.

Results `fa512-provider-selector-resolution-attempt001-analysis-20261027` under
the evidence root: handoff `e46bba77`, seal `0f660063`. Six offline replays agree
across three environments. Population/source account retains all additions and
explains the external package roots without pretending to trace each import.
No selector invocation, device, GPU command, recording or lock action occurred.
Stored credit flags remain false. CPU provider prerequisites passed at this scope;
they are not a shader speedup or proof of the eventual device-run population.

Next: results/population review through Blake using the verification lane's
REVIEW-REQUEST.txt; not sent. After accepted results, propagate both bootstrap
Fix A and selector resolver into a fresh shader-bisection packet as CPU work.
Actual GPU work requires its separately reviewed binding, coordination/lease and
Blake's device go. This attempt is spent; no retry or additional native probe.
Keep the compatible-gains retrospective queue and all59 release obligations open.

## October 27 selector-resolution successor — ready for encompassing review

Opus results review `3cdf5a96` confirms the first native child passed Fix A and
locates the next gap at static PyObjC selector lookup. Preserve that progress.
The complete CPU successor is `fa512-provider-selector-resolution-preparation-20261027`
under the evidence root: handoff `193be75c`, seal `41eab5d6`, binding `3c6cb4e7`.
It uses non-invoking class/instance namespaces for alloc/init, retains inner
failure types through the controller, and models lazy exposure in dynamic-only
fakes. The retained parent refuses on those fakes; the successor passes.

All132 tests pass in each of three environments, including113 inherited and19
new controls; source audits and full verifications pass. Bootstrap, cleanup,
limits and environment stay unchanged. The fresh binding names a seventh native
operation, `live_objc_selector_lookup`; old six-scope decisions refuse. Full SDK
byte inventory remains. No native execution or new decision/attempt exists.

Fable encompassing review `5dfe96e0` is accepted, Q1-Q3 yes/no defect. The
unchanged packet and its103 members reverify. The acceptance sidecar is
`fa512-provider-selector-resolution-review-acceptance-20261027`; it clarifies
that error_type names the exception, not the returned object's type. Namespace
versus method failure remains deliberately ambiguous. No runtime change or new
native run. A review-pinned draft is unapproved and carries no authority.

Next: Blake's direct seven-scope decision on binding `3c6cb4e7` and a
current quiet/no-lock window are needed for one CPU provider run. Framework-worker
freeze and real selector resolution are still unproved. Shader/device work stays
behind its own later binding, review, coordination and authority. This diagnostic
repair is not a kernel gain and does not displace the compatible-gains queue.

## October 27 retrospective — Opus corrections incorporated

The second handoff now has an additive supplement:
`optimization-composition-retrospective-supplement-20261027/HANDOFF.txt`
under the evidence root (handoff0ee91fad, seal7cc5ee40). It responds to Opus
review ff9f92c9. The sealed parent42 cases remain unchanged;14 added records
cover the omitted August campaign and June/July context, with28 commit records.
Coverage is a historical audit, not proof every scratchpad experiment was recovered.

Corrections: small descriptor is a source-predicted local signal, historically
FAILED, with no measured helper-selection counter. Its pure softmax/matmul public
costs remain visible. p503 is about5% faster than rc4 but10.6% slower than old main
on the standalone cell; both workload regression alerts fired. K layout is first
a public measurement-resolution question, not evidence of host cost masking a gain.
August causal skipping supplies a commit-reported example of a reverted change
becoming useful after dispatch repair; historical numbers are not current credit.

Preservation requires predeclared code-path populations and selection counters,
resolvable non-target gates, an owner/source hashes/revisit trigger for each lead,
and no retroactive promotion. Retain failure reasons and recovery pointers rather
than unsafe live code. Compare coherent compositions without requiring an arbitrary
per-piece gain, but never use an aggregate win to erase a regressed required cell.

FA512 bisection stays primary. After it, paper integration-debt order is cold
inventory, cooperative-attention workload census, small-descriptor S1–S6 checks,
then K-layout sample/resolution budget. Current Codex/Astra lane owns this queue;
Opus review is requested through Blake. No screen reopened or execution authorized.
The project GPU lock is absent and the old inode-bound controller cannot execute.
Coordinate re-establishment with Blake/cross-session protocol and fresh binding;
no lock was created or acquired here. Controller preparation has now been reviewed
by Opus (0b4d1a98); its repair is described below. All59 obligations remain open. Supplement ready for manual sharing, not sent.

## October 27 static-origin invocation — first-stage freeze accepted

One approved clean-env invocation ran under binding54b8b5b9/review74ca5349 and
decisionc2ebbb1f (written before launch; six scopes, no project lock, quiet window).
It ended UNAVAILABLE/exit1 after47.690s and is spent, no retry. The controller-provider
stage completed composition, freeze and closing, its89module/359image population
was accepted, and the child exited0/reaped. Neither warnings module was loaded in
that population. The static-origin repair passed this native stage without preload.

The framework worker completed the pre-device prefix and20attributes, then refused
static inspection of Metal.MTLCompileOptions.alloc. It never reached ABI checks,
its symbol stage or freeze; no worker population exists. The inner exception was
not retained. Do not call this a missing Metal method or a kernel failure. The
source shows inspect.getattr_static can miss dynamically exposed attributes;
that is a candidate explanation requiring a faithful selector-resolution model.
No device,command or GPU work ran. The second child was reaped(-9); final owned
census empty, no cleanup errors. Sampled peak0.886GiB,338samples; discovery incomplete.

Results: `fa512-bootstrap-static-origins-attempt001-analysis-20261027/HANDOFF.txt`
under the evidence root (0e12942b;seal8049a05d). Six offline verify/analyze replays
agree in three environments. Ready for Opus results review through Blake,not sent.
Next CPU-only model/source inquiry for non-invoking native-style selector checks
and fixed inner failure diagnostics; new review/binding/decision before any native
invocation. Preserve first-stage progress; do not re-open or discard it because
a later prerequisite now failed. GPU propagation/rebinding/review stays outstanding.
FA512 primary and broader compatible-composition queue remain; all59 OPEN/HOLD.

## October 27 static module-origin repair — historical reviewed preparation

Opus results reviewf90a19d0 supplies the CPU reproduction: frozen importlib.machinery
has __file__ but no __cached__; the missing-attribute hook imports warnings and
_py_warnings. This is a source-backed offline reproduction, not an import trace
from the spent native run. Its accounting remains accepted and unchanged.

Chose recommended fixA in `fa512-bootstrap-static-origins-preparation-20261027`
under the evidence root (handoff5dbd19cb,seal225f0430,binding54b8b5b9). Four bootstrap
metadata reads now use the built-in ModuleType dictionary descriptor. No warnings
preload or check retry. Exact module-set/origin and source/cache/image byte checks
remain. Stored cache metadata only is an explicit observation change; non-module
entries and dynamic missing-file cases conservatively refuse. Real framework
compatibility remains unqualified. All older CPU/GPU packets stay sealed.

Final wave:24jobs,113tests per environment plus fresh old/new probes and audit.
The3.14 old read adds2 file-backed warnings modules; the new read adds none at all.
An initial all-modules-versus-file-backed counter failure is retained; the repair
was to the probe only. All three full integrity verifications pass. Complete fresh
CPU binding and clean command are included for one combined review, not sent.
No actual decision or native attempt exists. After review, obtain a new specific
CPU decision and ps escalation. After qualification/results review, propagate
accepted bootstrap bytes into a freshly bound GPU instrument with coordination,
lock and encompassing review. No existing GPU binding is silently updated.
All59 OPEN/HOLD; broader compatible-optimization composition queue retained.

## October 27 freeze-diagnostics invocation — two module names retained

User answered yes to the exact one-run request: bindingfdee6835, reviewcb678bdc,
six native scopes, no project lock and a freshly confirmed eight-minute quiet
window. Decisione0eb9189 preceded launch. The single clean-env invocation stopped
UNAVAILABLE after29.176s, entry exit1. The new CPU attempt is spent, no retry.

The first child completed both libproc lookups and refused inside Guard.freeze.
The post-failure diff captures _py_warnings and warnings as two bound-stdlib
additions:89 frozen versus91 file-backed modules, no removals or opaque entries.
This identifies names and phase, not the importing operation. The parent already
had both modules. Pinned warnings.py imports _py_warnings; no stronger cause is
claimed and no preload fix or native reproduction was attempted. Shared bootstrap
and checks remain unchanged. No framework worker stage or GPU work ran.

Child reaped(-9), final owned census empty, no cleanup errors; sampled peak0.829GiB,
160samples, discovery incomplete. Results handoff2880e362/sealb1fb1a64 is in
`fa512-provider-freeze-diagnostics-attempt001-analysis-20261027`. Six offline
verify/analyze replays match across three environments. Ready for Opus results
review through Blake, not sent. Next CPU-only lazy-initialization investigation;
no GPU attempt until CPU qualification passes, and no new native authority.
The broader composition queue remains; all59 OPEN/HOLD.

## October 27 failed-freeze diagnostics — historical reviewed preparation

Opus results review cd410571 accepts the stopped run; the cause remains unknown.
The caller-only successor is `fa512-provider-freeze-diagnostics-preparation-20261027`
under the evidence root (handoff c986e18b, seal 8d2b60b9, binding fdee6835).
S1–S5 are implemented: separate freeze phases, bounded path-free module-name and
origin-category capture, controller closing capture, and real synthetic imports
that discriminate both worker failure points. Shared bootstrap and membership
checks are byte-identical; no exception is replaced or check retried. The diff
is a post-failure observation, not proof of what changed at the triggering check.

All 91 tests pass in each environment, plus audits and three full integrity
verifications. The complete fresh binding and clean command are in the same
review packet. No native run or actual decision exists. Next: Opus encompassing
review, then a separate specific CPU invocation decision and ps escalation.
No GPU attempt until a CPU run passes freeze. Attempt001 remains spent.
The broader optimization-composition queue remains scheduled; all59 OPEN/HOLD.

## October 27 approved CPU invocation — stopped at module freeze

The specifically approved one-shot invocation ran under bindingbc8ce21f,
reviewf6f63b9d and the clean launch. Decision8a97cf29 preceded it and explicitly
covered the six native scopes, no project lock and requested quiet window.
It ended UNAVAILABLE after46.979s with exit1. The CPU attempt is spent; no retry.

The first child completed the two libproc lookups, then refused `loaded module
population changed` during freeze/check. The child population is missing, so the
changed module and exact internal/following-check point are unknown. The second,
Metal/objc/numpy provider stage never launched. The direct child was reaped after
SIGKILL; final owned census was empty. Sampled peak0.83GiB,250samples, no resource
refusal. These are narrow accounting facts, not native qualification or a kernel
result. No GPU device/commands/attempt or project-lock operation occurred.

Results review: `fa512-provider-activation-attempt001-analysis-20261027/HANDOFF.txt`
under the evidence root (cf75184f;seal73ed1fd9). Six offline replays agree and all
inputs/source/raw records remain bound. The caller-side diagnostic successor above
now preserves a bounded failed-freeze diff; do not relax equality, extend caps
or rerun this attempt. Actual child module/image qualification remains open.
The FA512 and broader composition plans are unchanged; all59 OPEN/HOLD.

## October 27 activation accepted — historical pre-run checkpoint (superseded)

Opus reviewf6f63b9d accepted bindingbc8ce21f with one launch-only requirement:
clear the top-level environment. `fa512-provider-runner-clean-launch-20261027`
(seal2a8d835d) records the exact env -i prefix and unchanged reviewed argv. No
code change or reseal is needed. No actual decision or attempt exists. The user
has been asked for one explicit CPU run, six native scopes, no project GPU lock
and confirmation that other timing runs are idle for at least480 seconds. Issue
the fresh decision only after that answer and immediately before invocation.
This does not authorize a GPU step; actual populations require review afterward.

## October 27 CPU provider activation — ready for combined review

`fa512-provider-runner-activation-preparation-20261027/HANDOFF.txt` under the
evidence root (fe888017;seal2e697614;bindingbc8ce21f) responds to accepted Opus
review09db43a0. R1 fixed reasons/types, N1 single population read, entry activation
and a fresh enabled binding are complete in source. A new conservative deadline
anchor fixes a possible parent-after-spawn scheduling race without increasing
limits. Synthetic predecessor controls discriminate all three changes.

The final12-job wave passes71 tests in each environment. The first wave's
packaging-order failure and the passing intermediate wave are retained. All
124packetmembers/61,627pins, including60,543inventoried files, verify under all
three interpreters. No native qualification or backend speedup is claimed.

The exact command is ready for activation+binding review. No remaining entry
edit is proposed after review. The separately issued user decision must cover
all six native scopes across the controller tree and explicitly approve the
proposed no-project-lock CPU policy; it has not been issued. No native command,
ps, framework/ctypes load, device/GPU or lease action ran; lock not queried.
After a real CPU run, review actual frozen modules/images before GPU bisection.
The broader composition queue and all59 OPEN/HOLD obligations remain intact.

## October 27 bounded CPU provider runner — synthetic preparation

The combined runner checkpoint is `fa512-provider-bounded-runner-preparation-20261027/HANDOFF.txt`
under the evidence root (212c04dc; seal734540a3; disabled binding7cab83e5).
It composes the accepted provider surface with two serial child stages, the
reviewed allowlist, startup environment binding, population retention, explicit
budget reserve and directly owned child reap fallback.54 tests pass in each of
three interpreter environments; the inherited sources are unchanged. A full-byte
SDK/stdlib/native-site inventory binds60,542files and7,487symlink choices. Three
final verifiers agree over61packetmembers and 61,563pins, retained separately.

This is infrastructure preparation for FA512 bisection, not a backend gain.
No native provider/ps/framework/ctypes/device work or lock operation ran. The
entry and binding remain disabled. The combined review must cover the native
entry body and its remaining executable activation/dispatch requirements; this
packet supplies no runnable native command or approval. Actual provider/resource
fit and frozen populations remain unmeasured. Review and specific CPU-native
consent precede that stage; GPU lease/binding/window remain separate. The complete
optimization-composition retrospective queue above remains retained. All59
obligations remainOPEN/HOLD.

## October 27 FA512 qualification companion — reviewed response

Opus prefix review f93a8a7e is addressed by
`fa512-provider-surface-preparation-rev2-20261027/HANDOFF.txt`
(ad418ed4; seal239b0690). The full explicit local provider surface is20 attributes
plus static alloc/init inspection;9 are additional to the old prefix/preflight
union. The prefix now has an exact five-call allowlist, with a renamed-creator red
control, closed import forms and protected bindings. All23 sealed worker/support
sources remain unchanged; no CPU mode or enabled entry was added.24 synthetic tests
pass in each of three environments and final integrity replays agree. The first
verifier's seal-schema error and draft are retained; the repaired verifier handles
the controller's flat map explicitly. No native qualification is claimed.

Controller handoff eca09759/c08045b6 and this companion were reviewed by Opus
(0b4d1a98 and0d549c43); the additive repair below responds to their findings. A concrete separate provider runner and process-lifecycle runner still
need review; real Metal/objc loading needs Blake's specific decision distinct from
a device go. No such load, ps, device call or lock operation ran. The missing lock,
fresh binding and encompassing instrument review remain before GPU work. The
875-command schedule and all59 obligations are unchanged.

## October 27 combined controller/provider repair — ready for review

`fa512-controller-provider-review-repair-20261027/HANDOFF.txt` under the evidence
root (a6381e3e; seal d67badb4) addresses the two Opus reviews. The adapter retains
failed-arm/compiler diagnostics through the worker STOP record and controller.
The controller refuses debug environment prefixes before spend and checks the
verified binding's spent root. The prefix model rejects implicit invocation
forms; four ctypes symbols now have an explicit future qualification contract.

Final CPU replay: 99 tests per environment, 39 jobs, identical stdout; native/pure
share a binary, system Python is independent. Four expected predecessor assertion
failures discriminate the repairs. Audit reconstructs four changed copies; 71
copies remain unchanged. Three preparation issues are retained and disclosed.
That packet refused the inherited `MallocNanoZone` name. Opus review48bcce97
identified the policy mismatch; the allowlist successor below supersedes that
prerequisite without changing the live environment.

The source repair is complete at CPU scope. No new execution binding was made,
and all native entry fuses remain off. Next: review this combined packet, then
complete the separate provider/lifecycle runners before any native qualification.
Real framework loading/symbol resolution requires its own decision. Current lock
coordination, a fresh binding and encompassing review remain before any device go.
The 875-command shader design and all numerical/regression gates are unchanged.
Nothing was sent externally. All59 release obligations remain open.

## October 27 worker environment — allowlist successor

`fa512-controller-environment-allowlist-20261027/HANDOFF.txt` under the evidence
root (21b2cff6; seal1b77762d) responds to Opus R1 (48bcce97). Only HOME, USER,
LOGNAME and TMPDIR can be inherited; PATH is fixed. The controller supplies its
own authority and thread values. Every other incoming name is dropped and
reported without its value. MallocNanoZone no longer requires changing the host
app; Foundation/unknown debug names are not passed. MTL_*/METAL_* still refuse
before spend. This policy does not reset the controller or host system state.

The unchanged reviewer probe reproduces both old behaviors and demonstrates the
new drops on fakes. Final48 jobs pass,109 tests/environment, plus3 audits and3
verifiers; proof labels share one binary, system Python is independent. Only the
controller runtime and its old prefix expectation change;78 copies are identical.
No native environment qualification is claimed and no new binding was made.
Opus reviewfd369562 accepts this packet; its N1 startup note is handled in the
composition below. Provider/lifecycle qualification must use this policy before any device attempt; the current lock/fresh-binding/encompassing
review requirements and all59 open obligations remain unchanged.

## October 27 provider-stage composition — combined runner checkpoint

The allowlist reviewfd369562 is accepted. N1 distinguishes the launch environment
from Python/macOS startup additions. The new
`fa512-provider-stage-composition-20261027/HANDOFF.txt` (9cc0859b; sealbe851527)
implements the comparison: all sent values retained; only LC_CTYPE and
__CF_USER_TEXT_ENCODING permitted as optional additions, recorded by name. The
expected interpreter is pinned to relocated native Python; no system shim switch.

The separate probe now composes the checked pre-device prefix, full provider
surface, six existing ABI checks, later-use roles, and separate dyld/libproc
resolution stages. A dormant native lookup adapter is prepared, with actual
symbol-method orchestration tested against fake ctypes. No native function is
invoked.24 tests per environment pass plus audit/integrity checks; all runtime
source copies and helpers are unchanged. This is not full provider qualification.

Continue directly to the bounded one-shot wrapper: current full-byte provenance,
resource caps, owned-child lifecycle/cleanup and failure retention, then combined
review and a specific decision before real framework loading. This checkpoint is
not another isolated N1 review request. No native command or binding is issued.
The shader experiment, retrospective combination queue and all59 obligations stay.

## Priority change, October 26: shader regression first

Blake accepted the recommendation to prioritize CPU preparation for FA512 shader
bisection and add userspace session discovery to the deferred plan. The current
source-bound design is `fa512-shader-bisection-design-20261026/HANDOFF.txt` under
`/Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start`.
It prepares dependency-complete diagnostic source groups and an interaction-aware
comparison, not a product reversion. Nonfinite/tail correctness requirements remain
mandatory. Design review precedes a concrete instrument and any new device window.

Deferred: userspace numeric-session discovery in the recorder supervisor, if tracing
is still useful after the shader investigation. This does not improve kernel speed
by itself. The header-defined `KERN_PROC_SESSION` shortcut is withdrawn following
Opus review `e282adaa`; no exact host errno is inferred from the published source.
The candidate needs PID birth checks, unrelated-metadata privacy boundaries, retained
helper identities, session-change/disappearance handling, resource accounting and
reviewed native qualification. Polling remains incomplete. Keep per-step diagnostic
records in bounded memory and persist once per sample or at first failure, not
between primitive reads. Do not retry either held C4 rehearsal against the old
zero-valued `ps sess` premise. This backlog item supplies no execution authority.

## Development checkpoint, October 22

The active plan is now `DEVELOPMENT-STATUS.txt`. The text below is retained
historical context, not current execution authority or a request to retry.
The October25 research queue there retains six mathematics leads for exploration
at relevant workload transitions; it does not displace the active FA512 work.
The separate `dev/0.4.0` branch preserves the frozen source baseline; later
experimental roots have not been merged. Density rev4 failed its RISK gate
and is closed, with GAIN blocked. No release obligation has been waived.

Owner: Astra. Shared with Fable through Blake. This is the next-work sequence,
not a release promise or a relaxation of `CONTRACT.md`. The ledger remains the
workload/obligation register; this file is the short execution plan. Update this
plan at a completed experiment or a real dependency change, not every message.

## Current checkpoint October 11

Rev7 native-project ran once and reached 4,226 passes, ten skips and one failure
among 7,129 selected nodes, leaving 2,892 unreported. The SciPy softmax comparison
passed. The stop was an obsolete expectation that the host route refuse packed
arguments; the call returned, but no output value was retained. No numerical pass
or mismatch is inferred. The 577.223-second attempt is sealed over 1,477 files,
without phase credit. Cleanup, report restoration and the free lease are verified;
the clearance is spent.

The test now requires the exact host result 528. Its historical node, generated
kernel, inputs and MPS branch remain unchanged, with no production or installed
payload change. The accepted packed-ABI capability supports this expectation;
its older hardware evidence is not transferred to this call. Final CPU checks
pass: 300 existing ABI/ownership case executions in opposite orders, 32 recording
predicate probes, exact unspecialized 34-argument emission using two bindings,
four unchanged guarded collections and two cold bootstraps. Runner hook, ordering
and invocation mistakes are retained, corrected and disclosed in the handoff.

The complete rev8 successor is ready for one combined review:
`trifast-regression-controller-rev8-supplement-20261011/FABLE-HANDOFF.md`.
Binding `ae161ff736d890a0226224d6a162a9f03650a5a42176e606a1f10377b5d55aaa`
covers 134,483 files. Twenty-one controller modules are unchanged. All nineteen
CPU jobs, 32,942 synthetic states, twenty verifier attacks, four guarded preflights
and supplemental tamper/population checks pass. Native-project alone is proposed
for review and fresh authority. Pure remains withheld. No successor GPU execution,
receipt, clearance, commit or watcher exists. The warm-call sequencing pause and
release obligation states remain unchanged.

### Previous SciPy checkpoint October 10

Rev6 native-project ran once under the confirmed window and stopped at the missing
SciPy reference import after 2,800 passes, ten skips and one failure among 7,129
selected nodes. The MLX invocation preceded the import; its reference comparison
did not execute. The 279.481-second attempt is sealed over 1,477 files without phase
credit. Report restoration, reap, empty census and free lease are retained; its
clearance is spent.

The dependency repair and complete rev7 successor are ready for one combined review:
`trifast-regression-controller-rev7-supplement-20261010/FABLE-HANDOFF.md`.
Binding `7f0e0e95883c757d97b2e2bdb9bead28465e311394d690926b58942c9f79e140`
covers 134,212 files. No production or test source changed. Fresh relocated native
and pure installations add RECORD-verified local SciPy only, with full tree deltas
and passing dependency checks. Both old reference imports refuse; both repaired
CPU references pass and agree across modes. Four guarded censuses preserve the
7,129-node populations and marks, sixteen bootstrap refusals pass, and the static
dependency census names its remaining optional, synthetic and out-of-scope imports.

Twenty-one controller modules are unchanged; only the composition path changes.
All nineteen CPU qualification jobs, 32,942 synthetic states, twenty verifier attacks,
four guarded preflights and supplemental tamper/population checks pass. Native-project
alone is proposed for review and fresh authority. Pure applicability remains withheld.
No successor GPU attempt, new clearance, commit or watcher exists. The warm-call
sequencing pause and release obligation states remain unchanged.

### Previous helper image checkpoint

Rev5 native-project ran once and stopped at a checkout-only native-image fixture:
1,771 passed, ten skipped, one setup error; 7,129 selected and no required identity
missing. The 184.494-second stop is sealed over 1,477 files with no phase credit.
Restoration, reap, empty census and free lease are recorded; clearance is spent.

The same image-path defect was found in the dictionary-epoch fixture. Both now
select the installed helper image while retaining separate extension contexts;
all twenty test functions and their node identities are unchanged. Both parent
fixtures fail the guarded red probe; all 45 native cases pass in each fresh order.
Pure still refuses missing native helpers. Production and installed bytes did not
change. All four guarded censuses preserve the prior populations and marks.

The combined review entry is
`trifast-regression-controller-rev6-supplement-20261010/FABLE-HANDOFF.md`.
Rev6 binding `314a0b07ce73c911935ae6d1224cfcddaf645c245b72e877f360444495c8a919`
covers 87,956 files; 21 controller/support modules are inherited byte-for-byte.
Nineteen CPU jobs, 32,942 synthetic states, twenty verifier attacks, four guarded
preflights and supplemental tamper/population controls pass. The sealed rev5 delta
representation defect remains untouched; rev6's normalized replay passes.

Native-project only is proposed for independent review and new authority. Pure
applicability stays withheld, no requirement is waived, no new device execution
or commit follows from the CPU package, and no watcher is running. Warm-call
sequencing pause and all release obligation states remain unchanged.

### Previous child origin checkpoint

Rev4 native-project stopped after 676 passes, three allowed skips and one failed
historical pass. Its audit child selected the helper-free checkout instead of the
installed parent. The 114.585-second attempt is sealed with no phase credit;
cleanup completed and the lease is free. The clearance is spent.

The completed test repair and rev5 successor are ready for one combined review:
`trifast-regression-controller-rev5-supplement-20261010/FABLE-HANDOFF.md`, SHA256
`a476f67197381fcbfa0d243584ad80f84284a413af0d55497aa174d540a7bf59`.
Binding `1bf36ebc9fd22b00d6a2d4ef116b1780a770cbc7ac935494e7cfc3047d17b17c`
keeps production and installed payloads unchanged. Seven cold child commands use
one parent-root selector; programs and assertions are unchanged. All 7,119 prior
nodes and marks remain, plus ten new helper controls. Forty final helper-control
executions pass, three mutants are caught, nineteen controller jobs pass, and
the 32,942-state rehearsal and supplemental tamper/population controls pass.

Only native-project is proposed for a new reviewed clearance. Pure-project is
not ready: the accepted 32-node applicability policy misses native-helper
requirements exposed by the actual CPU children and source inspection. Nothing
is skipped or waived; finish a per-node CPU applicability census before a pure
phase. CPU recorder fixtures cannot establish that policy's completeness.

After review, native execution still requires fresh coordination, device
assessment, preflight, clearance and lease. No watcher is running. The other
release obligations and warm-call sequencing pause are unchanged.

### Previous Z3 repair checkpoint

The Z3 repair and complete successor are now combined for one narrow review:
`trifast-regression-controller-rev3-supplement-20261010/FABLE-HANDOFF.md`, hash
`a92a92545d49805ca957f53b5f932c25b9fdf8c692fcc943d2c8bec471adc825`.
Binding `104499347814828af4cc849374f8acc6f1730b21fb8cb75cca39e77a9dc40057`
covers both retained and new private installations. All predecessor origin
payloads remain byte-identical; local Z3 sources/image are the only added payload.
The exact CPU lazy-import program passes in both modes, with Z3 initially absent
from sys.modules and its actual loaded image bound. Both parent absences reproduce.
Both7106-node censuses stay identical. Twelve bootstrap refusals,19controllerjobs,
the32896-state rehearsal and20attacks pass; no new GPU work ran.

Seventeen controller modules are byte-identical and four origin/composition edits
reverse exactly. An audit JSON tuple/list comparison defect is corrected only in
the additive sidecar; the original controller seal and binding remain unchanged.
Next is one independent review of this complete delta, then fresh phase authority.
Routine CPU repairs were completed locally rather than handed off individually.
The protocol still requires the review to name the new binding. No old approval
is relabeled. Production/tests, baselines, paused work and obligation states stay
unchanged; no watcher or routine remains.

### Retained native project stop

Native-project rev2 ran once and stopped after 508 passes when the historical
Z3 lazy-contract pass became a skip. Both private installs lack Z3; its GPU
lazy-import test is also a historical pass. Torchvision is also absent but its
test was historically skipped, not a new required-pass blocker. Collection marks
were qualified; body-time dependency gates were inventoried but not reconciled.

The partial-stop packet is sealed in
`trifast-regression-native-project-results-rev2-20261010/FABLE-HANDOFF.md`, SHA256
`eff3758053346500f057af864663c2a23893b05341cfd5bbdd0c23bc921513b6`.
Journal/JUnit agree; no failed report, no phase credit. All six replacements
remain unreported. Worker reaped, empty census, report restored, lease free.
No retry, later phase, production/test/installed edit, commit or watcher.

Next: Fable review of the stop and diagnosis; new private dependency compositions
with Z3 provenance, guarded real cold/lazy imports and runtime-predicate coverage
before successor binding and fresh phase authority. Keep historical requirements;
do not preload away the lazy-import case. No baseline waiver. Full regression
before commit; operation-semantics matrix before further transport.

### Previous controller checkpoint

The complete successor controller is sealed for Fable instrument review:
`trifast-regression-controller-rev2-20261009/FABLE-HANDOFF.md`, SHA256
`9af146092083c83ed942ab9cd53dc92da4678221c47f12ce9489a7bb3eda257f`.
Its binding covers 44,058 files and records the reviewed mode rule. The local seal
covers 920 files. Historical maps and production/test sources are unchanged.

Before pure-project, the controller must reconstruct a completed native-project
result from raw evidence with its own verifier, pin the independent population
review, and keep the evidence hashes stable through execution. The final join
infers no pure passes. All six replacements and the 32 native-only obligations
remain required under their accepted rule.

Nineteen serial CPU jobs pass, including both installed origins, the exact live
predicates, authority and process-loss controls, 32,896 synthetic node states and
four guarded preflights. Additional attacks show active-worker tampering stops and
cleans up, and differing populations refuse. No project test body or device ran.

Next: Fable instrument review, then four separately cleared regression phases.
Each still needs fresh coordination, assessment, preflight, user go and lease.
No retry of the spent clearance. Population additions remain review items.
The operation-semantics matrix precedes further transport.

### Previous dependency checkpoint

CPU dependency preparation is ready for Fable in
`trifast-regression-dependency-preparation-20261009/FABLE-HANDOFF-V2.md`,
SHA256 `67fd0405c0cd77d295a0aceb3e0333c6d429c8a8024ce14dcf4722fb247d1459`.
Two new isolated source-derived installations include MLX and source-matched C++.
Both collect 7,106 nodes under native creator guards, restoring all 72 MLX
historical nodes. Two changed IDs have proposed source-bound continuity mappings;
1,013 additions remain review items, not a new baseline.

Native has no historical-pass skip marks. Pure has 32 native-helper historical
passes marked to skip: 18 identity, 11 packed and three binder tests. These remain
explicit gaps; the strict gate still refuses pure. No helpers were added to pure.
A no-device profile hides nine Welford nodes at module import; the complete census
therefore models availability explicitly while blocking real creators and execution.

The seal covers 43,326 files including installed trees and binds 475 external
inputs. Three guard jobs and ten accounting refusals pass. No test body, device
retry, authority, lease, commit or production/test edit. Next: Fable review, then
a reviewed mode-specific obligation mapping and installed bootstrap integration
into the successor controller. The operation-semantics matrix precedes further
transport. No obligation or workload state changes.

The original hash-list ordering verifier defect is retained unchanged;
verify_packet.py validates all original contents plus five correction files.
Both corrected replays pass. No census or test result changed.

### Previous native project stop

The first authorized native-project phase stopped at collection before any test
body ran. Handoff: `trifast-regression-native-project-results-20261009/FABLE-HANDOFF.md`,
SHA256 `fa74b7f32bcfb463c1f5a75dfed12a13dcd9475938cfaad43c4c8a8f628f0ecb`.
The 46-file stopped packet verifies in both modes; the approved verifier still
refuses the run. No phase credit or retry. The worker was reaped, report restored,
group absent and lease observed free after 12.055 seconds.

Collection selected 6,902 nodes, missing 74 historical passes and adding 881.
Seventy-two missing passes belong to four MLX modules skipped because MLX is not
installed in this private composition. Two old names are absent from current
sources: the C++ differential is now parametrized, and the direct-cache ownership
test requires source-level continuity analysis. Five C++ modules also skip for a
missing compiler extension; the native host helpers do not provide it.

Next: Fable stop review and CPU successor preparation. Reconcile historical
dependencies with the complete regression environment, prove changed-test
continuity and handle collection skips explicitly before another device window.
Do not erase missing passes, substitute additions or retry the unchanged setup.
No production or test edits; public dispatch and obligation states unchanged.
The operation-semantics matrix still precedes further transport work.

### Previous controller checkpoint

The regression controller is CPU-qualified and sealed for Fable's instrument review:
`trifast-regression-controller-20261009/FABLE-HANDOFF.md`, SHA256
`eda3c9efaf8e3263ff9d583786a20a4568c2211750f059e8ab09f138210f38c8`.
Blake approved preparation of four separately cleared phases: project at most
25 minutes and upstream at most 40 minutes per mode. This is not GPU execution
authority. Each invocation runs one phase, reserves cleanup time, spends its
clearance once, and requires the previous result and population review before
advancing. The accepted recorder and bootstrap are unchanged.

Each mode passes 38 authority/monitor checks and seven real process scenarios;
eight worker-boundary cases, nine verifier mutants and two modeled controller
cases pass. Full synthetic populations reconstruct 30,874 node outcomes; these
are not executions of the full suite. Both final guarded preflights pass outside
the sandbox. The monitor permission stop and sealing-import refusal are retained,
repaired and qualified. The packet seals 283 files with 1,502 bound inputs.

Next: Fable instrument review, then fresh per-phase coordination, preflight,
clearance and lease before device execution. Full project/upstream regression
remains required before commit. Build the operation-semantics matrix before more
transport. Production sources, public refusal and all obligation states remain
unchanged; no device work, production lease, commit, install or watcher.

### Previous exact-node checkpoint

Fable accepted readiness (`12afdd5e`). The next CPU component is sealed in
`trifast-exact-node-composition-20261009/FABLE-HANDOFF.md`, SHA256
`4c8b7c6a667db8550f7289906d224604356ccdc6933acf0eb44161ddf3143bf5`.
Each mode completes twelve real-pytest accounting scenarios and refuses fourteen
altered-evidence cases; both cold bootstraps pass with no unexpected attempts.
The native composition selects three verified helpers, pure selects none, and
Python sources come from the worktree. This is not installed-candidate credit.

The recorder retains setup, call and teardown and cross-checks exact node IDs
against JUnit. Historical maps retain 6,079 project passes and 5,780 upstream
passes. The first duplicate-node fixture was deduplicated by pytest's ordering
hook; its failed rehearsal is retained and the injection priority alone repaired.
No production code or accounting rule changed for that failure.

The full execution composition is still incomplete: next integrate the bounded
controller with source/mode identity, actual exits, cleanup and lease, then build
the operation-semantics matrix before further transport. No device request yet.
FTZ remains defined product behavior, not a demand for gradual underflow.
Production source/test digest, public refusal and all obligation states remain
unchanged. No project collection, GPU, install, commit or watcher this turn.

### Previous readiness checkpoint

Blake approved regression-first sequencing and engineering ownership of the
production design. `trifast-production-readiness-20261009/FABLE-HANDOFF.md`
records 342 passing CPU cases per mode in opposite orders, with no failures,
skips or unexpected device/child attempts; two inherited GPU cases remain
deselected. The full project and upstream device gate has NOT run. Its historical
upstream reference is packet906's exact 9342-node map (5780 pass,3562 skip), not
the older generic5560-pass floor. Eleven negative accounting controls refuse
incomplete results, lost passes and unreviewed population changes. Both generic
runner exit semantics and collection-time device creation require explicit care.

The production design keeps the oracle offline and targets program proofs plus
live structural admission and explicit device numerical behavior. Exact CPU
witnesses show why shapes cannot certify finiteness or underflow behavior.
Next: assess the full-gate execution preparation and numerical design; implement
the operation-semantics matrix before expanding a staged executor. No domain or
tolerance change requested. Production sources/tests unchanged, no GPU or commit,
and full suite/upstream gate in both modes remains required before commit.

### Previous capture checkpoint

Fable closed the sentinel hold (`9a992be7`). The next isolated CPU packet is
`trifast-capture-numerical-policy-20261009/FABLE-HANDOFF.md`, SHA256
`24d6f31b9c05631a4b72dd3b5837eadbae5fb74c4d81d055f9cdd1a4a766f22c`.
Both modes pass 60 controls in opposite orders. A standalone audit regenerates
208 numerical files with cross-mode byte and bound equality. The actual MPS entry
is implemented but unexercised positively; tests use a private CPU transport seam.
One producer sync, seven blocking transfers and an additional snapshot copy are
disclosed. The byte-derived oracle has a predeclared 64-unique-row budget and is
development qualification machinery, not a hot-path or general-input admission
proposal. Production sources and tests are unchanged; public dispatch and framework
compilation remain refused. Next: Fable CPU review, then authenticated transport
composition and explicit executor/publication qualification. No device proposal yet.

### Previous sentinel checkpoint

Fable accepted the allocation/snapshot mechanics with a sentinel-policy hold
(review `dcea8a14`). The repair is ready in
`trifast-admission-sentinel-repair-20261009/FABLE-HANDOFF.md`, SHA256
`27fccded826b51026c9747b0bcf8eb3d46d974fc78539b65fb5b7fd28b833761`.
Scale remains finite; sentinel now requires exact f32 negative infinity. Parent
red controls reproduce both wrong admission and wrong refusal. Both final modes
pass 256 cases in opposite orders and all four inherited mutants are detected.
The module remains unwired and dispatch blocked. No commit or device work.
Next: focused review of this repair, then numerical policy and transport capture.

### Previous allocation checkpoint

Fable accepted the first production package, review `7ce58b24`. Its emitter,
driver refusal and generated bytes remain unchanged. Full project regression and
the upstream ratchet in both modes are required before any commit; the focused
CPU results do not satisfy that gate. New-symbol products have no hardware credit.

The next CPU component is ready in
`trifast-admission-preparation-20261009/FABLE-HANDOFF.md`. The new unwired preparer
checks ordered-expression reach against real CPU allocations, refuses overlapping
backings, snapshots eleven binding payloads, and checks finite input values on
the immutable bytes. Both modes pass the same 244 guarded cases in opposite
orders (142 admission, 102 existing integration/compatibility); all four unsafe
mutants are detected. No unexpected device or child attempt occurred.

This is not complete numerical admission or execution authority. Finite input
can overflow inside the recurrence; MPS capture, general-input numerical policy,
live execution integration and framework compilation remain unqualified. The
snapshot copies 0.53 to 8.53 MB per call, not a speedup. Public dispatch remains
blocked; no installed payload, GPU state or lease changed. Next is independent
CPU component review, then numerical policy and transport qualification before
executor integration. Warm-call optimization remains paused as sequencing.

### Previous planning checkpoint

Fable accepted finite TriFast hardware correctness (review `5a04b26d`). The
external-loader coverage map now carries a separately scoped third window with
six new anchors; all eighteen anchors verify by exact path and hash. The actual
population includes eight poisoned attention operations and four pre-command
refusals across both modes; the sealed review's prose count slips are not copied.
No repeat device run is needed.

Production integration is planned in
`trifast-production-integration-plan-20261009/INTEGRATION-PLAN.md`, SHA-256
`71add6679a20832735198978877b85cc417c637b9eee7302b62296e1913ff3ad`.
A read-only census binds sixteen source files. The first implementation packages
the accepted proofs and renderer and exercises the real emitter entry; it does
not enable dispatch without source-bound geometry, storage, alias and numerical
admission. Native precision must come from the same module. The external fixture's
input hashes are not a production support contract, and framework shader compilation
cannot inherit the accepted metallib's floating-point qualification. The plan
sets adversarial, compatibility, cache/metadata and installed-mode gates before code
changes. Next handoff should be implemented CPU composition, not another unchanged
design acknowledgement. No production source or installation changed; no public
statement or GPU work. Warm-call remains paused as sequencing only.

## Earlier checkpoints

The reviewed TriFast hardware window completed once in both installed modes:
28 operations, 24 submissions, 22 waits, 32 responses and two normal exits.
Results are ready for Fable in `trifast-device-results-20261009/FABLE-HANDOFF.md`,
SHA-256 `03e52ef2e31c9945a0b1b5c517223ae5a08ff0a23401c64b46c5c336b0fc8ff4`.
The approved verifier and standalone analysis agree on 566 numerical files with
distinct inodes and byte equality across modes. All 2,297,856 finite O/LSE values
meet their accepted bounds; 1,081,344 poisoned-row values are NaN as required.
These values come from repeated finite fixture families, not independent cases.
Both probes observe positive zero from exp2(-inf), NaN, two and negative infinity.
The six uncertainties retire at their first Completed status observation, so no
prolonged in-flight retention is established. Device/queue-before-origin order,
unchanged live images, input/scalar/canary readbacks and joint publication hold.
The window took 43.6 seconds and peaked at 1.39 GiB sampled tree memory, with one
resident worker. Manual availability, fresh preflight and a fifteen-minute
clearance preceded execution; both owned groups are absent and the lease is free.
Results seal: 1,408 files and six anchors. No stop, retry, production change or
watcher. Next is independent results review, then mapping only the accepted finite
external-consumer scope. Public compiler integration remains separate: the installed
emitter still refuses the reporter. No performance or release closure; warm-call
remains paused as sequencing only.

Fable accepted the real composition (review `dc31f5cd`). The finite device
instrument is ready for review in `trifast-device-instrument-20261009/FABLE-HANDOFF.md`,
SHA-256 `fa937a0af23fdac0d31563cbaf35996b062e84bb9a002ff1443233cb69a94c71`.
The proposed window has 28 operations across two serial installed modes: sixteen
positive attention cases, two explicit floating-point probes, four pre-command
refusals and six uncertainty injections; 24 submissions and 22 waits, capped at
ten minutes, one resident worker and 4 GiB sampled tree memory.
Build-only registration is now private to a closure, recording providers refuse
on the production path, and live image observations accompany the unchanged
identity evaluator. The accepted consumer and attention shaders remain unchanged.
The modeled controller reconstructs 566 numerical files and 32 responses; both
modes pass 41 controls each, 19 verifier mutants refuse, three process-loss cases
stop without credit, and 24 clock pairs plus four owned cleanup cases pass.
The bound device-free preflight passes in 28.5 seconds with zero device attempts.
Binding `b1b55089f0a965aff2659613eac04042f415097da76107f2a97f706ae6a4e06f`
covers 6,174 inputs; the local seal covers 2,314 files. Development failures remain
retained. No device execution, review receipt, clearance or lease acquisition.
Next is Fable instrument review, then fresh coordination and authority before a
single finite device attempt. No capability or performance credit yet; warm-call
remains paused as sequencing only.

Fable accepted the bounded oracle and consumer (review `68d2bd1c`).
The real transport composition is ready for CPU review in
`trifast-real-composition-20261009/FABLE-HANDOFF.md`, SHA-256
`811a45490341797e924cbcc598cedb899adb4f26e05caefa5874b5778effa8e1`.
The accepted consumer is unchanged. A controller-owned catalog binds inputs and
bounds; fresh compilation binds exact source and products to the live origin
captured after device and queue initialization. Eleven distinct transport buffers
and two ordered private readbacks preserve joint publication and input checks.
Both installed modes pass 52 composition controls in both orders, 43 inherited
controls each, cold bootstrap, and real compiler rehearsals. Three mutants refuse;
108 numerical files reconstruct independently. All real-object calls used recording
receivers; no device was created or real library loaded. Three development stops
and the exact source versions remain retained. Seal: 831 files, 3,717 anchors.
Next is Fable component review, then a separately bound finite device instrument.
Real NaN propagation remains a device observation, not an assumption. No production
or installed changes, no speedup or release credit; warm-call remains paused.

Fable accepted the compilation gate (review `3c628d6b`). The independent bounded
oracle and CPU recording consumer are ready in
`trifast-bounded-consumer-20261008/FABLE-HANDOFF.md`, SHA-256
`d1836b0466e296adbb05c33a7579eb232c6fe821157ff6d2c05d5c7f050cc2c6`.
The oracle covers 1,024 retained rows, 32,768 O values and 1,024 LSE values,
with maximum widths 5.82e-6 and 1.48e-5 respectively, zero subnormal fallback,
and an exact full replay. Twelve source mutants fall outside those bounds.
The consumer has eleven separate staged bindings and returns both validated
outputs in one immutable result. Both 43-control orders pass, plus exceptional
masking and three consumer mutants; 144 retained numerical files reconstruct
literally. Two fixture failures and a brief preliminary CPU job overlap remain
disclosed. Seal: 1,308 files and 335 anchors. No compiled library was loaded,
no device ran, and no production or installed source changed. Next is independent
review of the bound construction and lifecycle, followed by the real composition
with fresh origin-bound products. No device proposal or release credit yet.
Warm-call remains paused. Earlier checkpoints follow.

Fable accepted the renderer prototype (review `3cc12c7a`). The exact reviewed
padded and masked MSL now compile and link unchanged under pinned Metal 3.2,
no-fast-math and no-contraction flags. The next handoff is
`trifast-metal-compilation-20261008/FABLE-HANDOFF.md`, SHA-256
`6a0efd3b09ebdb928a1bcfb206c3c999e6944ea4c104be92a7c3c7f62241efde`.
Four screens produce identical 7,263/7,279-byte metallibs. Two corrected screens
bind all 74 resource-header files; the first incomplete inventories remain
retained. Actual AIR agrees apart from three path-metadata lines, keeps separate
arithmetic, and explicitly permits denormal flushing and assumes non-aliasing.
Fourteen audit-predicate mutants refuse. Seal: 357 files and 95 anchors.
The numerical design requires independent per-output error bounds before hardware;
the consumer design requires non-alias staging and publication of O and LSE as
one result, without claiming two in-place copies are atomic. Neither CPU consumer
nor interval oracle is implemented yet. No production/install/device changes.
Next: implement and qualify those CPU components; no device proposal yet.
Warm-call pause and release states remain unchanged. Earlier checkpoints follow.

Fable accepted the source-semantics proof and finite CPU oracle (review `ee62db93`).
The next CPU gate is ready in
`trifast-expression-renderer-20261008/FABLE-HANDOFF.md`, SHA-256
`9ba18dccd7b13484a2644b3d0e775289bc96ae1892e8c1de8556310575d57e10`.
Native precision is read from both real dots and matched to the graph from the
same module; IEEE only. Both installed modes emit identical MSL through a
row-local expression renderer, not the legacy coefficient template. Per mode:
82 refusals, eight finite host admissions, 14,770 rendered address comparisons,
28 source-subset lane executions and ten detected rendered-source mutants.
An initial CPU context-lifetime crash is retained and repaired; no device ran.
Audit and seal verify 2,343 files and 597 anchors. No installed or production
change and no Metal compiler qualification. Next: review the CPU prototype,
then device-free pinned compilation and the numerical contract for FMA,
association, transcendentals and tails before any device proposal. Warm-call
pause and obligation states are unchanged. Earlier checkpoints follow.

Fable accepted the complete typed-address proof (review `663db5f6`). The source
scale, recurrence and saved-LSE CPU component is ready for review in
`trifast-source-semantics-20261008/FABLE-HANDOFF.md`, SHA-256
`7d6bdc24725d29076ea36803c62da9724ff13492c616948ba1092dee05c3ae21`.
Both installed modes pass two branch proofs, 66 semantic refusals and 36 finite
O/LSE cases, with 324 numerical files per mode and cross-mode byte equality.
Addresses come directly from accepted full-link records; scale and bias round
before the dot, and a retained f32 counterexample rejects reassociation.
The independent audit reproduces every numerical error and the seal binds
1,205 files and 591 anchors. Installed dot precision remains source-requested
rather than native-enum observed, and the full-tile CPU model is not a GPU or
tail oracle. No production or installed source changed, no reporter MSL exists,
and no device work occurred. Next: component review, actual native precision
capture and an expression-preserving renderer prototype. Warm-call stays paused;
no obligation closes. The prior checkpoints below are historical.

Fable accepted the uniform-expression component (review `5d80bb7f`). The complete
typed-address CPU integration is now ready in
`trifast-typed-address-integration-20261008/FABLE-HANDOFF.md`, SHA-256
`9cdfe0e8a9eb4209acb7691f74fca3890654ef6e7e5de17198b43e98b83023f4`.
Each installed mode passes 14 complete role/branch address records, 14,770
arithmetic comparisons, 25 adverse controls and four same-lowerer compatibility
emissions. The isolated three-function detector splice now reaches the unchanged
runtime softmax-scale refusal in both branches; the baseline installed emitter
still refuses at Q strides. Def-use, ownership, axes, mask-head signed division
and ordered i32 addptr links are bound. PADDED LSE's (32,1) shape remains distinct
from the masked (32,) shape. A retained earlier proof fails definition-after-use.
Audit and seal verify 232 files and 587 anchors. No production or installed
change, device activity or emission authority. Next: review the address integration,
then prove the exact scale/bias rounding and remaining mask/recurrence/LSE path
with independent O/LSE oracles. Do not feed these expressions to the legacy
widened renderer. Warm-call pause unchanged. The checkpoint below is historical.

Fable accepted the reporter-exact CPU diagnosis (review `60bcfbb8`). The next
CPU component is ready in `trifast-derived-stride-proof-20261008/FABLE-HANDOFF.md`
(SHA-256 `c69914fc6898a7abe8af85afc9750e90873917041d56488a6f9a1c3d8c7e3540`).
It represents uniform derived strides with immutable i32 expression trees and
preserves wrap semantics, with a separate model of signed pointer extension.
Each installed-mode process passes 14 real-IR role/branch checks, 5,085 arithmetic
comparisons, 18 adverse controls and four unchanged runtime-stride reference
emissions. A retained predecessor fails the new stale-graph-index regression.
The seal binds 189 local files and 584 anchors; zero unexpected device/child
attempts. No production or installed compiler changed: the exact reporter still
refuses before MSL. Expression witnesses are not complete address proofs.
Next: review this component, then integrate a typed address proof retaining axes,
factorization, mask-head division and each addptr extension boundary before
enabling any new emission. Later LSE/recurrence checks and independent O/LSE
oracles remain required. No GPU window requested; warm-call pause unchanged.

Fable also accepted the grouped external-loader hardware window (review
`2dbb7fdc`): 14 operations, 10 submissions, 8 waits and 132 distinct byte-exact
numerical files in both installed modes. Full and partial grids establish the
reconstructed PR #6 grouped ordering at A[64,32] by B[32,64]. The coverage map
keeps this window separate from the earlier elementwise products, with six new
hash anchors and its own limitations. No unchanged rerun or further review is
owed for this bookkeeping. The grouped preparation history below is now complete
at this finite scope, not a pending execution plan.

Completed source reconciliation retained the issue #4 reporter's exact TriFast fork at commit
`92abd91978774f4a9d8355e5bbd83cb224d7c9b7`, then reconcile its forward source
with the accepted triangle K-layout launcher evidence. The exact fork derives
contiguous strides and rebuilds pointer tensors inside the loop; it is not the
retained runtime-stride reconstruction. Saved LSE, source-block rounding and
the external-consumer path must still be assessed explicitly. Provenance and the
first CPU emission refusal are accepted, not device capability. The 42 historical
toolchain failures still lack individual linked reproductions and stay unassigned.

Fable accepted the finite external-loader hardware result in both private installed
modes (review `708499e4`). All 18 operations passed with 14 submissions, 12 waits
and 112 byte-exact, distinct numerical files. The device-and-queue-before-origin
repair held through real submission and normal close. The coverage map is
`external-loader-coverage-20261008.json`; it includes the exact evidence hashes,
accepted cases and exclusions. This completes that finite correctness task, not
the broad external-runtime row or release. No rerun is owed for unchanged inputs.

All six injected uncertainty cases were already Completed at their first reaper
observation. Prolonged in-flight retention, actual device faults, other buffer
owners, concurrency and broader applications are not covered.

Warm-call optimization remains paused by Blake as sequencing only. No regression
is accepted and no guarantee is weakened. The earlier native-import comparison
plans below are historical, not instructions to resume that rejected candidate.

The September 30 C++ device-cost review already accepts an unresolved result in
all six cells. Its separate resource review establishes identical host staging
for the finite C++/MSL corpus. Do not repeat either unchanged screen or claim
parity, a hardware floor or a C++ speedup. Both broader obligations remain open.

The first uncovered finite external-consumer case is now named: grouped flat-grid
fp16 matmul, A[64,32] B[32,64], from the existing PR6 source reconstruction (not
the reporter's exact kernel). Installed CPU preparation in
`external-loader-grouped-matmul-preparation-20261008` emits identical MSL and
metadata in native and pure modes: four positive emissions and fourteen near-miss
refusals total, zero device/child attempts. This is not a new compiler defect or
hardware credit. The emitted contract has three fp16 buffers, nine int32 scalar
buffers and 256 threads per group. The old external adapter cannot be reused
unchanged: it admits two buffers with four-byte elements and 32-thread groups.

Fable accepted the CPU consumer in `external-loader-grouped-consumer-20261008`
(review `feae06c0`). All 69 controls pass in four fresh installed
orders (276 final executions); four targeted mutants fail their named tests.
The parent transaction, reaper, close and exception/finally clauses are retained.
The integer-input oracle makes every fp32 intermediate and fp16 result exact;
it does not qualify general half rounding. A standalone integer/bit-encoding
audit recomputes 24 modeled output files and 36 oracle files. No Metal work.

Fable accepted the real composition (review `af9ee60c`) in
`external-loader-grouped-real-composition-20261008`: 33 transport controls in four
orders (132 executions), two cold bootstraps, three named mutants, and two real
compiler rehearsals on recording receivers. Both installed modes produce identical
metallib bytes under the unchanged nine-child grammar and -fno-fast-math. Real
policy drift at final encoding refuses before commit. Transport changes are limited
to the entry, 256-thread pipeline limit and finite geometry; runtime changes only
three population counts. The accepted consumer and lifetime mechanisms are unchanged.

Two sandboxed rehearsals stopped before compilation because private-cache xcrun
discovery selected the Xcode stub. The same private-cache version query outside
the sandbox found the bound installed toolchain. Device-guarded rehearsals then
passed outside the sandbox without any policy change, installation or download.
Retain that permission-environment requirement for preflight and workers.

The bounded device instrument was accepted by Fable (`ac7d4921`) in
`external-loader-grouped-device-instrument-20261008/FABLE-HANDOFF.md`
(sha256 `a1f4dffd93042208c1a9b3f106ea39f7615dbf13187a594c8ba3307b307ab066`).
It proposes 14 operations, 10 submissions and eight waits across two serial modes,
with 132 numerical files. The final 17-job CPU qualification passed: 73 controls
in four orders (292 executions), cold/pin/clock/real-late-image checks, the complete
controller rehearsal, three process-loss cases and nineteen verifier mutations.
The bound preflight passed outside the sandbox in 18.55 seconds at 1.13 GiB sampled
peak, with recording receivers only. Binding `27d8b215` has 3,751 files; the local
seal has 1,752. Accepted composition/ownership code is unchanged; the controller's
only delta is startup-count accounting. No production or installed source changed.

The hardware window then completed once under Blake's fresh confirmation, its own
outside-sandbox preflight, bound clearance and lease. Both modes passed all fourteen
operations: 132 byte-exact distinct numerical files, ten submissions, eight waits,
eighteen responses, eighteen startup children and two normal exits. Standalone integer
and binary16 reconstruction agrees with the approved verifier. Runtime was 20.124 s,
peak sampled tree RSS 1.39 GiB; no stop, no stderr, empty final census and free lease.
All six uncertainties were already Completed at their first reaper observation;
prolonged in-flight retention and device faults remain unobserved.

Results were independently accepted by Fable (`2dbb7fdc`); the retained handoff is
`external-loader-grouped-device-results-20261008/FABLE-HANDOFF.md`
(sha256 `0e23053f1249c019a0a24ac354c5f7d9453c5516a3c44ce02a52ba665f957e4f`).
The 377-file results seal, five anchors and audit replay verify. The accepted finite
coverage is now mapped, and the exact TriFast source is the next CPU preparation target.
No unchanged rerun, performance claim or broad release closure is proposed.
Full grid (4,1,1) checks every tile; partial grid (2,1,1)
distinguishes grouped order from another bijection. The first proposed three-tile
discriminator stays retained as wrong; the first consumer order also stays retained
with two fixture-traceback retention failures, repaired in tests only.
No production change was needed. The 42 historical toolchain failures
remain unlinked to the proposed capability-probe cause, not 42 confirmed current
defects or 42 closed cases. M1 and exact reporter-source coverage remain separate.
No new device window or unchanged benchmark follows from this checkpoint.

## Historical checkpoints through October 2

- October 2: the approved default-policy comparison completed once under Blake's
  fresh compute clearance. All1,200 calls,32normal closes and7,968distinct backing
  files pass the unchanged verifier and independent raw audit. Results for Fable:
  `native-import-default-comparison-results-20261002/FABLE-HANDOFF.md`
  (48634aa0),seal10,727; binding1,324/preparationseal60 unchanged.
  All32primary cells are unresolved, not parity; envelopes639–4,591microseconds,
  with no null false positive. Exact targets and contrasts remain visible.
  Explanatory post-block diagnostics show added native submission and retirement
  cost in all8blocks, not a qualifying public-call effect. After results review,
  investigate counted per-call import/bind/retirement overhead and safe amortization
  rather than repeat the unchanged timing design. No optimization implemented yet.
  Window303.739s, sampled peak summed RSS2.00GiB; groups absent and lease free.
  Compute released, no watcher, no production/default/package changes.
  Python warm-call and C++ performance/resource obligations remain OPEN and owed.

- October 2: Fable accepted the resident correctness prerequisite (b7cc77a6).
  The executable default-policy 1,200-call comparison is now implemented and
  ready for Fable instrument review, not hardware execution. Handoff:
  `native-import-default-comparison-instrument-20261002/FABLE-HANDOFF.md`
  (8473ae05), binding1324/seal60. All89CPU controls pass both final fresh
  orders with zero device/real-child attempts and no residual ownership.
  Seventeen inherited sources remain byte-identical; raw timing uses the
  accepted helper, witness-only ABI accounting separates observer probes,
  and the modeled controller/verifier covers all32workers/1200calls/7968files.
  Development repairs and superseded receipts remain visible. No GPU,
  authority, production/default change or watcher. Next: instrument review,
  fresh bound identity preflight, then a separately coordinated timing window.
  No performance or resource obligation is closed by this preparation.

- October 2: the reviewed rev2 resident correctness window completed once.
  Both routes passed nine calls and normal close, with exit zero and no retry.
  Independent literal-byte audit verifies168distinct files, pool reuse,
  expected assertion publication, native observed ownership balance and exact
  cross-route bytes. Real native preparation/calls preserve the repaired
  image-before-identity ordering. Both groups absent and lease free.
  Results for Fable: `native-import-resident-correctness-results-rev2-20261002/`
  `FABLE-HANDOFF.md` (53ee5214), seal240; binding1260/preparationseal59 unchanged.
  The old stopped run remains stopped. No performance/resource closure.
  After result review, implement the accepted1200-call default-policy comparison
  with the repaired order/pins/population/abort handling and separate timing
  authority. No further setup experiment is proposed; no watcher runs.

- October 2: the resident correctness window stopped after nine original calls;
  native first call correctly refused a stale identity after late runtime loading.
  Original84 backing files and normal close verified; whole window not passed.
  Controller signal permission error and later independent absence/lease-free
  check are retained separately, with no GPU-cancellation or successful-reap claim.
  Implemented the early-load repair, instance launch pin, exact raw-file census
  and bounded denied-signal cleanup before handoff. Real sanitized identity
  probes discriminate late/early; exact parents fail two new regressions;
  73 controls pass both fresh final orders with zero device/child attempts.
  Successor handoff: `native-import-resident-correctness-preparation-rev2-20261002/`
  `FABLE-HANDOFF.md` (5f783b40), binding4bf91f34,1260files,seal59 verified.
  Await combined Fable stop/repair/instrument review, then manual device check,
  immediate bound identity preflight, fresh clearance and lease for18calls.
  No successor run, automatic retry or watcher. All performance obligations
  remain open; the separate1200-call comparison is still next after correctness.

- October 2: Fable accepted the finite setup envelope (review f19cdd28).
  The next resident correctness instrument is implemented and CPU-qualified:
  `native-import-resident-correctness-preparation-20261002/FABLE-HANDOFF.md`
  (8b042044). Two sequential workers, 18 proposed calls, no timing credit.
  Both finalized orders pass 63 controls with zero device or child attempts;
  modeled controller and raw-file verifier pass, first-failure stop is exercised.
  Binding1241 and seal46 verify. No hardware was run or authorized here.
  Ready for Fable instrument review, then fresh preflight/clearance and leased
  correctness execution; the separate 1,200-call comparison stays next.
  No performance obligation closes and no watcher was created.

- October 2: the approved rev3 setup-only window completed after the required
  manual device check and bound identity preflight. Results for Fable review:
  `native-import-setup-results-rev3-20261002/FABLE-HANDOFF.md` (5a906f36).
  All32 workers closed normally and exited0, eight four-resident blocks,
  zero backend launches;192 MPS initializations are framework work only.
  Measured window272.568s and sampled summed RSS peak1.85GiB supply the missing
  setup planning envelope, not a paired performance or physical-memory bound.
  The repaired inherited-name rule was exercised once. Approved verifier and
  independent raw audit agree; resultseal228 and binding1192 verify.
  All groups are gone and lease free; other session's four-core CPU window
  may now start. Next after result review: exact-composition correctness and
  the accepted default-policy raw comparison instrument under separate authority.
  No performance obligation closes and no watcher remains active.

- October 2: the stopped setup monitor now has an implemented, CPU-qualified
  rev3 successor, ready for one combined Fable repair/instrument review:
  `native-import-setup-preparation-rev3-20261002/FABLE-HANDOFF.md` (d26a1f93).
  Constructor-return and actual-ps-end timestamps constrain the inherited
  framework-image name to a creation-overlap observation; PID, exact argv,
  topology, startup phase and all other refusal boundaries remain mandatory.
  Eighty guarded controls pass in both orders. Real identity/admission/refusal
  checks pass, and an eight-worker CPU stress probe naturally captures the
  transition: old matcher refuses, successor accepts, post-constructor mutation
  refuses. Ten composition/controller/authority components are byte-identical.
  Audit repeats, binding1192 and seal87 verify; earlier lanes are untouched.
  No device work or execution authority. Review is still required before a
  new setup window, with the prior manual-device-check and fresh-preflight
  requirements preserved. No watcher is left running. Performance and resource
  obligations remain open; combining repair and tests does not weaken review.

- October 2: the approved rev2 setup execution stopped during fourth-worker
  startup after the required real-identity preflight passed. Three workers
  reached ready; an exact journaled xcrun child was sampled as `(Python)`,
  which the command-name predicate refused. No retry or instrument edit.
  All four workers were killed/reaped, groups absent, lease observed free.
  Partial setup observations only; no normal teardown or whole-window credit.
  Stopped evidence and proposed narrow repair boundary are sealed in
  `native-import-setup-results-20261002/FABLE-HANDOFF.md` (891561ce).
  Audit twice agrees; approved verifier refuses the stopped population;
  binding1170 and preparation seal74 are unchanged, results seal44 consistent.
  Next: Fable review of the stop and constructor-return timestamp proposal.
  A successor is not yet implemented or approved. No task-owned watchers run.
  Python warm-call and C++ performance/resource obligations remain open and owed.

- October 2: the setup instrument's descendant-process blocker is repaired in
  `native-import-setup-preparation-rev2-20261001/FABLE-HANDOFF.md` (e3726ff5),
  binding 345327d5. Awaiting Fable review, not hardware approval. Exact seven-query
  startup journals, sampled command/state/PID matching, after-ready refusal and
  post-reap empty-group verification preserve the purpose of the held gate.
  Sixty-nine guarded controls pass in both orders. Two real CPU-only identity
  probes pass; the controlled-lifetime witness exposes all seven children and
  reproduces the old gate's refusal on nine samples. Live negative children refuse.
  The accepted composition, pins, authority and limits are unchanged; no GPU,
  production edit, clearance or receipt. Overnight CPU work stayed within available
  resources. Next: Fable instrument review, fresh setup authority/lease, then the
  bounded 32-process measurement. Exact-composition correctness and raw default-GC
  comparison remain separate; all performance/resource obligations stay open.

- October 1 Fable two-family component approval a3a2ed4a accepted. The bounded
  setup-only instrument is ready in
  `native-import-setup-preparation-20261001/FABLE-HANDOFF.md` (8e2614b6).
  Thirty-two fresh workers, at most four resident, exact accepted assignment;
  zero backend calls and warmups. Six MPS tensors per worker are explicitly
  framework GPU initialization for the later clearance, not CPU-only work.
  Final 62 controls pass in two fresh orders with zero device attempts and no
  residual owners; 133 source inputs agree. Consumer referent/value pins close
  the swapped-launcher gap. Synthetic controller/verifier and first-failure
  replay pass; resource observations remain unmeasured on real workers.
  Audit rerun agrees, binding1124 files and seal35 files verify, Ruff clean.
  No GPU run, production/default change or execution authority created.
  Next Fable instrument review, fresh coordinated setup clearance and lease,
  then the one setup measurement. Correctness on the exact composition and
  raw comparison remain separate gates. Literature scan changes no target or
  schedule. Python warm-call, C++ performance/resources and original goals
  remain open and owed.

- October 1 the resident two-family session and raw adapter are ready for
  Fable's CPU component review:
  `native-import-two-family-session-20261001/FABLE-HANDOFF.md`.
  One shared prepared native device and pool, two fixed package/kernel owners,
  no per-call session switch; unchanged launcher, bridge and native invocation.
  Fifty controls pass in both fresh orders and five mutants refuse, with zero
  device attempts and no pending owners. Final source-bound audit and seal agree.
  Raw invoke/timer bodies retain session checks inside the two clocks.
  This is CPU mock evidence, not MPS correctness or timing qualification.
  Next after review: separately bounded32-process setup measurement,
  exact-composition correctness and executable comparison review/clearance.
  Python warm-call and C++ performance/resources remain open and owed.

- October 1 Fable accepted the default comparison design with an assignment
  repair, now implemented in
  `native-import-default-comparison-preparation-20261001/CHECKPOINT.md`.
  Groups01/23/02/13 and directions F,F,R,R remove the session-half confound;
  an explicit control rejects the parent assignment. Estimator AST unchanged.
  Two CPU replays verify the1200-call request stream and its eight mutants,
  with1024 raw primaries and first visits retained separately.
  This is a preparation checkpoint, not an executable instrument or clearance.
  Next: CPU-qualified resident two-family native composition and no-wrapper
  raw adapter, then instrument review. Exact-composition correctness and a
  separately authorized measured32-process setup phase precede timing.
  All original obligations remain open; no performance conclusion follows.

- October 1 default-policy comparison design is ready for Fable:
  `native-import-default-comparison-design-20261001/DESIGN.md`.
  Proposed1200calls,1024raw timed, both families/all4positions; no new run.
  Four route-fixed fresh workers per block cross slot assignments and enumerate
  all predecessor/receiver edges. Matched own-route nulls gate raw full-call
  contrasts; post-block cuts are explanatory only. CPU count/support and finite
  nuisance controls pass, not hardware resolution. Actual ordering and the
  changed history-specific calibration rule remain explicit design-review
  questions, not inherited approval. No controller implemented or authorized.
  All original obligations remain open; the primary quantity stays default-GC
  absolute full-call change, never component substitution.

- October 1 Fable GC results review073473a6 accepted; no obligation closes.
  `native-import-within-session-support-20261001/DECISION.md` checks the
  proposed simple per-visit policy extension before spending another window.
  All256 assignments fail full four-context support in three continuing pairs
  under direct common-predecessor matching. This is not a general impossibility
  or resolution claim. No extra GC experiment is booked. Return design effort
  to the actual original/native comparison under default GC, within-session
  with matched null and proved support, preserving full-call quantity and targets.
  That executable successor remains owed, not authorized by these CPU counts.

- October 1 the approved GC-tail run completed once:
  `native-import-gc-tail-feasibility-results-20261001/FABLE-HANDOFF.md`.
  All1216 calls,6144cuts,four clean exits and protected zero-GC-event checks
  pass; binding unchanged, lease free. Independent audit and verifier agree.
  Continuing submission9/16 and retirement16/16 fail the unchanged primary
  rule; first visits separately unqualified. No performance obligation closes.
  Frozen historical thresholds flag993 calls, but the encoding level shifted;
  a separately labelled posthoc current-stratum count is32. No cross-session
  GC effect is inferred. Collection inside the interval is not necessary for
  residual excursions, while outside work and deferred debt remain unbounded.
  Next is Fable results review then a bounded mechanism decision; no rerun,
  larger matrix or target relaxation. Public default-policy cost remains owed.

- October 1 Fable GC design review c7fb5cec accepted. Executable successor
  `native-import-gc-tail-feasibility-preparation-20261001/FABLE-HANDOFF.md`
  is ready for instrument review: 78 controls twice, zero device attempts,
  matching audits and a 1160-file binding. Actual GC builtin pins, an 83-file
  source census, per-worker policy/deferral receipts and full-launcher failure
  controls implement the accepted scope. Same 1216-call population, cuts,
  combined-span rule, first-visit tier and targets; no GPU authority created.
  Suppressed-policy resolution is not default-policy public-call performance.
  Re-enabling does not guarantee debt repayment; arm-dependent deferred cost
  remains owed. Next: independent instrument review, then fresh coordination
  and clearance. All release obligations remain open.

- October 1 Fable results review3c9b6e81 accepts the launch-split result.
  `native-import-gc-tail-design-20261001/DESIGN.md` requests review of the
  bounded per-call cyclic-GC discriminator and isolated CPU prototype.
  `gc.disable`, not object freezing; restore entry state, preserve consumer
  exceptions, zero callback events and unchanged stats required. Actual3.14.4
  probe shows why counters alone miss incremental work; failed first controls
  and sources retained. Revised16 controls pass twice, no device attempts.
  Combined submission, first-visit tier,1216 calls and both targets unchanged.
  Fixed historical encoding thresholds and commit bins are diagnostic only;
  neither fewer tails nor GC deferral would establish a public-call saving.
  Next after review: full-launcher failure/pin controls and executable binding,
  then separately reviewed/cleared hardware. All original obligations stay open.
- October 1 the approved launch-split same-code run completed once:
  `native-import-launch-split-feasibility-results-20261001/FABLE-HANDOFF.md`.
  All1216 calls,6144 cuts,four clean exits,zero warm pooled allocations and
  real selector pins pass; unchanged verifier and independent audit agree.
  Continuing submit15/16 and retirement16/16 do not meet the all-cells rule.
  The failed combined-submit cell passes24us but misses its stricter20.330us
  target. No target waiver, pooled cells, repeat run or hardware-floor claim.
  Median continuing submit envelope6.348us versus wait22.338us; full134.255us.
  First visits separately unqualified. Both submission subpieces passing is
  diagnostic only, not permission to replace the combined-span qualification.
  ASIC coordinated; ordinary UI background disclosed; lease free and released.
  Fable results review requested before the next bounded design decision.
  Python warm-call and C++ performance/resource, reporters and M1 remain owed.
- October 1 Fable reviewd38c3c29 accepts the overhead gate only. Executable
  `native-import-launch-split-feasibility-preparation-20261001/FABLE-HANDOFF.md`
  is ready for instrument review, binding635af7cf over1071 files. Accepted cuts
  are byte-identical;1216 calls/all1024timed traced, two-tier decision and both
  targets unchanged. Eight separate series retain gap/wait/launch/full. Actual
  command-class native descriptors and queue are checked around every timed call
  and tied to existing witness calls, with no extra command. Report refusal is
  a poisoned stop before verification reads, never a dropped sample.63 controls
  pass in two fresh orders, no devices or owners; audit consistent. No GPU,
  receipt or clearance. Next: Fable instrument review then fresh coordination.
  All Python/C++ performance and resource obligations, reporters and M1 stay open.
- October 1 the lower-overhead recorder's CPU screen passes, pending Fable review:
  `native-import-launch-split-low-overhead-20261001/FABLE-HANDOFF.md`.
  Only Cuts/Command classes change; six cuts, deferral and ownership preserved.
  37 controls pass twice with no device attempts or residual owners. Independently
  balanced null side/labels and an added tight body yield40000 retained pairs;
  all8 cells pass the unchanged2.4us median/p95 and null gates, worst p952.083us.
  Millisecond outliers remain: no hard bound, trace-neutrality or speedup claim.
  Independent audit and21local/99external seal verify. No GPU or production edit.
  Next is the executable successor after review, with concrete command-selector
  pins and its own instrument review/clearance. All original obligations stay open.
- October 1 Fable prototype review95594623 accepted. The exact-proxy CPU screen
  in `native-import-launch-split-overhead-20261001/FABLE-HANDOFF.md` fails its
  predeclared overhead gate:3/6 cells qualify, with original-arm p95 increments
  up to3.125us against2.4us and two null-bias failures. Native probes pass both
  orders.30 controls pass twice, including the full original launcher under
  exact proxies;30000 pairs retained, independent audit and seal consistent.
  No GPU, rerun, target relaxation or performance closure. Next is a separately
  bound lower-overhead recorder design and CPU screen before the GPU successor.
  The resource wrapper's exit1 after complete output is disclosed, not called
  a clean process receipt. Python warm-call and C++ performance/resource,
  reporters and M1 remain OPEN. Fable review requested on the failed screen.
- October 1 Fable design review313ecb46 is implemented in
  `native-import-launch-split-prototype-20261001/FABLE-HANDOFF.md`.
  Actual-consumer CPU controls cover six cuts, native preparation/launch
  decomposition, witness-only ABI counts29/30 and raw return codes. Required
  two-tier outcome separates continuing qualification from first visits without
  changing targets or population. Clock-error injection exposed a lease trap;
  the recorder now defers its own clock error through the current consumer's
  normal wait/cleanup, raises before copy-back, and preserves consumer-error
  precedence/quarantine. This policy is explicit for independent review.
  Final22 controls pass twice, zero device attempts/pending/owners; retained
  first recorder is red. Independent audit and39local/103external seal verify.
  Fable prototype review and new exact-proxy overhead qualification precede
  the executable successor; no GPU authority or timing claim. All original
  performance/resource obligations, reporters and M1 remain OPEN.
- October 1 Fable accepted the component-feasibility limitation at review1796a738.
  `native-import-launch-split-design-20261001/FABLE-HANDOFF.md` supplies the
  next source-bound design, not an executable GPU instrument. Qualifying old
  cells actually had ONE difference per block, not a median of two. Submit
  return, pre-wait gap, wait return and retirement remain separate, with native
  ABI/status asymmetries disclosed. Proposed all-traced sampling keeps1216 calls,
  both families, positions and orders:16 continuing-history cells get3 pairs per
  block;16 first-history cells retain1 and still gate qualification. No hidden
  pooling or target relaxation; raw neutrality and public-call comparison stay
  owed.14 stdlib arithmetic controls pass;7local/13external seal verifies.
  Fable design review precedes actual-consumer fault/overhead controls and the
  executable successor. No GPU authority, source/default change or performance
  closure. Original obligations, reporters and M1 are unchanged.
- October 1 the approved component feasibility run completed once:
  `native-import-component-feasibility-results-20261001/FABLE-HANDOFF.md`.
  All 1216 calls pass the retained correctness/topology checks; 5376 root spans,
  zero post-warmup pool allocations, four clean exits and free lease. The
  ordinary OS/UI background condition was explicitly disclosed after Blake's
  clarification, not represented as a literally inactive desktop. Independent
  audit and unchanged verifier agree: 0/32 component and 0/64 full-call cells
  resolve both targets; zero null false positives. The launch method accounts
  for a median 93.18% of component time and includes submission/completion wait,
  so the measured component is not a pure CPU-dispatch boundary. Cause of spread
  remains unseparated. Fable results review precedes any successor; no automatic
  rerun, larger matrix or weaker target. Experiment complete, performance NOT
  closed. All original performance/resource goals, reporters and M1 stay open.
- October 1 Fable review08b7c3f9 accepts the root-only CPU overhead screen.
  `native-import-component-feasibility-preparation-20261001/FABLE-HANDOFF.md`
  supplies the successor executable controller and estimator for review. The old
  mode flip left 16 history/split combinations missing; cohort-plus-family mode
  assignment now gives each order/history cell four even calibration and four
  odd held-out blocks without changing the 1216-call population. Strict worker
  alternation includes witnesses; exact ten/eleven root entries are required.
  Component and full-call series retain separate envelopes and both signed
  24us/exact2% tests. Full traced/raw contrasts are descriptive, not causal.
  Final 63 guarded controls pass twice; audit agrees; 1060-file binding verifies.
  No GPU authority or execution. Fable instrument review precedes fresh idle
  coordination and lease. All performance/resource obligations remain OPEN;
  original goals, reporters and M1 remain unchanged.
- October 1 Fable review1da9713a accepted the corrected equation and inside-span
  budget. `native-import-component-overhead-20261001/FABLE-HANDOFF.md` repairs
  zero component spans, proves parent-red/successor-green, and passes 31 guarded
  controls twice. The single shared-host CPU screen passes all ten cells:
  11x median 0.462us, worst 11x p95 1.375us, against 2.4us. These are descriptive
  per-entry statistics, not a public-call or hard overhead bound. Independent
  audit and 22-local/33-external seal agree; sealed parent unchanged. A CPU-only
  schedule proposal proves strict alternation including witnesses and balanced
  traced/raw calls, preserving the 1216-call population. Matching by predecessor
  and held-out order plus the executable successor GPU controller remain owed.
  Fable repair/screen/schedule review is requested; no GPU authority or run.
  Python warm-call and C++ performance/resource stay OPEN and owed; original
  goals, reporters and M1 are unchanged.
- October1 Fable accepted the feasibility result at review0a1fd54a.
  New `native-import-component-prototype-20261001/FABLE-HANDOFF.md` supplies
  CPU-qualified component tracing and an engine attachment, not a GPU controller.
  The proposed faithfulness subtraction was mathematically wrong: full minus
  component is remainder, not the original full call. A deterministic witness
  accompanies the full-to-full correction. Property checks and nested union
  accounting are covered; omitted descriptor-selection costs are disclosed.
  Final23 controls pass twice, zero device attempts, audit/seal consistent.
  Fable review of the correction, observer budget and prototype precedes any
  overhead screen or successor GPU protocol. No performance closure, weaker
  target or automatic larger matrix; original priorities remain unchanged.
- October1 the approved same-code full-launcher feasibility run completed once:
  `native-import-feasibility-results-20261001/FABLE-HANDOFF.md`.
  All1216 calls pass byte checks; resident pools allocate nothing after warmup,
  all128 bracketing witnesses pass, four workers exit cleanly, and the lease
  is free. The unchanged verifier and independent audit agree;435-file result
  seal verifies. The2% research target fails in all16 cells, with zero null
  false positives. Envelopes137–583us exceed the23–27us offsets. This is an
  instrument limitation requiring a reviewed mechanism-based redesign, not a
  hardware restriction or an accepted weaker target. Fable results/design
  review is next; no import comparison or automatic larger matrix is cleared.
  Python warm-call and C++ performance/resource obligations remain OPEN and
  owed. Original goals, reporters and M1 remain unchanged.
- October1 Fable approved the exact1216-call feasibility instrument at review85fd6b80.
  The1048-file binding rechecks and the review receipt is recorded in
  `native-import-feasibility-results-20261001/REVIEW-RECEIPT.json`.
  Execution is held for quiet-window coordination: ASIC Astra confirms its own
  lane idle but cannot confirm the independent ASIC Fable lane. Blake's
  confirmation is pending. No clearance or GPU run exists; refresh live
  statistics and coordination before execution. This does not close any
  performance obligation or change the original work sequence.
- October1 Fable's cost-model additions are implemented in
  `native-import-feasibility-preparation-20261001/FABLE-HANDOFF.md`.
  The original-only same-code executable protocol has1216 intended backend
  launches:1024 timed,128 witnesses,64 warmups in two persistent-worker
  cohorts. Pooled allocation is forbidden after warmup; assertion status
  allocations remain explicit. Per-position/order held-out null and two-sided
  synthetic2% resolution rules precede any import comparison. Expanded CPU
  census records session/ticket/second-reservation work. Final28 controls pass
  twice with zero device attempts;1048-file binding consistent. Fable instrument
  review, fresh quiet-window coordination and lease precede hardware. No
  execution authorization or timing result exists. Original priorities unchanged;
  Python warm-call and C++ performance/resource remain OPEN and owed.
- September30 Fable review59eba71a accepted the rev2 full-launcher checkpoint.
  `native-import-equal-boundary-design-20260930/FABLE-HANDOFF.md` now records
  the CPU operation census and requests review of the cost/measurement design.
  Import makes29/30 native ABI entries versus12/13 original Objective-C
  dispatch messages plus2 size constructions, not comparable time units.
  Both retain original MPS staging; net saving is unmeasured. Three count
  mutants refuse; zero device attempts and no residual owners. The35-file
  seal verifies. Equal-boundary measurement must preserve session lifetime,
  assess null resolution and include certified-pool overhead. No timing or
  new correctness execution was performed. Original priorities unchanged;
  Python warm-call and C++ performance/resource remain OPEN and owed.
- September30 approved rev2 full-launcher correctness completed once:
  `native-import-full-launcher-results-rev2-20260930/FABLE-HANDOFF.md`.
  All15 calls pass, including six exact MPS driver/import pairs, real
  Metal-owned fallback, capacity retirement and actual existing-buffer
  refusal. Independent audit rehashes174 backings, with156 shared prior files
  unchanged; final provider sweep and native/registry retirement pass.
  New clearance and coordination, child exit0/reaped, lease free, no retry.
  The stopped predecessor remains partial and untouched. Fable results
  review requested. Next is the separately reviewed equal-boundary
  comparison including producer waits, staging, assertions and copy-back.
  No performance/resource claim or installed-JIT qualification follows.
  Python warm-call and C++ performance/resource remain OPEN and owed.
- September30 the approved full-launcher window stopped at the capacity
  refusal after13 complete call records. All six paired cases and the real
  Metal-owned fallback produced exact retained bytes, but the missing capacity
  row, existing-buffer case and final provider sweep prevent whole-window
  credit. No retry; child reaped and lease free. CPU diagnosis reproduces an
  instrument-model error: terminal failed leases await the normal registry
  reaper, while the worker demanded the raw dictionary empty too early.
  See `native-import-full-launcher-results-20260930/REPORT.md`.
  The narrow successor observes that existing retirement only after the proved
  null-submission refusal, retaining strict stops for pending/unknown state.
  Final42 CPU controls pass twice; binding30bb144a verifies1,021 files.
  `native-import-full-launcher-instrument-rev2-20260930/FABLE-HANDOFF.md`
  requests diagnosis/delta review before a new clearance and GPU window.
  No production or accepted-instrument edit. Python warm-call and C++
  performance/resource obligations remain OPEN and owed.
- September30 Fable approved the full-launcher integration at CPU scope.
  The separately reviewed hardware instrument is now requested in
  `native-import-full-launcher-instrument-20260930/FABLE-HANDOFF.md`.
  Final40 controls pass in two fresh orders, zero devices; binding2365fa8f
  verifies1,016 files. Planned15 calls and13 intended backend launches plus
  Torch producer/copy-back work: six paired cases, Metal-owned fallback,
  capacity refusal and actual existing-buffer refusal. Per-call storage routes,
  separate device/framework assertion oracles and real consumer identity are
  required; the metadata is explicitly AOT, not JIT/installed qualification.
  No hardware execution or clearance yet. After instrument review, Astra
  owns fresh ASIC coordination/statistics, the bounded correctness window and
  result review. Python warm-call and C++ performance/resource remain OPEN
  and owed; no timing claim or change to original priorities.
- September30 Fable accepted the real-import hardware result at review55b6e11e.
  The next full-launcher CPU checkpoint is returned in
  `native-import-full-launcher-20260930/FABLE-HANDOFF.md`: 31 controls in two
  fresh orders and five rejected mutants. It uses the complete unchanged
  constructor/call body and real Torch CPU tensors with the native mock and a
  synthetic identity provider. New attachment gates package/pipeline identity
  and keeps the canonical registry and utilities owner through quarantine.
  Fable integration review is next, then a separately reviewed real MPS
  producer/copy-back and fallback correctness instrument. No hardware or
  performance credit from this CPU increment; full JIT/installed scope remains
  unqualified. Python warm-call, C++ performance/resource and all broader
  obligations remain OPEN and owed. No production source or defaults changed.
- September30 real-buffer-import GPU correctness passed in
  `native-import-correctness-results-20260930/FABLE-HANDOFF.md`: 13 calls,
  six byte-exact driver/import pairs and the capacity refusal with no submission.
  All21 anonymous-mapping imports matched real extraction and witnessed canonical
  registry refusal; all78 backing files satisfy the independent byte oracle.
  Fable's approved instrument ran unchanged after fresh coordination/statistics,
  under the lease, with clean exit and release. Results review is requested.
  This establishes extracted-driver/bridge correctness on M4 Max, not the full
  public launcher, a speedup or a resource bound. Full-driver integration and
  separately scoped equal-boundary comparisons remain next. Python warm-call,
  C++ performance/resource and all broader obligations remain OPEN and owed.
- September30 Fable approved the imported bridge at CPU scope, with an explicit
  canonical-registry binding requirement. The real-import instrument is ready in
  `native-import-correctness-instrument-20260930/FABLE-HANDOFF.md`: 98 bound files,
  23 passing guarded CPU controls, planned 13 correctness calls with 12 intended
  submissions and no timing. Canonical overlay identity and nested admission
  refusal are witnessed; six safe driver/import pairs precede a capacity-based
  import-only refusal. Native ABI observations and source-derived internal
  command/commit arguments are distinct. No full JIT or installed-backend claim.
  Instrument review is requested before fresh ASIC coordination, statistics and
  the project lease. No GPU run or execution clearance yet. Python warm-call,
  C++ performance/resource and broader obligations remain OPEN and owed.
- September30 imported-buffer driver bridge CPU checkpoint is ready in
  `native-import-driver-bridge-20260930/FABLE-HANDOFF.md`. Final44 controls pass
  in two fresh orders using the pinned driver retain/launch/assertion fragment;
  five bridge mutants are rejected, with no device attempts. The bridge owns
  the certified pool and registry reservation through native uncertainty, then
  discards abandoned acquisitions without late framework publication. Fable's
  accepted adapter findings and suggestions are carried into this composition.
  Independent bridge review is requested before Astra's separately reviewed
  real-import GPU correctness controller. No full public-launcher, installed,
  real PyObjC or performance credit follows. Python warm-call and C++ performance
  and resource obligations remain OPEN and owed; reporters/M1 unchanged.
- September30 isolated imported-buffer adapter CPU checkpoint is ready in
  `native-import-adapter-20260930/FABLE-HANDOFF.md`. Generation-checked owned
  allocation history, certified imports, transaction lifetime and uncertainty
  controls pass 26 tests in each of two fresh-process orders, with no device.
  Fable review requested at ownership/component scope. Astra next owns public
  driver bridge integration, the real handle extractor and the reviewed GPU
  correctness instrument. No performance or real-import credit from CPU tests.
  C++ performance/resource obligations remain open and owed.
- September30 import successor accepted at its explicitly certified-domain CPU
  scope in `astra-native-buffer-import-rev2-review-20260930/REVIEW.md`. F1/F2
  from the prior entry are closed at that scope, not for arbitrary external
  memory. Independently reproduced 113 mock controls, 12 ARC receiver controls,
  green/red gates and exact 15-file patch reconstruction. Astra now owns the
  isolated adapter: establish allocation provenance at creation, preserve it
  through reuse, and retain pool leases across every native handle. A warmed
  pool entry is not its own provenance proof. Then a separately reviewed real
  import correctness instrument and coordinated GPU window. No Fable repair
  pending; C++ performance and all broader obligations remain open and owed.
- September30 native-buffer import prototype reviewed at CPU scope. Existing
  105 mock controls, 12 real-ARC fake-receiver controls and green/red gates
  reproduce. Adapter integration is held on the backing-alias proof and explicit
  ownership of bound but unsubmitted invocations. The executed remapped-backing
  witness covers the core/mock, not real Metal import. Fable's successor request
  is `astra-native-buffer-import-review-20260930/REVIEW-AND-HANDOFF.md`.
  Astra retains adapter integration and later reviewed GPU qualification.
  Clarification superseding older closure wording: the device-cost experiment
  is complete, but C++ performance and broader performance/resource obligations
  remain OPEN and owed. Native-runtime sequencing does not retire them. The
  observed timestamp granularity is not a hardware waiver or a performance floor.
  Astra owns the next discriminating C++ mechanism and sensitivity calculation.
- September30 device-cost screen completed after Blake confirmed both ASIC lanes
  idle: 1,728 timing commands plus the prior 92 correctness commands, all passing.
  All six family/mode comparisons are unresolved against the same-binary null.
  See `cpp-device-cost-results-20260930/REPORT.md` and its sealed results handoff.
  Verifier and independent raw recomputation agree; no samples removed, no retry.
  Workers retired, lease free, ASIC hold released. Stop this bounded screen;
  no public-call, equivalence or floor claim and no automatic timing expansion.
  Fable accepted the results in review96b45867; this bounded screen is closed.
  The observed41.667ns timestamp lattice was independently reproduced. Native
  runtime resumes in `native-runtime-resume-20260930/`: four untimed mock cases
  count the accepted adapter's allocations/copies/ABI calls. Next is an isolated
  shared-Metal-buffer import ownership/CPU prototype, not another kernel family
  or a speed promise. Fable handoff is prepared; Astra owns adapter/comparison.
  Import must preserve backing aliases, external owners and uncertain-work
  quarantine. Later GPU work needs a fresh reviewed instrument and coordination.
  Reporter/M1 and the original wider plan remain unchanged.
  Defaults, artifacts and original obligations remain unchanged. This supersedes
  the execution holds in the historical preparation entries below.
- September30 the independently approved device-cost instrument passed its real
  correctness prerequisite: 92 commands, both compiler binaries under both
  relocated interpreters, exact complete backings and Completed status throughout.
  See `cpp-device-cost-results-20260930/CORRECTNESS.md`. Verifier and independent
  audit reproduce; workers retired and lease free. Timing has not started and is
  held for separate quiet-window coordination and fresh phase-specific clearance.
  No public-call or speed claim; original priorities and defaults unchanged.
- September30 Fable accepted the resource result at scope. The next planned
  diagnostic is implemented in `cpp-device-cost-instrument-20260930` and awaits
  instrument review: exact qualified binaries, a 92-command direct-replay
  correctness prerequisite, then a separately cleared 1,728-command device
  timing window with a same-pipeline null. Both-order effects must exceed each
  block's null envelope; otherwise the screen reports unresolved and stops.
  Final preparation: 22 CPU controls per mode, two guarded import preflights,
  zero devices, 604-file binding. No GPU run or clearance yet. This is device
  replay, not public-JIT latency or throughput. Original priorities unchanged.
- September30 resource window completed in
  `cpp-resource-results-20260930/REPORT.md`: 92 installed GPU calls, four
  fixed-policy MSL/C++ and native/pure cells, exact outputs and whole-backing
  checks. All four resource profiles agree: five allocations per process, then
  reuse; 49,160 bytes retained in the observed free pool; add copies 2,048 bytes
  per call and masked cases 4,864. Capacity is not copied bytes or whole-process
  memory. No latency claim; setup spans are unpaired observations. Fable results
  review requested. Next is the already planned, separately reviewed paired
  device-cost diagnostic on the same computations, with its own quiet-window
  coordination. No public-wall calibration restart or speculative optimization.
  Source, defaults, artifacts and all wider obligations remain unchanged.
- September30 bounded C++ distribution checkpoint closed after independent archive
  review. `cpp-private-baseline-archive-20260930/BASELINE.json` records preferred
  extension archive789af3bb with unchanged image8191b9c8; provider0ff1048a/73d92785
  and core payloads stay fixed. Earlier baselines/receipts remain untouched.
  Archive replay is not source-to-native extension reproducibility or release
  readiness. Use the corrected ZIP writer at the next core repack, no new lane now.
  Performance/resources resume in `cpp-performance-resource-entry-20260930/NEXT.md`:
  receipt census2119inputs/244dispatches and7CPU controls, no new GPU result. Next
  implement/review a bounded cost instrument on three existing fp32 kernels,
  separating device, allocation/copy and cold/setup boundaries. No speculative
  optimization or repeat public-clock calibration. Fresh ASIC coordination,
  statistics and lease precede execution. Native runtime/reporters/M1 and all
  original obligations remain; no scope/default/publication change.
- September29 cold installed review and bounded host-component checkpoint complete:
  `host-component-consumer-20260929/REPORT.md`. Fable accepts the native/pure cold
  increment (~1.1–1.2s representative first-result saving, extra CPU disclosed).
  Corrected reviewer count19/24, not21, within1.0–1.8s; all24pairs positive.
  Bounded modeled installed host preparation:12,288 CPU samples, median C-P+0.443us,
  mixed signs/comparable same-code noise, independently reviewed as unresolved.
  Stop applied: no more wall-clock calibration or matrix from this line; no
  public-call/nonregression claim. The ~87ns lookup increment stays disclosed.
  Exact reviewed eight-path ownership delta is now integrated into primary;
  combined ownership/inventory406passes under device guard, source de7b06e6.
  Package/test bytes match the tested cold-installed source composition; full
  current gates remain owed. Next existing compiler/provider and runtime/reporter
  obligations, with fresh resource coordination before any heavy build/GPU work.
  This checkpoint changes progress, not priorities, criteria or default-off policy.
- September29 authorized cold-identity window complete:
  `astra-cold-parallel-integration-20260929/RESULTS.md`. Integrated exact repaired
  production e912e7fe; source5b1d948a. Focused214passes twice and broader host-aware
  525passes/22skips; earlier sandbox/guard failures cleared, receipts retained.
  Ten fresh processes, 15inventories/89,782records exact plus key/stamp. Identity
  wall medians6.413/5.402s, paired saving1.036s all4positive, CPU+0.308..0.558s.
  Sharedhost/warm pagecache; explicit sourceoverlay on same disk payload, not
  separate installed artifacts or firstkernel result. Lease free, no shaders.
  Fable result-review handback prepared. Next private native/pure artifacts and
  installed cold qualification, no new optimization. Other priorities unchanged.
- September29 successor cold inventory is integrated at scoped CPU acceptance:
  `astra-cold-parallel-integration-20260929/REPORT.md`. F1 failure ordering and F2
  measured-source provenance closed; exact production handback e912e7fe. Added26
  permanent cases plus serial fixture. Helper-equipped focused selection214passes
  in both file orders, zero device attempts. Broader historical selection hits
  sandbox OS-query failures and blocked device initialization; full qualification
  remains owed, not replaced by the focus. No fresh timing or installed-wheel
  claim. Blake resource coordination requested for host-aware/fresh-process work.
  Source5b1d948a, all obligations open, no default/policy/plan-sequence change.
- September29 primary cold-inventory review is in
  `astra-cold-parallel-review-20260929/REVIEW.md`: supplied62 CPU controls pass;
  two new controls expose later inline/traversal errors masking earlier pooled
  failures. A genuinely pooled alias control passes. Integration held for that
  narrow repair and the screen/delivered-candidate source-hash mismatch, not a
  rejection of parallel inventory. No production edit, GPU or benchmark.
  Fable's successor handback is prepared in the review; no acknowledgement round
  or new optimization requested. Existing cold/C++/attention/runtime/reporter
  priorities remain, along with the separate bounded host-component closure.
- September28 Fable found the candidate's weak pipeline receipt could not admit
  PyObjC objects. The repair is in `host-packed-abi-pyobjc-repair-20260928`:
  strong pipeline/library receipts scoped to loader MetalUtils, retired by the
  existing unload_module lifecycle; duplicate library loads counted. A weak index
  targets Python utilities owners only, not PyObjC objects, since loader and
  dispatch utilities are distinct. Device-free actual NSObject/Metal options,
  nonweak stand-in dispatch, wrong identities, unload, owner teardown and
  reentrant finalizers pass. New source-bound sweeps305 passed/2 GPU skips per
  dependency environment, still worktree overlays rather than installed or
  independent pure-helper qualification. One production file/three test files
  differ from the reviewed parent; source freeze445entries. Next Fable's small
  delta review then the already planned installed correctness/performance runs.
  Strong receipts persist until matched unload or utils teardown; they are not
  weakly collected pipelines or a size-capped cache. No GPU claim or plan reset.
- September28 host packed-ABI production candidate is in
  `host-packed-abi-integration-20260928/REPORT.md`. Seven modified production
  files plus one new ABI module and two tests; no shader-body change. Audited
  producer/final-source/selected-binary and loaded-pipeline binding, captured
  scalar payload, original logical output indices and existing host storage,
  cleanup and pending ownership. Optional schema3 leaves the common twelve-field
  descriptor intact. Review found and repaired a pre-alias source-hash mismatch
  and a declared/backing width gap; primary fixed signless pointer over-refusal.
  Focused source-bound CPU selections:295 passed/2 GPU skips each in two relocated
  dependency environments, overlaid with current worktree. This is NOT installed
  native/pure qualification. New Packed grants reject width-changing wrappers
  pending proof; same-width bit reinterpretation is not numerical dtype proof.
  Next composed review, then installed public host-versus-compile_shader exactness
  with real source/pipeline/slot witnesses and full-call performance retention.
  No GPU run, new latency claim, default promotion, commit or publication here.
  Earlier K-layout public timing still establishes no public-call gain; all wider
  obligations, cold identity, C++ provider, native runtime and reporter work remain.
- September28 host packed-ABI foundation in
  `host-packed-abi-recovery-20260928/REPORT.md`: device-free immutable declared
  layout/payload prototype,53 guarded controls per environment, current-helper
  AST red control2fail/5pass. No production edit or host recovery yet. Sol's
  census finds three packed emission sites and distinguishes positional denseFA,
  C++ AIR, MLX and replacement contracts. Next explicit producer/binary metadata
  and safe host materialization, then composed/installed comparisons; do not infer
  packed ABI from family/count. Full-call retention timing remains required.
  Fable's timing review is accepted: **no established public-call K-layout gain**
  on this host, pure/runtime only a weak directional hint. The original8/8 rule
  and measurements below are retained, not retroactively changed. The new lane's
  TIMING-REVIEW-CLOSURE records descriptive null overlap without calling it a
  formal significance test. Default-off, cold and other obligations unchanged.
- September28 installed public K-layout timing completed in
  `triangle-k-layout-public-timing-20260927/REPORT.md`, unchanged private packages.
  1,448 warm calls/960 samples: native contiguous0.9931x and runtime1.0250x,
  pure contiguous1.0172x are mixed; pure runtime1.0307x/15.916us saved has8/8
  faster blocks, with noisy nulls and order-stratum caveats. No general win.
  Cold24 fresh-process calls6.193–6.567s; three pairs per row, descriptive only.
  Both controllers complete, no retry, leases released. Independent result review
  agrees at scope, with a corrected historical timeout attribution. Default stays
  off; no source changes or release action.
  Parent installed lane had a final review write after its seal; an append-only
  rebind pins that sole changed file and unchanged1,234 other files, preserving
  the old seal rather than pretending it still verifies as-is.
  Next: shared declared typed producer/consumer packed ABI, then real installed
  host-versus-compile_shader exactness. Do not merely remove the31-binding guard.
  `HOST-PACKING-DESIGN.md` and `HOST-REPAIR-NEXT.md` define the next bounded repair,
  scalar/ownership controls and retention-cost measurements. No triangle-only
  exception. Cold identity, C++ provider, native runtime and other obligations
  remain scheduled, not displaced by this integration gap.
- Installed K-layout exactness completed in
  `triangle-k-layout-installed-rev2-20260927/REPORT.md`: 27 pairs per private
  native/pure mode, 108 witnessed public compile_shader calls, 972 raw files,
  486 cross-mode files equal. Exact source, entry, group and pipeline checks;
  nine finite-source-oracle pairs per mode, eighteen reference/no-write only.
  Both children exited zero and the lease is released. Public timing is now above.
  Attempt 1 remains a failed 300s native timeout before final provenance, with
  pure not run. Rev2 fixes instrument-only duplicate provider hashing within
  each sweep, fresh bytes every sweep, independently reviewed and 27 CPU
  controls per mode. No runtime policy change or acceptance relaxation.
  Important integration gap: the public host-metallib fallback does not pack
  this 32/36-leaf triangle ABI; 27 host pairs remain blocked by its 31-binding
  prefix. No direct-Metal substitution or host-route qualification. Keep a
  separate host packed-ABI repair obligation alongside supported-route public
  warm/cold measurement. FABLE-REVIEW.md is the installed receipt handoff.
  Source digest21dbc1a9 and default-off remain; all broader obligations,
  cold identity/C++ provider/runtime/reporter/M1 work stay on the plan.
- K-layout emitter candidate is implemented, default-off, in
  `triangle-k-layout-emitter-20260927/REPORT.md`. Four production files and two
  new test files; exact screened sources reproduced only for the selected half
  triangle family. Dense and sampled neighbors retain the original source.
  CPU selections: 83 focused; 130 passed/74 GPU skips broader; 65 cooperative
  and cache controls. Counts overlap. Caller census: 12 maker comparisons and
  19 route rows/76 lowerings. Independent review found a True/string-True option
  hash collision and mixed-Bias qualification gap; both reproduced and repaired
  with retained red/green controls. Current source/test digest starts 21dbc1a9.
  Sol closes both narrow review findings in REVIEW-CLOSURE.md; 212-file evidence
  seal starts eba019e9 and the source/receipt checkpoint verifier passes.
  No GPU, installed artifact or new performance qualification in this checkpoint.
  Fable subsequently approved this CPU/source checkpoint; the newer installed
  result, public timing and host-route gap are above; no dense broadening/default
  promotion.
  Q staging remains separate/unintegrated; cold identity, C++ provider, native
  runtime, reporter and M1 work remain on the existing plan.
- K-only tile layout screen completed in
  `triangle-k-layout-screen-20260927/REPORT.md`: 27 pairs exact, 25 CPU controls,
  828 total GPU calls including 640 timed samples. Adjacent balanced A/B pairs
  and same-reference null controls: device 1.153746x contiguous / 1.125710x
  runtime-stride, all eight blocks faster in each; contiguous direct wall
  1.067191x/all eight, runtime wall mixed. Null directions mixed, both experiment
  orders positive; unexplained timing bands remain. Independent raw audit passes.
  Retain for emitter/integration validation, not production/default promotion.
  The shared maker serves triangle and dense callers: INTEGRATION-NEXT.md names
  the affected census and installed-route obligations. No new Fable acknowledgment
  task; result return is FABLE-RESULTS.md. Evidence sealed at 900 files,
  SHA-256 prefix bea005db. All broader obligations remain open.
- Q-reuse follow-on screen completed in
  `triangle-q-reuse-screen-20260927/REPORT.md`.27pairs exact O/LSE and allbackings,
  finite fullblock source oracle passes; tails/exceptionals explicitly exact-arm-only.
  Real pipelines fit1024threads and20,736/28,928B.19CPUcontrols;54correctness
  dispatches then386in separate timing run; bothchildren exit0, no retry.
  Contiguousdevice medianpaired1.01243x/all8faster, but wallmixed; runtime-stride
  device/wallmixed. Large contiguousblock5 timing shift retained and unexplained.
  No established direct-call improvement and no integration. Next distinct
  hypothesis is K-only shared-memory transpose without added storage, not further
  Q-staging retries. `FABLE-TRIANGLE-RESULTS-AND-K-LAYOUT.md` prepared; primary
  owns measuring and bounding timing instability. No production/test change,
  all broader obligations and cold/C++ work retained. Sol's independent raw audit
  passes and the primary reproduced it. Evidence sealed at 897 files,
  SHA-256 prefix 6d3dbc84. This is not release qualification.
- Affected source coverage is now explicit in
  `attention-installed-source-census-20260927/REPORT.md`: 64 installed CPU
  lowerings preserve off/on classifications and primary bytes; only the retained
  O-only case receives the cooperative descriptor. Eight adjacent variants still
  refuse in both arms. This does not imply all LSE sources refuse. A separate
  twelve-emission retained triangle discriminator admits fp16 through the
  tiled1024 path with source BN32, half Q/P rounding, LSE and 3-D mask addressing;
  contiguous inner strides do not remove that rounding constraint. The fp32
  contiguous control selects SIMD256. Zero device attempts across four children.
  Next is a bounded Q-reuse hypothesis in this actual tiled source family,
  not an unsupported claim that the O-only speedup solves the reporter's case.
  `FABLE-TRIANGLE-Q-REUSE-HANDOFF.md` is prepared, not yet delivered/accepted.
  Primary owns baseline/controller; Fable may own the shader proof/prototype.
  This is the affected-coverage step already planned, not a reset or removal
  of cold-identity, C++ provider, M1 or external-runtime obligations.
- Installed public attention timing has now completed after Blake's device
  clearance and quiet-host/lock checks. See
  `attention-public-timing-execution-20260927/REPORT.md`: native3.3468x and
  pure3.0591x paired warm speedups for the retained case, all8blocks faster
  in each mode;320 timed samples and80 positive route witnesses. All14children
  exited0 and were reaped, with no retry. Cold first calls remain6.8-6.9s;
  pure cooperative is slower in3/3 observations by27-197ms, native signs mixed.
  No cold improvement, M1/report-resolution or whole-family claim. Default-off
  remains. Sol's independent raw-result review passes;357-file seal3f617ab0
  binds the execution. Fable's results review is received: no finding, bounded
  retention accepted. See `ATTENTION-PUBLIC-TIMING-REVIEW-CLOSURE.md` for the
  consistency-versus-attribution caveat; no exact host remainder is newly measured.
  Standing affected-coverage work remains. Blake's SPI firmware session requires
  a fresh resource check before intensive work, not a blanket hold or substitution
  of light work for the planned implementation/validation.
  This supersedes the Stage G timing hold below, not any workload obligation.
- Integrated attention exactness now has real-route evidence in
  `attention-route-exactness-20260927/REPORT.md`: 16 cases on each of three arms,
  48 witnessed dispatches, exact bits across all arms on this host. The first
  candidate run exposed an opaque-MPS-handle overlap bug before dispatch; the
  repaired check uses buffer identity and relative offsets. Sol reviewed it,
  122 CPU tests pass, and independently reparsed the completed raw GPU receipts.
  Fable's exactness review found no issue at that scope. Default-off remains.
  `attention-installed-preparation-20260927/REPORT.md` now records isolated
  native/pure builds, payload binding and four fresh/relocated guarded CPU
  smokes, zero device attempts. Installed execution now follows in
  `attention-installed-correctness-20260927/REPORT.md`: both modes completed
  sixteen cases on all three positively witnessed routes, 96 calls total,
  exact bytes across both comparisons. Controller reviewed by Sol; raw receipts
  reparsed. No new runtime/test edits or public latency claim.
  Fable's installed review also found no issue at scope. The next controller is
  prepared in `attention-public-timing-20260927/REPORT.md`: uninstrumented warm
  pairs bracketed by witnesses, and separately labeled instrumented cold API
  intervals. Sol's stale-output finding is repaired with per-call reset and a
  no-op negative. Fable reviewed it and found the inconsistent 1.10 estimator
  cutoff. The successor `attention-public-timing-rev3-20260927/REPORT.md` fixes
  that: all eight faster blocks count as a consistent measured improvement,
  magnitude reported. BINDING-003 also binds mode-own truth, per-block uptime
  and buffered cold events. Nine guarded CPU controls pass in each mode,
  zero device attempts; narrow Sol closure review recorded separately.
  That preparation was held until Blake cleared the sharing window. The completed
  execution is recorded above; parent sealed preparation evidence is intact.
- Routing successor: `attention-integration-routing-20260927/REPORT.md` now
  connects the candidate through a default-off compiler option and the sealed
  attention descriptor. Scalar metallib/stash remain unchanged; cooperative
  zero-copy selection uses actual pipeline facts and retained callable identity.
  114 CPU tests pass; two executed red controls catch Sol's repaired stash and
  occupancy findings. GPU/installed/public-call qualification remains owed.
  Fable's CPU review was received and its required dispatch witness implemented.
  The newer result above supersedes the pending worktree numerical check, not
  the installed or latency obligations. No default change.
- September 27 attention integration checkpoint: the reviewed shader A/B stands
  at 8.522x device and 4.090x direct-Metal speedup, not public-call speedup.
  `attention-integration-cpu-20260927/REPORT.md` records the new, unconnected
  builder/selector building blocks, 43 CPU passes and 24 offline compiles.
  Sol found and helped close argument-name collisions. Next is actual
  source/metadata/stash/pipeline binding, then installed per-route numerical,
  alias and latency checks. Fable I1-I11 are explicit obligations, including
  codegen versioning, actual pipeline limits and compile-option parity.
  PyTorch's underlying kernel object exposes resource facts; the wrapper must
  consume them without substituting a host-copy route. No default changed.
- Published 0.3.0 is distinct from this dirty, uncommitted 0.4.0 worktree. No new
  commit, push, default change or publication is authorized by this plan.
- F1 discovery ownership is approved and integrated. Ten new permanent tests;
  installed native 100/100 in two orders; pure 71 pass, 18 skip and 11 unchanged
  native-helper-required failures. Six new pins fail on the old baseline.
- F1 installed artifacts change only the framework-contract Python payload;
  three helper images are byte-identical to the retained baseline. Complete CPU
  evaluator timings are about 3.3 us native-fast, 12.3 us native-strict and
  12.5–13.3 us pure in that run. These are not workflow or launch timings.
- Fable's subsequent C discovery engine is retired, not integrated. Its complete
  evaluation timings lose; its profiler behavior also differs. No new waiver.
- C++ lowering/CFG repairs and source-tree execution evidence exist. Fable has
  now built an experimental separate extension wheel, but clean installed C++
  execution is blocked: the pinned Triton wheel does not export its required
  compiler symbols. The 221-control diagnostic used a developer provider, not
  a reproducible installed pair. Three small host helpers are not this compiler.
- Fable's Metal-up design response is received. Its runtime/package slice can
  use today's MSL metallib and does not depend on fixing the C++ provider.
  The isolated runtime rev3 closed its earlier review findings and passed the
  bounded six-case native GPU smoke. The prepared Python bridge now has CPU
  evidence only; its host comparison controller awaits independent review.
- Eight obligations and 51 workload rows remain open. That does not mean 51
  demonstrated defects. It means current evidence is incomplete at those scopes.

Evidence root throughout:
`/Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/`.
Current accepted experimental source/artifacts:
`discovery-retention-integration-2026-09-26/execution/`.
Do not import retired engine wrappers or its extra probe into this base.

## Transition: same goals, a bounded new implementation route

This is not a reset of the eight obligations, workload comparisons or accepted
repairs. It replaces an open-ended series of host micro-optimizations with a
measured opportunity to share the existing launch implementation across runtimes.
It does not retire Python, pure mode, current MSL or any supported workload.
The native slice starts with one kernel family; it does not earn whole-system
credit or become a universal rewrite prerequisite for 0.4.0.

| Existing obligation | What carries forward | What the new direction changes |
| --- | --- | --- |
| Warm launch latency, native and pure | Existing comparisons, both observation phases, resident attribution and all workload rows | Shared native runtime is one candidate implementation; AOT timings cannot clear Python/pure alerts. |
| Attention O/LSE and device costs | Source-aware numerical tests, layout/rescale/source-block discrimination, actual device timing | Nothing waits on the runtime or exporting compiler. A host-runtime result does not clear a shader issue. |
| Cold identity/compile cost | Full-byte inventory, prior profiles and byte-faithful repair requirement | Separating AOT runtime from compiler is an architectural benefit to test, not a fix for JIT startup or permission to skip inventory. |
| C++ compiler capability and installation | Accepted lowering/CFG controls, default-off route and positive GPU witnesses | Reproducible matched exporting Triton/provider build replaces dependency on the old checkout. |
| External-driver integration | Packing, bounds, lifetimes, assertions, queue ordering and no-replay obligations | A versioned package and native runtime start implementing what the guide currently asks callers to duplicate. |
| Release correctness, distribution and compatibility | Full affected gates, exact installed bytes, pure/native coverage, M1 and other missing evidence | Additional runtime ABI/packaging evidence is required if admitted. No current gate or limitation is waived. |

### Entry checkpoint

Keep F1's source and artifacts as the accepted experimental anchor. Retire the
losing discovery engine and preserve its receipts; do not rerun it to start this
transition. C++ artifact inventory is returned with a blocker, not still queued
as an unexplored task. The resident remainder discriminator is complete:
12 setups/72 correctness checks, 2,052,000 component calls, 73-second lease on
the retained installed diagnostic baseline. Route setup adds about 0.9–1.3 us
to the earlier prefix; matmul preparation is about 3.5 us including diagnostic
overhead; submission create/begin is about 0.08 us. These are inclusive,
non-additive diagnostics, not recoverable savings. Instrument review is still
owed before it supports a repair. The two-fixture package producer now has nine
passing CPU test methods and exact-byte export receipts; the host-staged adapter
and compatibility controller are the next primary implementation.
Fable's rev2 has since returned. The producer successor now matches its four-key
access object and both packages open on the exact image, with eight negative
controls refusing. The GPU request's under-grid case is an oracle mismatch,
not a native BOUNDS refusal; correction is filed before any device run. Full
successor review remains owed. Use `native-runtime-package-slice1-rev2-20260926/`
for composition; the initial producer is retained frozen.
The full rev2 CPU review now reproduces 84/84 controls, six fake Metal completion
cases and 20 mock-client iterations. HOLD remains on three executed failure-path
findings: late-property exception ownership, a defensive capacity-error use-after-
free, and client output reads/reuse after UNCERTAIN. The client timing also
includes its output oracle and needs that interval separated. Narrow successor
request: `astra-native-runtime-rev2-review-20260926/REVIEW.md`. This does not
reopen the accepted reach schema or block primary adapter preparation.
Rev3 has now closed that hold: 52 hashes, 84 controls, six fake wait cases,
red/green late-property and ASan witnesses, and eight client arms rechecked.
First real-Metal smoke passes all six cases (15 actual shader launches and
three pre-submission refusals). Reports: `astra-native-runtime-rev3-review-20260926/`
and `native-runtime-device-smoke-20260926/`. This is not the shared comparison,
an adapter result or performance evidence. No more native repair is requested
at this scope; primary adapter/controller is the next active implementation.
No current performance figure is described as an unavoidable floor.

## Next work and ownership

| Order | Owner | Concrete deliverable | Exit / handoff |
| --- | --- | --- | --- |
| First primary checkpoint, completed | Astra | Resident-launch remainder discriminator on add/softmax/matmul, report under `resident-remainder-discriminator-20260926/`. | Diagnostic scope and residual uncertainty recorded. No fresh representation-tweak series. Library lookup stays retired. |
| Independent implementation, rev3 accepted at scope | Fable returned; Astra reviewed | Runtime/client failure-path repairs and revised oracle/timing controls accepted. Six-case native device smoke passed. | No further repair or acknowledgment requested. Shared comparison, adapter and performance still owed; do not expand runtime families automatically. |
| Independent build, prepared | Fable may delegate fixed procedure; Astra reviews compatibility | Export-enabled pinned Triton plus extension rebuilt with the same recorded inputs. | Fresh/relocated installed CPU controls, no developer provider. If not delegated, queue the heavy build around runtime work; do not imply one person is doing two full implementations simultaneously. |
| Bounded runtime checkpoint completed; no production promotion | Astra authored/executed; Fable ownership review, Sol accounting/receipt review | `native-runtime-host-comparison-20260927/`: 67 CPU controls in six runs; 18 real-Metal correctness rows; 24,000 paired timed calls. Bridge144–151µs versus current111–114µs, zero winning blocks in all four cases. | Reject staged bridge as a latency improvement, retain fixture-only runtime capability. Full framework qualification remains open, no forged metadata; next primary attention discriminator, not another runtime family. |
| Next available safe GPU window | Astra controls lease; Fable reviews affected lane | Run attention O/LSE/source-block discriminator irrespective of runtime readiness. Qualify installed C++ and native runtime only when their CPU reviews are ready. | Separate correctness/timing and retain exact route, shader and binding receipts. Readiness gates apply per lane, not globally; no window waits merely to batch unrelated work. |
| After the bounded runtime slice result, no second runtime family yet | Astra | Return to the measured attention repair; if the discriminator gives no actionable mechanism, take the retained byte-faithful cold-identity investigation. | Native runtime expansion does not automatically take another turn of the roadmap. Preserve small shapes and all numerical outputs. |
| Next Fable slot after runtime handback | Fable | Review the admitted primary repair or take the cold-identity implementation, based on non-overlapping path ownership. | Reuse existing profiles. Do not let additional ABI features or compiler build iterations silently consume this slot. |
| Composition, not after every micro-edit | Astra integrates; non-author reviews changed mechanisms | Fresh installed native/pure/C++ evidence, affected workloads, full project/upstream selection, packaging and compatibility matrix. | Every claim bound to exact bytes; remaining failures and absent hardware explicit. Only then propose release disposition. |

### Implementation specifications and transition decision

Under the evidence root, use:

- `FABLE-NATIVE-RUNTIME-SLICE-ASSIGNMENT-20260926.md`: owned paths, schema,
  invocation isolation, true buffer capacity, retained lifetimes, explicit queue
  staging, error states, independent oracle and CPU/GPU boundary.
- `FABLE-CPP-PROVIDER-DECISION-20260926.md`: matched provider build and installed
  qualification. A separate compiler process is contingency research, not another
  implementation lane started by this plan.
- `METAL-UP-SYSTEMS-DIRECTION-20260926.md`: architectural rationale, not a second
  roadmap or a promise of universal ecosystem support.
- `astra-native-runtime-review-20260926/REVIEW.md`: first implementation review,
  CPU witnesses and narrow reach-schema decision. Fable owns native enforcement;
  Astra owns source-to-descriptor proof and adapter. This local HOLD does not
  postpone attention/cold work or retire the native-runtime direction.

After the first runtime result, choose explicitly:

1. **Retain experimentally** if the defined slice is correct, installable and its
   measured costs meet predeclared comparison criteria. State whether the benefit
   is AOT integration, Python performance, or both; do not conflate them. Keep
   unqualified modes off the new route and return to attention/cold work.
2. **Repair once at the demonstrated cause** if a bounded implementation defect
   explains a miss. Do not expand scope to solve an unrelated ecosystem at once.
3. **Defer this implementation** if it needs a larger lifecycle change, cannot
   preserve required behavior or lacks a credible cost path. Preserve the source
   and receipts; return to the existing implementation plan. The integration
   objective stays open rather than being declared impossible.

Neither a successful one-kernel demo nor a rejected prototype determines the
whole 0.4.0 architecture. A production migration requires a separate reviewed
coverage expansion and measured effects on the existing workload register.
No flag-day driver replacement or requirement to finish a universal platform
before addressing current regressions is introduced here.

## How we keep moving without whack-a-mole

- A lane begins with the affected user outcome, current evidence, one hypothesis,
  one falsifier, owned paths and a stopping condition. No second planning packet
  is needed when those are already in the handoff.
- One primary implementation hypothesis at a time. Independent C++ packaging,
  source review and mechanical verification can run alongside it. Only one writer
  owns a production path. Only one controller owns a GPU window. Heavy builds
  and timing do not overlap; work on CPU-only source review while a window runs.
- After one bounded screen and, if justified, one implementation/correction cycle,
  record retain/reject/blocked and rotate to the next named mechanism. Further
  iterations require new causal evidence, not an unchanged hope. A failed shape
  is retired; the workload obligation is not declared impossible or removed.
- Use the cheap instrument for increment decisions, and full affected workloads
  for user-impact decisions. No arbitrary per-piece percentage bar that prevents
  useful composed savings; no summed nested timings presented as recoverable cost.
- Correctness pins must fail the old defect and pass the repair. Retain observer
  counts, exact exceptions, allocation/lifetime behavior, O/LSE/gradients and
  numerical source semantics as applicable. Loud refusal is not recovered support.
- Review only new code, new instruments or changed claims. A script can recheck
  hashes/counts/arithmetic; no frontier-model turn merely to acknowledge success.
- Do not repeat full warm matrices for a rejected prototype. Do repeat exact
  affected installed qualification after an admitted composition. Reference old
  results with their dates/bytes rather than relabeling them current.
- A new correctness finding blocks its affected integration. It does not freeze
  unrelated safe work. Safety restrictions are not bypassed by model routing.

## Retired and open are different

### Performance entry budget (September 27 correction)

Before a performance implementation, record the same-contract baseline and target,
work removed/retained/added, and a break-even calculation with uncertainty. Use
`T_new = T_base - removed + added`; do not sum overlapping measured components.
Classify the entry as promising, falsified by an optimistic bound, or missing one
decisive measurement. In the last case, build only that cheap discriminator before
the optimization. A capability prototype must be labeled capability work and
cannot quietly become a performance promise. Keep setup amortization, concurrency,
staging and dependency costs in the budget, not just the inner native call.

For device work, operation/traffic counts constrain hypotheses but do not predict
latency without effective issue, cache and occupancy data. Synchronization is not
a GPU timestamp; logical loads are not DRAM bytes. A bound without those terms is
conditional, not a proven ceiling. Do not manufacture parity targets from different
shapes, incorrect/refusing historical arms or different timing boundaries.

Current application: `attention-layout-budget-20260927/BASELINE-PROTOCOL.md`.
Lane-spreading idea held before GPU, with no positive cost model. First measure
the unchanged shader's paired device/wall spans and source-faithful O/padding
checks. No replacement lowering or new optimized shader chosen by that screen.

September 27 successor: baseline completed; cooperative 16-lane design reviewed
in `astra-attention-design-review-20260927/FABLE-RESPONSE.md`. Ten CPU emulation
cases reproduce and barrier/ownership checks support a bounded A/B, not production
integration. Fable owns the isolated shader/controller implementation; Astra owns
its review and GPU window. Sol corrected the cost model: 15.5x is a source-work
proxy, not predicted latency, and redundant scans/barriers are not free. The
25% baseline sensitivity is not a per-piece stop rule. Measure actual paired wall
and device differences, retain exact-bit/alias gates, and report uncertainty.
No new design acknowledgment required. Cold identity and matched-provider work
remain next; no expansion to another runtime family is implied.

The shader-only A/B is now completed in `astra-attention-ab-run-20260927/`:
16 exact cases pass; same-session paired GPU speedup8.522x and direct-wall4.090x,
all eight blocks faster. Sol independently reviewed controller and receipts.
No production or public-call claim. Four harness corrections have red/green
CPU controls; both shaders and all case bytes are unchanged. Next primary work
is integrating the measured BM32/BN64/D64 noncausal schedule under the existing
source proof, with explicit shared-memory and512-thread launch/pipeline metadata,
correct old-route retention, then affected gates and installed native/pure public
comparisons. Fable results review is ready in FABLE-RESULTS.md; no acknowledgment
wait blocks primary implementation. A32-lane rung is not assumed faster or allowed
to erase the16-lane gain. Cold/provider obligations remain scheduled.

Retired: flat-token continuation, token comprehension, bound-get environment
micro-screen, the rejected small copier reorder, and this public-C-API discovery
engine. The copier continuation remains deferred on bounded benefit and semantics,
not presumed impossible. Reopen any of these only with a materially different
mechanism and a new discriminating prediction.

Open: larger shared-launch cost; pure full evaluation; attention and saved-output
fidelity/performance; cold byte inventory; historical model comparisons; C++
installed coverage; MLX/external-driver integration; optional coverage and hardware.
None disappears because we rotate work. Use the ledger, not historical roadmap
claims about hardware ceilings, to retain them.

## Lightweight routing and reporting

- **A: judgment.** Contract/ABI/ownership design, adverse witnesses, unfamiliar
  root causes and final independent review: Astra or Fable. Either may implement;
  neither automatically monopolizes author or reviewer role. Daybreak may take
  appropriately authorized specialist review, not generic receipt bookkeeping.
- **B: bounded execution/implementation.** Opus or Sol where available and chosen
  by the owner, with an explicit spec, tests and stop rule. This is a proposed
  assignment class, not a universal model-quality or price ranking.
- **C: deterministic.** Run the verifier/build/analyzer script. Use a model only
  to adapt a missing procedure or explain a failure, which usually moves to B.

Daybreak Blue is documented as a flagship-model alias for defensive security,
not a cheap mechanical tier. Do not infer price, access or capability from the
name: [official model description](https://developers.openai.com/api/docs/models/gpt-daybreak-blue-latest).
No routing is used to evade a restriction. An unknown runner named `jev` is
unassigned until identified; it does not block the available scripts.

Each handoff gets: owner, tier/reason, source, owned paths, outputs, stop rule,
reviewer and actual state (prepared / accepted / running / returned / closed).
Record one completion line with outcome and measured token/runtime cost if
available; otherwise say unavailable. Transcript length is not a dollar estimate.
Escalate a missed semantic class until its oracle/procedure is repaired, not
permanently every task that resembles it. No arbitrary three-clean-review ritual.

## Done means

An experiment ends in a reproducible retain/reject/blocked result. A lane closes
only at its stated user/capability/evidence scope. The release needs the broader
ledger and Blake's explicit disposition, not just a green host benchmark. Pure,
M1, external-driver or training limitations must remain named, never silently
traded for a native improvement. Missing hardware is not a hardware impossibility.

- 2026-10-02 Astra: accepted Fable default-comparison results review 4ad7aa51 without parity or closure. Implemented ready-input successor in native-import-drained-comparison-instrument-20261002: exactly one pinned framework synchronization after reset and before observers/clocks, with per-row drain receipt and verifier containment/stable-identity checks. Timed launcher/helper, schedule, estimator, ownership and defaults unchanged; ready-input is a changed precondition, not a relabel of the sealed pending-producer result. Queued-work regression executes exact parent (3 fills pending at helper entry) and successor (0), both routes; 98 controls pass both final fresh orders with identical inputs, zero device/child attempts/pending/owners. First screen's missing copy-back-sync count retained with exact source; repaired test only. Full modeled controller/verifier and failure cleanup pass. Audit replay/Ruff clean; parent seals unchanged. Handoff FABLE-HANDOFF.md sha82b433705fce4e627cc4bb06ebe852e5c24d6fa96a09f669317ce230a72b2079; binding0c0784b111db0f016cbc49fc5ee40624299af014b91dc38b668501a8a5fc1d15,1461files,seal67. Independent retained-witness census:64native calls,29/30entries,12getter and18import/getter/release entries per call; potential combined descriptor or lifetime-owned import cache are unimplemented count hypotheses, not speedups. Ready for Fable instrument review, then own identity preflight/new quiet clearance/lease for unchanged1200calls plus1200untimed drains. No GPU/production/default change or watcher. Python warm-call,C++performance/resources and all release obligations remain OPEN and owed. Token and exact preparation resource cost unavailable.

2026-10-28 Codex: accepted Fable fixture-preparation reviewedd6ebe7 and worker-window results review845ff31d (Q1-Q3yes,no defect); prior875-command result/lifetime repair now independently reviewed, formal contrasts remainUNRESOLVED and compatible leads retained. Implemented fa512-rowlocal-native-integration-20261028 (HANDOFFd49d0169,HASHESfda56357,242members/152external pins): actual adapter/worker/controller/archive/offline reduction wired for exactC/S/B/SB/F,23fixtures,230correctness commands,92guarded buffers,offset64,256threads,width32,21888scratch. max_threads>=256 admits supported dispatch and records deviations from1024 as resource findings. Full buffer bytes retained/rechecked; noGPUtiming getters. Final12syntheticjobs/33tests each3environments pass,equalstdout;3sourceaudits pass. Local121file closure reaches actualNativeAPI call boundary beforebody with zero unboundreads. Retained/fixed fake mechanicalSyntaxError and direct-loader unboundcache-read failure; cases now uses snapshot import. No framework/native/ps/compiler/device/GPU/lease run. Binding remains disabled: fresh full provider/bootstrap/final binding,preflight and current reviewed execution authority remain. Carry >=0.32s pre-command idle/unknownGPUstate and~2us position trend into futuretiming contract,not corrections. No production/default change or externalmessage. N1release,RG-02,CP-01,all59OPEN/HOLD.

Next: finish provider inventory and final runtime binding for correctness of the composed five arms. Keep S, B and SB together through qualification; investigate candidate/C differences before timing. Do not treat standalone timing resolution as a reason to discard compatible local improvements or double-count baseline zero-copy/cold-inventory gains. Native schedule cost and late-provider population remain unproved.

2026-10-28 Codex: completed fresh rowlocal paired final bindings and encompassing review packet fa512-rowlocal-final-binding-preparation-20261028 (HANDOFFd3e9350f,HASHES ece60f56,283sealedmembers). GPU0e4291ec/CPU80ff2166 share full61629file/2,017,253,924byte/7487link inventory,417additions/no changed or removed priorfiles.169copy relations/163identical:GPU runtime only gate enable plus preflight arm-list update from immediate parents;4tests adapt final entry. All27providerPythonfiles unchanged;20-provider-name checks conservatively cover16currently used names plus4unused numpy checks. Actual final manifest/Snapshots/serve reaches pre-NativeAPI boundary,all92fixturefiles read bound;unapproved CPU/GPU actual entries refuse before output/spend/native.45qualificationjobs/186tests each system3.9+native/pure3.14 pass,equalstdout;prooflabels samebinary.3fullbyte verifiers pass. Final readiness flags enabled for future authenticated entry only,authority_supplied=false;no decision/receipt/clearance/framework/ctypes/ps/device/compiler/GPU invocation. Private spent root0700empty,CPUattemptabsent,shared lease stat-only/unacquired. Correctness-onlyC/S/B/SB/F,230commands/23fixtures,source/guard/oracle contract unchanged;92pin native cadence still unmeasured. Encompassing Fable request includes actual integration d49d0169/fda56357,not only activation lines;ready through Blake,not sent. Next current reviewed CPU provider stage then separate GPU decision/receipt/clearance within existing freshness and lease rules. No production/default/timing promotion;compatible-gains queue retained;N1release,RG-02,CP-01,all59OPEN/HOLD.

2026-10-28 Codex: accepted Opus encompassing integration/final-binding review f8584a3a (Q1-Q3 yes,no defect; reviewer is Opus,not Fable). Under Blake explicit Approve CPU provider preflight, CPU80ff2166 completed once in56.065s including entry work,354samples,974487552byte peak,0.0991s maximum sampling gap;both children exit0,final owned census empty,no cleanup errors. Decision recorded beforelaunch;source-bound offline report names GPU0e4291ec,controller2/worker31records,zero device/public calls. Raw fa512-rowlocal-provider-final-attempt001-20261028 retained;recount in fa512-rowlocal-final-cpu-results-20261028. Separate GPU decision pending;freshness deadline16:18:46UTC,receipt/clearance not issued,lease unacquired,correctness230commands still unrun. No timing/release/promotion credit;compatible S/B/SB leads preserved;N1release,RG-02,CP-01,all59OPEN/HOLD.

2026-10-28 Codex: completed approved rowlocal native correctness run under GPU0e4291ec/CPU80ff2166 and encompassing Opus reviewf8584a3a. Separate CPU and GPU replies recorded beforelaunch. Receipt/clearance issued together285.144s afterCPU completion;controllerentry0.03698s later. One spent marker,5freshC/S/B/SB/F builds,230commands/23fixtures/2poisons ALLPASS;candidate/C byte differences0,allarms poisoninvariant,guard/inputbytes intact,exceptional/zero/subnormal fixtures pass. Original maxabs0.0001397639049<0.004. Runtime171.692s,1922samples,976453632byte peak,0.126s maxgap;workerexit0,noerrors,finalownedcensusempty,lease released. Allpipelines1024maxthreads/32width/21888scratch;dispatch256. Frozen272modules/554images stableatclose;CPUdiff16boundharness additions/11CPU-only removals/2changedrecords+2expectedOSdriverimages explained. Threeofflineanalyses byteequal,system3.9 independent;prooflabels samebinary. Full61629file/7487link originalbinding integrity passed. Results fa512-rowlocal-native-final-results-20261028 HANDOFF72d2f8ae,HASHES082c4a53 pins635inputs. Review request ready,not sent;native provenance rests on authority/workerchain,not rowlabels. Timing preparation retains C/S/B/SB factorial and Fcontrol,conditional/composed contrasts and savinginteraction S+B-C-SB;no standalonelead discarded for unresolved timing. NoGPUtiming measured/authorized,no production/default or performance/promotion/release credit. Next independent results review then concrete balanced/null-controlled timing instrument;carrypriorcadence/order uncertainty. N1release,RG-02,CP-01,all59OPEN/HOLD.

2026-10-28 Codex: accepted Opus rowlocal results review08c3da60 (Q1-Q3yes,nodefect), confirming S/B/SB/F byte-identical to C on23fixtures,bothpoisons. Built CPU timing instrument preparation fa512-rowlocal-timing-instrument-preparation-20261028 HANDOFF2509f8aa,HASHES55c79080 (191sealedfiles/138pins). Actual adapter/worker/controller/archive/reducer wired for430isolatedcommands,30exactlyposition-balanced Williams blocks,2replicas,per-command poison+acceptedCbyteoracle,separate7contrast/null floors including savinginteraction S+B-C-SB. Five shaders/allfixtures unchanged;timestampgetters/decoder exactASTcopies fromacceptedworkerwindow;lease/monitor/cleanupcaps retained;NATIVE_READY=False,noexecutionbinding.21safejobs/45testsperenvironment pass(system3.9,native/puresame3.14binary),stdoutbyteequal;3sealverifierspass. Localclosure missing23.f64refs caught beforeNativeAPIbody,retained/fixed table only. Retained-noise sensitivity screen FAILS:18.6-31.7us floors exceed old different-shader reference scales S8.063us/B1.437us/SB9.917us;combined22.25us,interaction31.67us. Not anewprediction/bound orfailureofS/B/SB;candidateHELD_FOR_RESOLUTION,notdeviceapprovalready. Budgetscenario321s/420worker,401s at1.25x,notbound. SeparateunimplementedAMPLIFICATION-DESIGN explores independentguardedoutputs/Kinstances,additiveversusmultiplicativenoise;doesnotwaivebatchingNO-GOortransferisolated/endtoendclaims. ReviewrequestQ1-Q3ready,not sent. No framework/compiler/ps/native/GPU/lease action,newtimingorproduction/defaultchange. Keepcorrectcomposedleads;next resolve measurement regime before final provider/binding. N1release,RG-02,CP-01,all59OPEN/HOLD.

2026-10-28 Codex: accepted Opus timing-preparation review bdeea790; corrected the old HOLD explanation (max-single-null floor mismatched the median statistic, not proof gains are unresolvable). Built fa512-rowlocal-timing-instrument-rev2-preparation-20261028, HANDOFF4319b209, HASHEScf00befd,261sealedfiles/332pins. D1 saves completed buffers and incremental raw timestamps before validation, tested through actual adapter/archive with retained-parent red control. Sixth exact-C source C0 separately compiled with distinct build identities. 504single-dispatch commands,13six-row Williams cycles,all seven standalone/conditional/composed/interaction/F contrasts position-balanced within each cycle;matched C-C0 null rotates throughout. Median-cycle order-statistic intervals use conditional common-median/independence assumptions,not unconditional performance credit;350ms minimum host wait-to-commit cadence,not GPU idle.24final CPUjobs/51tests perenvironment plus redprobe pass across system3.9/native/puresame3.14binary,identicalstdout;3sealverifierspass. Retained stopped firstwave audit-name failure and passing secondwave;finalwave003 includes six-arm dormant preflight/audit. Original five shaders/all23fixtures/lease/monitor/cleanup unchanged. Budgetscenario376.229s vs420worker leaves43.771s,not bound;sixth build/cadence/runtime unpriced. Amplification deferred. NATIVE_READY=False,noBINDING/authority/ps/framework/compiler/native/GPU/lease action or production/defaultchange. Next Opus review Q1-D1/ownership,Q2 estimator/null/cadence,Q3 budget before fresh provider/finalbinding. Compatible S/B/SB leads retained;all59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

Next: review the concrete median-matched six-arm timing preparation, then bind provider/bootstrap/runtime only after the statistical and budget questions are settled. No new timing approval is requested by this CPU packet. Preserve all individual and composed candidates; unresolved standalone B is not grounds to discard B within SB. Any measured local gain still needs the public-backend and end-to-end path, per-workload regressions and startup/resource accounting.

2026-10-28 Codex: accepted Opus timing-rev2 review fe9ad5f2 (Q1-Q3 yes; N1 switch labeling). Prepared fresh paired timing bindings in fa512-rowlocal-timing-final-binding-preparation-20261028: HANDOFF db53e5ad, HASHES b6e5456a,353 sealed files. GPU c8938452 / CPU 01bea0c3. N1 reports451 switches and17 repeats across468comparison commands; exact504-command schedule and all numerical contrasts unchanged. Runtime delta from reviewed rev2: metadata helper/fields,adapter model pin,authority activation only;3CPU entry tests adapt final gates/manifest and3new controls discriminate metadata/statistic preservation. C0 means distinct API objects,not separate compiled code. Full fresh shared inventory62276files/2037132186bytes/7487links:647additions,no changed/removed priorfiles. Provider27Pythonfiles unchanged,20attributes cover16used+4conservative.54finalCPUjobs/207tests each system3.9 and native/pure(shared3.14binary) pass with equalstdout;3fullbyte verifiers pass. Actual final-manifest closure reaches pre-native boundary;missing approval refuses before spend/native/lease. Retained firstwave audit omitted3intentional test edits;fixed exact audit/rebuilt unissued bindings,no runtime change. Six qualified shader bytes/fixtures/oracle/estimator/cadence/caps unchanged. Native flags enabled for future authenticated entry only;authority_supplied=false,CPUdraft unapproved,spent root private/empty,results absent,lease stat-only/unacquired. No provider/framework/native/ps/compiler/GPU work or performance credit. Ready for final binding review through Blake,not sent;after review fresh CPU decision/preflight then separate timing GPU go within freshness/lease rules. Keep standalone and combined S/B/SB leads;N1release,RG-02,CP-01,all59OPEN/HOLD.

Next: final paired-binding review of the exact six-arm timing instrument, then current CPU provider approval/preflight and separate GPU timing approval. Do not discard B because standalone resolution may be weak; retain B_given_S, S_given_B, SB and interaction alongside individual changes. Isolated device gains remain leads until public-backend and end-to-end tests qualify the combined path.

2026-10-28 Codex: honored standing GPU approval and fresh-preflight request without another approval pause; repaired/retried under explicit prior troubleshooting instruction. Original GPUc8938452 spent/UNAVAILABLE at shared build object after6builds/0commands. Codex-qualified c58cfd69 permits only identical-source C/C0 same-kind library/function sharing, retaining distinct options/pipelines/all other checks; actual READY shows function-only sharing. Its504-command run failed closing because model lazily imported contract; raw sequence retained and offline-correct, not relabelled complete. Eager-import successor3164dba4 (model import relocation plus adapter pin;54jobs/209tests per environment+cold red/green) completed504commands in342.452s,3841samples,978501632byte peak,normal workerexit0/cleanupcomplete/lease released. CPU9fc62cbc passed52.105s; authority issued from fresh118aaaed preflight under Codex assessment95c134c8. Final frozen279modules/557images equal closing; full command correctness and independent cycle reduction match under system/native interpreters. Median cycle means(us):C239.104,C0237.715,B237.736,S685.694,SB689.708,F693.118. Matched null passes;B_at_C/B_given_S/interaction unresolved;S/SB/F contrasts conditional-negative. Earlier failed-close sequence has same outcome labels, retained separately/no pooling. Rising S/SB/F cycle costs remain unexplained, conditional median assumptions unproved. Keep B and composed leads; investigate common materialization/lowering costs, do not claim end-to-end gain. Results/review packet fa512-rowlocal-timing-close-repair-results-20261028 HANDOFF1b7f9a91/HASHESbd8c0d55,1759retainedfiles/27roots plus62645input pins/7487links. Two runtime repairs are Codex-qualified, not independently Opus/Fable-reviewed; external review pending before evidence-ready promotion. Original attempts and packaging sort-only failure retained. No production/default change;all59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

Next: independent consideration of both Codex-qualified repairs and retained timing results. CPU source/resource inquiry should explain why S/SB/F are much slower and progress across cycles while C/B remain steadier. Preserve B as unresolved both standalone and conditional; explore combinations that address the shared lowering/materialization bottleneck without assuming its cause. No evidence-ready row or end-to-end speedup is claimed. Standing approval persists; do not ask again merely because a preflight window expires. A new material experiment remains separately scoped.

2026-10-28 Codex: received Opus timing-close/repair results review f0499d2e (no code/results defect). Added immutable-sidecar correction: successor GPU runs used standing repair/continuation instruction as interpreted by Codex, not new per-successor replies; parent fresh-review/decision discrepancy remains explicit, no retroactive authority or evidence-ready promotion. Median 690us/239us is not a stable ratio: S/SB/F rise ~0.48-0.51ms to ~0.72-0.76ms, adverse all13cycles in both retained sequences; B remains unresolved/retained. Completed CPU-only fa512-fragment-lowering-inquiry-20261028 HANDOFF5855baff/HASHESe1498912: exact expansion of all four-row loops in C/B/S/SB/F, original controls beside each. Ten offline shader builds plus ten AIR decodes pass. All five fresh baselines match retained AIR except lane paths; expansion removes float accumulator alloca and11/11 ordinary64float vector loads/stores in every arm, retains two half-matrix arrays,21888scratch bytes and16MMA sites. Static H1 criterion met, not native spill/register/occupancy or speed evidence; common change in C/B does not explain S/SB/F-specific cost or temporal progression. Nine tests per environment/9jobs pass under system and both shared-binary proof labels; aggregate replay byte-identical.688sealedfiles/24externalpins; parent1759retainedfiles/full62645input manifest/7487symlinks verify unchanged. Source-generation path typo retained, no compiler launched before correction. No framework/ps/device/GPU/lease or production change. Next independent CPU transformation review then all23fixture native correctness with original/expanded controls before any timing; retain B_U/SB_U and public-backend/end-to-end obligations. All59,N1release,RG-02,CP-01 OPEN/HOLD.

Next: source-bound review of the exact ten-source CPU lowering packet. Build correctness coverage before timing; test original/expanded C and B as controls for general index lowering, then S/SB/F and their expansions. Keep within-run progression as a separate falsifiable runtime question. Do not equate removal of AIR allocas with physical spill removal or a speedup; do not discard unresolved B or assume the original adverse S cost is intrinsic. No fresh GPU run or additional approval is requested by this CPU step.

2026-10-28 Codex: accepted Fable fragment-lowering review ee82b6bd (Q1-Q3 yes,no defect) and completed paired native correctness instrument preparation, fa512-fragment-native-preparation-20261028 HANDOFF4c07b35d/HASHESdd258377,359sealedmembers. GPUbeb1974a/CPUe1da3485 bind10exact original/expanded C/C_U/S/S_U/B/B_U/SB/SB_U/F/F_U sources,23unchangedfixtures,2poisons,460commands,92guardedbuffers,offset64,256threads,width32,21888scratch. Added per-original/expanded46comparisons each and retained differing case/poison IDs before timing; independent oracles/tolerances unchanged. N2 uses already native-qualified max_threads>=256 rule unchanged in live/model/offline;1023 survives full synthetic460row run as resource finding,255/bool/float refuse. No new1024 stop. 13GPUfile deltas include3adaptedtests;27providerPythonfiles unchanged; bootstrap/native transport/authority/process/cleanup/caps remain inherited. Full shared inventory61394files/2011914057bytes/7487links;182additions,no changed/removed versus named worker-window ancestor. Final45CPUjobs/192tests each system3.9+shared-binary native/pure3.14 pass,identicalstdout; actualmanifest closure reaches pre-NativeAPI body withzero unboundreads. Fullbyte/retainedseal verificationPASS. Retained test helper typo,stale232response assertion and missing unchangedproviderprofile packaging failures plus both unissued inventory generations; resolved before finalwave,no native attempt. CPUdraftfalse,attemptabsent,spentrootempty,leasestat-only; no compiler/framework/native/ps/GPU execution or authority issued. Ready for Fable encompassing instrument review,not sent. Later timing requires predeclared in-attempt reset/no-reset control and per-command/per-arm progression,not another pooled median; B/SB composition and public-backend/end-to-end work retained. No retroactive authority ratification or performance/release credit;N1release,RG-02,CP-01,all59OPEN/HOLD.

Next: Fable review of the complete fragment native correctness instrument (4c07b35d/dd258377), then current CPU provider preflight and source-bound GPU authority for460correctness commands under the reviewed freshness/lease rules. Source expansion review is complete; do not repeat it as another standalone design gate. No GPU timing until all23fixtures pass and any paired byte differences are explained. Keep the unchanged>=256 capability rule and all resource findings. Develop H2 reset-control semantics for a later timing design without pretending a new pipeline resets device state.

2026-10-28 Codex: accepted Fable instrument review87ac78dc Q1/Q2yes,Q3blocked D1. Parent GPUbeb1974a/CPUe1da3485 kept sealed/unspent,superseded; no CPU decision issued. Repaired exactlyone runtime line decision.ARMS five->ten in fa512-fragment-native-rev2-candidate-20261028; allotherGPU/provider/shader/fixture/cap/cleanup bytes unchanged. Added actual preflight.reduce -> window_authority.verify regression against successor header/bound snapshots with deleted expired syntheticCPU/authority records, no controller/worker/Lease. Retained realparent gate rejects same report at preflight arm population; successorpasses; malformedarms/freshness/links/unapproved records stillrefuse; realclock rejects testauthority. Audit checks cases/decision/preflight equality. GPUheader cosmeticFive->Ten-arm only beyond freshpaths/inventory. SuccessorGPU7110a619/CPU32b85e17; full61574files/2017352504bytes/7487links includes all61394parentinputs unchanged plus180additions. Final48CPUjobs/201tests each system3.9+native/pure shared3.14binary pass,all16stdoutidentical; bothparent andsuccessorfullverifiersPASS. Freshspentrootempty,CPUattemptabsent,draftfalse,lease stat-only; no nativepreflight/GPU/authority. Delta packet fa512-fragment-native-rev2-preparation-20261028 HANDOFF3d11e9d2/HASHES0eb831f1,292sealedmembers,ready forFable shortdelta review,not sent.460commands/23fixtures/2poisons unchanged; >=256threadcapacity resourcepolicy unchanged; laterpairedtiming reset/no-reset/percommandprogression obligations retained. All59,N1release,RG-02,CP-01 OPEN/HOLD.

Next: Fable shortdelta review of one-line D1 repair and real reduced-report authority regression. Only successor pair7110a619/32b85e17 may proceed after review/current decisions; blockedparent e1da3485 must not be used for CPU. Stage actual decisions and clearance coordination fields before native preflight to preserve300s freshness. No changes to shaders/caps or additional source-design review.

2026-10-28 Codex: Fable rev2 review46d07a11 accepted; Blake explicitly approved CPU32b85e17 and conditional GPU7110a619 together, including ps. Recorded decision before CPU; no repeat approval pause. CPU completed68.292805s/405samples/1002651648byte peak; GPUreceipt+clearance issued15.050414s after CPUcompletion and controllerentered0.060314s later. One reviewed10-build/460-command correctness attempt completed377.371971s,3632samples,1012695040byte peak,workerexit0,cleanupcomplete/noerrors,finalownedcensusempty,leasereleased. All23fixtures/twopoisons accepted;230/230original-expanded pairs byte-identical,allpipelinesmax_threads1024. No retry/runtime repair. Three offline retained-buffer reductions match exactly,system3.9 independent/native+pure shared3.14. Full61574input/7487link/292instrumentseal+1005retainedinput resultverifierPASS. Provider267modules/552images->GPU272/554 explained:16harness additions,11provider removals,2entry/pathchanges,2AGX/IOGPUOS-buildimages; closingpopulationidentical. Results fa512-fragment-native-rev2-results-20261028 HANDOFF7c0e6a9f/HASHES290bbc6a,ready forFable Q1-Q3 resultsreview,not sent. Keepallfivepairedleads includingB_U/SB_U. Next coherent pairedtiming design with predeclared in-attempt reset/no-reset and per-command/per-arm progression; a pipeline rebuild is not aGPUdriver reset. Actualelapsed exceeds317s projection; account actualcost before timingschedule. No timing/public-backend/speedup/promotion/releasecredit or productionchanges;all59obligations,N1release,RG-02,CP-01OPEN/HOLD.

Next: Fable results review of completed rev2 correctness (7c0e6a9f/290bbc6a), then one paired timing design carrying original/expanded C,S,B,SB,F and explicit reset/no-reset controls. Preserve per-command progression and arm-specific nonstationarity; do not use a pooled median as a stable speed ratio or discard a composable lead. The native correctness prerequisite is now fulfilled on these exact sources; do not repeat the unchanged transformation review. Timing remains separately bound and reviewed.

2026-10-28 Codex: accepted Fable results review d258a64b (Q1-Q3 yes, no defect): exact ten shaders native-correct on23fixtures, all230expanded/original pairs byte-identical; no speed credit. Prepared CPU-only fa512-fragment-paired-timing-design-20261028, HANDOFF72d2c574/HASHESa792fc07,53sealedfiles/58pins. Proposed12arms including C0/C_U0 nulls, two sequential workers, four72-command comparison epochs,360total commands/24builds; restart between middle epochs and no-reset holds on either side. Allthree host gaps target90s; >100s retains data but labels H2GAP_UNMATCHED. Startup dispatches differ at restart; no isolated process-causality or GPU/driver reset claim. Three reverse-pair cycles per epoch cancel linear position effects only; quadratic counterexample retained.19paired/composition contrasts and2matched nulls per epoch, full per-command progression, descriptive-only labels/no confidence intervals. Measured protocol budget replaces317s projection:289.54s/worker,709.08s controller work; 1.25x sensitivity361.93/886.36s. Worker420s/4GiB unchanged; future controller900s/authority1200s proposal explicitly needs reviewed binding/decision, not applied now. Final9jobs/22tests per environment pass identically across system3.9/shared-binary native+pure3.14. Retained first-wave tiny floating-budget discrepancy; fixed by integer-nanosecond accounting. Full prior61574input/7487link verifier passes unchanged. No native/process/ps/compiler/GPU work or authority this step. Next integrate two-generation ownership, monitored pause/first-commit enforcement, exact-output/timestamp/population bindings and real authority regression; then one encompassing Fable instrument review, not another unchanged shader review. Retain B_U/SB_U, interactions and public-backend/end-to-end objectives; all59obligations,N1release,RG-02,CP-01 OPEN/HOLD. Nothing sent externally.

Next: turn the sealed CPU timing design into the actual instrument. Test the continuous lease across two strictly sequential worker identities, intermediate closing/reap, no-reset resource continuity, monitored 90-second pause and first-commit target; bind every sample to its worker generation and source. Reuse exact accepted outputs for timed commands and retain raw timestamp failures. Extend the actual provider-reducer/authority regression to12arms and2workers before a fresh binding. Size the final code against measured costs and declared new whole-attempt limits. Request one encompassing Fable review once this integration is concrete; no native launch is authorized by the CPU design.

2026-10-28 Codex: completed paired fragment timing instrument, fa512-fragment-paired-timing-preparation-20261028 HANDOFF07cb01d8/HASHESc825ec5c (463sealedfiles). GPUd406dee9/CPU4b4df46c bind12arms/24builds/360commands in2sequential180-command workers underonecontinuouslease. S/B/SB and expanded interactions retained,19descriptive contrasts/2matched nulls per epoch,4epochs with full per-command progression. Actualcommit enforces90s no-reset/restart floors; >100s withholds H2 as GAP_UNMATCHED. Process restart includes unmatched startup work,not causal GPU/driver reset. Oldworker close/exit0/emptycensus/kernelidentity retirement required before newworker; failure never launchessecond. Exactoutputs,256threads,420sworker/4GiB unchanged; proposed900scontroller/1200sauthority explicitly pending review/decision. Real preflight-reducer->authority regression passes12arms and rejects retained10-arm constant,oldcounts/caps/staleness; actual snapshot closure stops before NativeAPI body withzero unboundreads. Full63256files/2127641860bytes/7487links retains63007parentunion unchanged plus249additions. Final48CPUjobs/229tests per environment pass withidenticalstdout under system3.9 andshared-binary native/pure3.14; fullverifierPASS allthree. Packaging/wrapper/fixture/seal-order preparation failures and unissued drafts retained,notcredited; no runtime repair or native attempt. No framework/provider/ps/compiler/GPU execution orauthority; lockstat-only,spentrootempty,attempt/decisionabsent. Ready forone encompassing Fable review,not sent; do not repeat unchanged shader correctness review. No production change/performance/releasecredit; all59,N1release,RG-02,CP-01 OPEN/HOLD.

Next: Fable encompassing review of paired timing instrument (07cb01d8/c825ec5c; GPUd406dee9/CPU4b4df46c), then explicitly scoped CPU/GPU decisions and fresh preflight under this binding. A single scoped approval may cover both stages without another pause after CPU success. Retain process-reset confound, matched-gap rule, null resolution, B_U/SB_U interactions and public-backend/end-to-end obligations. No further unmodified design/correctness review.

2026-10-29 Codex: accepted Fable paired instrument review af5df683 and recorded Blake's approval before CPU launch, covering CPU4b4df46c and conditional GPUd406dee9 with ps, full1200s window and awake condition. CPU completed69.338s; receipt/clearance issued8.343s after completion, controllerentry14.537s later without another approval pause. One GPU attempt completed24builds/360commands in553.467s; 5439samples,1023524864byte sampledpeak,.127480s maxgap; both workers exit0/reaped/kernelidentityretired,finalownedcensusempty,lease released,no repair/retry. All360outputs byte-identical to timing fixture oracle;24pipelines max_threads1024. Actual boundary gaps90.145086/90.148020/90.146815s match; restart proven by generationstate/READY/CLOSE/protocol/root-retired,not constants alone. CPU267modules/552images->bothGPU280/557 explained:19GPUharness+statistics/fractions/decimal/_statistics/_decimal,11providerremovals,2path/entrychanges;3fileimages mathchain+2OS-buildGPUimages; closingpopulationsunchanged. Three-environment offline analysis/population replays byte-identical; final full retention/inventory verifiers allPASS withidenticalstdout/emptystderr (results-verification001-20261029). Raw per-epochmeans retain strong expansion lead:S479-486us->228-233us,SB474-482->226-229,F465-483->214-221;C~232-235us. Epochs0/1/3 fail one matchednull each:all19contrasts NULL_CHECK_FAILED there. Epoch2 has4descriptive savings(unrollS/SB/F,F_at_C_expanded18.847us),4costs,11unresolved; restartcomparison NULL_CHECK_FAILED despite matchedgaps. No changed acceptance/no pooling/no causalresetclaim. Shorter12arm exposure differs from earlier rising6armrun; cause remainsopen. RetainB_U/SB_U/interactions and backend/end-to-end objective. Results fa512-fragment-paired-timing-results-20261029 HANDOFF5a1d7d4b/HASHES366944b7,35sealedfiles/452retainedpins/5populations plus original63256files/7487links; FableQ1-Q3results handoff ready locally,not sent. Plot-tool missingdependency/cachewarning and layoutrepair retained; no measurementrerun. No production/default/publicpath/promotion/releasecredit;all59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

Next: Fable results review of 5a1d7d4b/366944b7. Then CPU source/lowering explanation of the selective S/SB/F benefit and expanded F lead; design the next combined-change/null-ordering experiment before any further GPU run. Preserve all current labels, B_U/SB_U compositions, shorter-exposure confound and the public-backend/end-to-end obligations. The approved run is finished; do not repeat it or ask for its approval again.

2026-10-29 Codex: accepted Fable paired timing results review 918afd8e (Q1-Q3 yes, no defect; C1 population condition met). Completed CPU-only source/lowering account fa512-fragment-lowering-account-20261029, HANDOFF 4d50a0e0 / HASHES b3061bd5, ready for Fable, not sent. Exact ten shader hashes join both native build generations; old expansion spans reconstruct all ten sources. New line-level AIR checks distinguish S/SB/F's two indexed accumulator-load/product/same-pointer-store paths from C/B scratch rescale; expanded S/SB/F's eight products flow from SSA accumulators/coefficient loads through SSA into MMA, with no ordinary float-vector memory access. All-arm accumulator removal alone cannot explain selective timing; physical spills/registers/occupancy and runtime/offline compiler equivalence remain unproved. Source geometry enumeration counts 2048 coefficient/scratch writes per full block for S/SB versus F's shared256, with different SIMD/group publication scope; counts are not hardware transactions or time attribution. C/B rescale has32matrixstores/32loads versus S/SB/F0stores/32loads. Keep all compositions including B_U/SB_U and possible future F+B (not generated/qualified). Initialization/rescale/epilogue loops expanded together, so exact causal loop unresolved. Pinned current backend template guard/fallback and inherited composition map preserve public-path/end-to-end goals. Fair independent sign model reproduces 9/16 and14749/65536; independence is an explicit assumption, not implied by unbiasedness. More cycles alone do not establish null equivalence; next design must fix practical resolution, uncertainty/exposure/outlier rules and bounded observer policy before fresh data. Existing three failed-null epochs and unqualified restart remain unchanged. Account/check/verify:9jobs across system3.9 and shared-binary proof3.14 labels allpass/identicalstdout;17positive checks+12adverse controls per environment;25sealedfiles/43inputpins/6lanes1654files unchanged. Report-writer path-key collision retained and fixed without changing analyzer/inputs. No compiler/framework/native/process/ps/lease/GPU action, product source change or promotion. All59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

Next: Fable review of the CPU source/lowering account (4d50a0e0/b3061bd5), then one concrete composition and measurement design. Keep the large S/SB/F raw lead, smaller expanded F lead, unresolved B/SB interactions, all-loop expansion confound and changed-exposure limitation. Establish new null precision/cycle/observer choices before another timing run; do not relabel the accepted result or drop useful components because standalone effects are unresolved. No repeat review of unchanged native correctness is needed.

2026-10-29 Codex: accepted Fable source/lowering review 353a70fa (Q1-Q3 yes, no defect). Retained parent; sidecar corrects S raw lead to248.1-254.5us, names lid=32*sgitg+lane assumption, and distinguishes Fable's uncomputed engineering digest from Codex's reproduced current checker result. Prepared concrete CPU-only fa512-fragment-resolution-design-20261029 HANDOFF d5460fe5 / HASHES 42c41ff9, ready for Fable, not sent. Same12sources/19contrasts/two identical-source nulls; two sequential fresh workers,12forward/reverse rotated cycles each,648commands/24builds. Every arm covers all24absolute positions once per worker; no90s holds or causal reset comparison. Primary F_at_C_expanded; B/SB interactions retained without standalone-gain prerequisite. Prospective5us pairwise engineering resolution, cycle/block variation penalty, half-run stability and both-worker agreement replace sign-only crossing; all labels descriptive, no confidence/power/family-wise guarantee and no old data reclassification. Constant small null can qualify; wide crossing/drift/spikes withhold; explicit candidate-specific-bias counterexample remains indistinguishable from a real gain. Observer50ms/cadence/350msfloor, safety/byte/oracle/ownership and420/900/1200caps unchanged. Measured protocol maxima0.789/0.813s; projection325.4sworker/690.8scontroller,1.25x406.75/863.5s leaves13.25/36.5s margin, not a bound. More exposure and memory unqualified. Final check/budget/verify9jobs across system3.9 andshared proof3.14 allpass/identicalstdout:68checks,8refusals,16synthetic scenarios;35sealedfiles/21pins/5lanes635files unchanged. Initial budget mean floating-sum cross-interpreter difference retained; integer-nanosecond reporting repair only. No compiler/framework/native/process/ps/GPU/lease or product change. Future instrument must exercise real648-command authority and old360-command red control before execution review/decisions. All59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

Next: Fable design review of d5460fe5/42c41ff9, then build one concrete instrument with real authority/preflight/count tests and full lifecycle checks. Do not execute this CPU model as an instrument or reuse the finished timing approval. Retain original/expanded compositions, prior failed-null labels, observer/exposure limitations, future F+B possibility and the public-backend/end-to-end objective.

2026-10-29 Codex: accepted Fable resolution-design review57a2b915 and built the complete resolution instrument, not another design round. fa512-fragment-resolution-preparation-20261029 HANDOFFcbf4c086/HASHES6d0dc965; GPU949b7c60/CPU5e2767dd. Exact reviewed reducer/schedule,648commands/24builds/two324-command workers; all12sources/19contrasts retained, no epoch holds or causal reset comparison. Reserved anchor authenticated zero, unchanged350ms native commit floor,50ms observer/select,256threads,420/900/60/1200/240caps and4GiB. Real preflight reducer->real648 authority->actualcontroller/sequence with fake transport/lease; actual360header and count-specific controls refuse. Native transport/observer/protection bytes identical; real methods tested on fake clocks/census. Full63491files/2187569319bytes/7487links retainsall63256parentinputs unchanged.177sourcepairs:161identical/16declaredchanges. Final51jobs across system3.9/shared-proof3.14 pass/identicalstdout:218unittests per environment plus68designchecks/8refusals/16scenarios. Final437file seal/fullinventory verification all3PASS/emptystderr. Test-fixture/sys.path/ctypes-initializer and seal-order preparation failures retained; ctypes.dlopen was blocked but preceding_ctypes extension import is disclosed, not claimed zero native loads for that failed wave. No framework/provider/device/experiment/ps/compiler/GPU/lease execution, no authority; newattempt/decision/spentrootsabsent,5parentlanes546files andlockmetadata unchanged. Thirteen review notes folded in: halfclauses implied, predecessor/nullcopy carryover imbalance, poison-position interaction, means-only spikes,250us synthetic scale,31s floor request and narrowbudget. No old reclassification or credit; allcompositions/end-to-end goals retained. Ready forone combined Fable readiness review, locally notsent. All59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

Next: Fable one combined execution-readiness review of cbf4c086/6d0dc965 (GPU949b7c60,CPU5e2767dd). No additional design round. After acceptance, scoped CPU then conditional GPU decisions may be covered by one explicit reply; continue after fresh CPU success without a redundant pause. No execution yet. Retain carryover/bias limitations, existing labels, S/B/SB/F combinations and future public-backend/end-to-end validation.

2026-10-29 Codex: accepted Fable combined readiness review8297f6e9 (Q1-Q3yes,no defect) for resolutionGPU949b7c60/CPU5e2767dd. Documentation-only acknowledgment and unapproved decision proposal in fa512-fragment-resolution-readiness-acknowledgment-20261029; reviewed packet and bindings unchanged. Recorded recipe omissions, fake lease/peer boundaries, exact count-red-control scope, external passing verification and procedural1200s/empty-spent/review-name/outer-wait conditions. No more design review needed. Next is one explicitly scoped CPU preflight plus conditional GPU decision, then continue without redundant pause after success, subject to capacity/coordination and existing lease. No execution/authority issued, no new data or credit. All59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

2026-10-29 Codex: executed the explicitly approved CPU5e2767dd then conditional GPU949b7c60 under Fable8297f6e9 without another approval pause. CPU complete68.208s; receipt/clearance33.688s after completion, entry16.265s later, full1200s. One GPU attempt COMPLETE:24builds/648commands,574.827s,5720samples,1014513664byte peak,0.1266745s maxgap; bothworkers exit0/reaped/kernelidentityretired,finalownedcensusempty,lease released,no cleanup errors or retry. All648outputs byte-identical to oracle. Worker protocol spans280.032/282.748s. Population267/552->275/554 identical bothworkers/unchangedclose:19boundharness additions,11provider removals,2entry/pathchanges,2AppleGPUimages explained,no new third-party package. Fixed rule WITHHELD_NULL_RESOLUTION for all19 contrasts: worker0 null envelopes4.986961/12.780876us; worker1 6.588302/3.296386us,limit5us. No selection/trimming/widening/relabeling. Raw F_U215.217/215.503us versus C_U234.776/233.297us retains19.559/17.793us lead; S/SB/F expansion leads retained, not discarded or promoted. Not backend/end-to-end speedup. Capacity86%free128GiB aggregate; no GPUquiet/thermalclaim. External caffeinate watched ownedcontroller,started afterlaunch,exited0; fully recorded as operating-condition helper,not instrumentchild. Results fa512-fragment-resolution-results-20261029 HANDOFFcf1079fc/HASHES105ce83a,27sealedfiles/1147retainedpins. Analysis/population6jobs and finalverify3jobs across system3.9/sharedproof3.14 allpass/byte-equal stdout/empty stderr; original63491files/7487links and physicalpopulation verified unchanged. Fable results handoff ready locally,not sent; no further design round requested. Next retain components and prepare source-preserving actual-backend lowering integration/coverage, with unresolved observer/carryover and all numerical/safety contracts explicit. No product/default change or performance/release credit. All59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

2026-10-29 Codex: accepted Fable resolution results review c71747f9. Added sealed-result corrections and a concrete backend integration plan in fa512-backend-fragment-integration-plan-20261029 (HANDOFF aa0a1c25, HASHES fe50c05e). Independently reduced retained cycle means: S/SB/F rise 112-130 us within both workers; all 19 labels remain WITHHELD_NULL_RESOLUTION. Recorded 576 statistical rows versus 648 correctness rows, nine provider files plus two stdlib removals, 27 commands before keep-awake, capacity capture/record distinction, and issuer/verify provenance gaps. Actual generator recreates accepted C and C_U byte-for-byte with source-block32 tail settings. 72 CPU emission cases, five retained source pairs, four adverse controls and five route-predicate checks pass under system and both proof labels with identical output; 1126 source/evidence files unchanged. Three probe-setup failures retained, not product defects. No production/default change, framework/device/compiler/GPU work or new timing. Primary next implementation: default-off dense producer composition with expansion plus shared coefficient rescale, retaining S/B/SB and F+B; preserve all other routes, cache identity and source semantics. One concrete patch/readiness review after implementation, no further standalone design review requested. Kernel-local leads remain useful even if later host costs mask end-to-end gains; no speedup or release credit. All 59 obligations, N1(release), RG-02 and CP-01 OPEN/HOLD.

2026-10-29 Codex: implemented backend fragment compositions in five product files, new CPU tests/fixtures and FA-FRAGMENT-INTEGRATION.txt. experimental_fa_fragment_lowering defaults off; seven closed modes include literal-row C_U/S_U/B_U/SB_U/F_U and new F_U+B. Real retained TTGIR through GenericLowerer and emit_msl recreates six accepted sources byte-for-byte; new shared-output differs only in eight epilogue barriers, with coefficient publication and block-end synchronization retained. Existing source/role/value proofs precede selection; wrong recurrence refuses before maker. Scale/initializer/half-accumulate neighbors preserve original emission; default98-case matrix includes asymmetric/biased/mask/LSE/causal/precision variants. Cache hashes separate modes; explicit requests get selected/reason metadata; descriptor remains unchanged. Native/pure proof labels each pass258 CPU tests under refusing audit launcher with conftest/plugins disabled; same3.14 binary, not independent replication. System3.9 and both proof labels agree on98 default source hashes/seven generated sources; no Metal/objc/torch import, named library load, MSL compilation, GPU, ps or lease. Native Triton CPU IR and current-process Python ctypes handle allowed. Initial own barrier-count test failure and narrow guard correction disclosed. Evidence in fa512-backend-fragment-integration-cpu-20261029, live16-file snapshot and reproducible delta. Prior1124 evidence/source inputs unchanged; two deliberate live-source pin changes supersede old plan, no sealed-results mutation. Source/test/gate digest now c8c2966f64a479bd796b6ac57d3d2eec2126b424c689df875fc26e212e01558d (parent32f1998b). No native qualification for F_U+B, installed credit or timing relabeling. Next CPU work: concrete native/public-path correctness instrument with actual emitted sources, then one combined Fable readiness review; no new prose-design review. All19 timing labels withheld; all59 obligations, N1(release), RG-02 and CP-01 OPEN/HOLD.

2026-10-29 Codex: built concrete backend-composition native correctness instrument, fa512-backend-native-preparation-20261029 HANDOFFc9ef3f65/HASHESf7b6a913; GPU569f6988/CPUd0819f15. Seven exact backend-emitted sources C/C_U/S_U/B_U/SB_U/F_U/F_UB,322commands,23unchangedfixtures/twopoisons,276baselinecomparisons. Allcaps/numerics/lease/fullbyte/snapshot/cleanup protections retained;420sworker,~256sprojection from prior367s,notbound. NewF_UB remainsunqualified. ActualTTGIR/emitter/dispatch_flash_attention passes7sources×23geometries with syntheticruntime plus post-invocation noreplay; direct-native correctness and CPUpublicwiring are explicitly separate, actualJIT/Torch/installed/end-to-end remainsopen. Realpreflightreducer->authoritygate regression passes,stale5/10arm populations refuse.47jobs pass:204tests eachprooflabel(shared3.14binary),200system3.9tests;15commonstdout identical across3,backendstdout identical across2. Inventory61916files/7487links retainsall61574parentinputs unchanged,adds342;103currentproduct/test/contractpins. Backendpatch unchanged,Everify passes. Initialunissuedphase-label mismatch and undeliveredseal-order mismatch retained/corrected beforedelivery; no native/preflight/ps/framework/compiler/device/GPU/lease action,onlyCPU TritonIR and ctypescurrentprocess handle in backendtest. Lockstat unchanged,spentroot empty,noattempt/authority. One combinedFable readinessreview next,covering existingbackendpatch plus concreteinstrument; no further standalone designround. All19timinglabels withheld;all59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

2026-10-30 Codex: accepted Fable combined review8e3eaa1c (Q1/Q2yes,Q3blocked D1). Our path-copy preparation retained a stale environment_source digest; repaired exactlythat CPUheaderfield in fa512-backend-native-pin-repair-20261030/BINDING.json CPU87ca14fc. ParentCPUd0819f15 sealed/unspent/superseded; GPU569f6988,backend/product/runtime/shader/fixtures/counts/caps remainbyte-identical. check_pin drives realenvironmentloader and actualcontrollerprefix against old/newheaders; oldrefuses beforeclock, newpasses to deliberateprelaunchdeadline beforeattemptaccess. RealGate accepts successorheaderlocation underexpiredtemporarytestdecision; realclockrefuses. derive_binding joins actualsnapshot/GPUcontroller/SOURCE-PINS/manifest to regenerateonepin. Realreducer->authority10controls reused with newCPUheader. Final9jobs across3environments pass19tests each plusderivation,identicalstdout; prooflabels share3.14binary. FullverifyPASS56successorfiles,61916inputrows,7487links andparentpacketpopulations unchanged. HANDOFF540a0034/HASHESa592d0cc. Emptyspentroot0755->0700 recorded,lockstatunchanged,noattempt/authority/native/preflight/ps/GPU/lease. Correctedprojection~266sworker/276scontroller includesgaps/close;no capchange. Minorunusedtuple/docstring/placeholder/localassert notesretained withoutscopecreep. Next Fable one-pin delta readinessreview before CPUdecision; no newdesignround. All19timinglabels withheld,all59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

2026-10-30 Codex: executed user yes approving CPU87ca14fc then conditionalGPU569f6988 under Fablee930d781 withoutsecondpause. CPU COMPLETE69.138s,433samples,1001193472bytepeak,cleanupempty. Receipt/clearance6.72256safterCPU,entry21.011618slater,900swindow. GPU COMPLETE7builds/322commands,279.334s,2693samples,1013776384bytepeak,maxgap0.124541s,exit0/reaped/finalownedcensusempty/lease released/no cleanup errors/no retry;599.649sclearanceunused. All322rows pass23unchangedfixtureoracles/twopoisons,guards/inputs/geometry/sourcepipelineownership;all276baselinecandidatecomparisons byte-identical toC,including46F_UB. Maxabsoracleerror0.0001627604;do not claimallFP64referencebyteequal. All7pipelines1024maxthreads/32width/21888scratch. Pop267->272modules:16boundharnessadds,9provider+2stdlibremovals,2entry/bootstrapchanges;552->554images2OS-build-boundAppleGPUimages identicalprior;closepopulationstable. Direct-native emittedsource correctness only;actualpublicJIT/Torch/installed/end-to-end unqualified,experimentaldefaultoff. Hostprotocolprepare28.279s,meancommand0.701252s,close1.466s,span268.496s budgetaccountonly. Nineofflinejobs3interpreters passbyteequal/emptystderr;fullverify31resultfiles/683pins/61916boundrows/7487linksPASS,priorsealsandproductunchanged. Results fa512-backend-native-results-20261030 HANDOFFc39d6484/HASHES355f951e. LocalFableresultshandoffready,nothingexternallysent. NextCPUpreparation actualpublic-path integration coverage,retainallcompositions/reusecorrectnessevidence;freshreview/binding/decisionbeforefutureGPU. All19timinglabelswithheld,all59obligations,N1release,RG-02,CP-01 OPEN/HOLD.

2026-10-30 Codex: accepted Fable backend-native results review0c5ea20a and recorded its CPU-clock/half-rounding/population/budget/provenance corrections in a new addendum; old results untouched. Implemented public-path CPU checkpoint fa512-backend-public-path-preparation-20261030 (HASHES144fed22,185files,473inputpins), not a native-ready binding or another standalone design-review request. Concrete kernel[grid] adapter, Torch-owned guarded buffers, terminal owner retention, 322-call acceptance sequence, exact-code route observer and actual Torch pipeline-capacity checks. Real CPU JIT binder/frontend census:84geometry/mode rows,3signatures/21mode-specializations,14MSL hashes.22fixtures reproduce7accepted sources; N=1 produces7variants with exactly the runtime-N parameter removal and N=1 constant binding; retained, not discarded. Final qualification003:16jobs all0;32tests system/34eachproof plus84frontendcases eachproof;stdout equal, prooflabels sharebinary. Three full322-call synthetic sequences;161actualPython-dispatch/fake-library cases eachproof. Prepares21JITspecializations/14Torchsources before workload; actual native construction, module freeze and compiler-child observer remain to wire and bind for ONE combined readiness review. No framework/device/ps/lease/GPU execution or timing; public source retrieval only. Prior31/51-file seals verified,lock stat inode189249476,size0; no start-of-turn before snapshot claimed. Product source digest remainsc8c2966f. All compositions retained;19timinglabels withheld;N1release,RG-02,CP-01 and all59 obligations OPEN/HOLD.

2026-10-30 Codex: advanced public-path implementation in fa512-backend-public-path-integration-20261030 (HASHES c9ff74e5,318 files,750 pins), no separate partial review requested. Real adapter now composes with Session/controller drive,325-request protocol, persisted frozen population plus independent controller acknowledgement,322-call guarded-buffer journal/offline reduction and terminal owner retention. Added strict 14-source compiler grammar,13-module origin/import plan,provider callable/code pins, bounded JSON wire and startup-only descendant admission around retained strict authenticator/monitor. Real compiler/cache bodies with fake identities/device/tool outputs prove21 preparations->14 builds/7 hits; cache corruption after startup seal refuses. Final qualification002:29 guarded CPU jobs PASS,72 system tests/73 eachproof (218 total), shared stdout equal; prooflabels samebinary. Three seal verifies PASS; prior185-file checkpoint,31-file native results and51-file Fable review unchanged; product digestc8c2966f unchanged; lock stat inode189249476,size0. No frameworks,Metal compiler,ps,signals,lease,device,GPU or timing ran; test interpreters and product CPU extensions only. Native worker/factory and controller/Peer wiring,provider preflight,real final-header authority regressions,new resource account and fresh binding remain BEFORE one combined readiness review. All7 compositions and23 fixtures retained;19 timing labels withheld;N1release,RG-02,CP-01 and all59 obligations OPEN/HOLD.

2026-10-30 Codex: completed concrete public-path native readiness candidate fa512-backend-public-native-candidate-20261030 with sibling provider packet; HANDOFF76dd196e/HASHES4ebd5426 (465 files), GPUe684c4a6/CPU4d98eced. Native factory/worker and supervised controller now drive actual Triton-JIT/MetalLauncher/FA-dispatch/Torch route,325 requests/322 public calls,all7 compositions/23 fixtures/two poisons,21 specializations/14 sources; public calls explicitly not native-command count. Provider supplement resolves30 public symbols and4 route code objects after13-module imports; actual shared-header reducer/authority regression uses expired fixtures and full inventory. Startup compiler admission, frozen acknowledgement/root-only transition, independent saved-buffer reduction, terminal owners and bounded direct-child cleanup fallback connected. Proposed900s worker/1080s work/60s cleanup/1440s clearance; no completion forecast, public costs unmeasured. qualification002:56 CPU jobs all0,215 system/216 eachproof tests (647 executions),common stdout equal;prooflabels sharebinary. qualification001 six inherited-harness failures retained and repaired, no runtime relaxation. Three final full verifies PASS:62147 input rows/7487links,all61916parent rows unchanged. 1080 parent/source pins;prior318/185/31/51-file seals and16 product files unchanged;source digestc8c2966f,lock inode189249476,size0. New spentroot empty0700,bothattemptroots absent,noauthorityissued. No framework/device/Metalcompiler/ps/signals/lease/GPU/timing ran;test interpreters and product CPU extensions only. ONE combined Fable readiness review next,ready locally not sent;no separate design round. All19timing labels withheld;N1release,RG-02,CP-01 and all59 obligations OPEN/HOLD.

2026-10-30 Codex: accepted Fable readiness review b522582c (Q1/Q3 yes,Q2 no D1-D3) and built sealed successor fa512-backend-public-native-rev2-candidate-20261030 with provider sibling. HANDOFF9c4f63aa/HASHES18602d16 (362 files), GPU914db0c1/CPUc41b8784. D1 complete110-file worktree package now bound/admitted in CPU stage; old91-file subset red control refuses. D2 derived29-module snapshot closure includes tensor_buffers; real hook test with all evidence paths removed from sys.path reproduces old missing import and passes successor. D3 explicit python-absent helper policy checks owner None, registry absence and PathFinder absence; unchanged native default still refuses; no helper build or installed claim. CPU supplement checks package/helper/closure before any GPU spend. Preserved7 authority/13 guard/6 package checks per call, counted through all322 fake-provider calls; review-estimate structural scaling2.0875 gives2.505..3.966s/call,not timing/bound. Proposed2400sworker/2700swork/60scleanup/3060sclearance; hard600sstartupseal anchored beforebootstrap,30sEOFwait. CPU120s worker cap unchanged/first-use. GPU decision link now enforced; shader source UTF8 explicit; full Xcode27.0SDK bound and runtime resolver response checked. One read-only xcrun SDK-path query retained with warning; no compiler/device/ps/signal/lease/GPU execution.59 guarded CPU replay jobs PASS,227system/228eachproof tests (683 executions),common stdout equal; prooflabels samebinary. Three full verifies PASS:96010inputfiles/15629links,all62147parentrows retained unchanged;1693evidencepins,parent465-file andreview146-file seals unchanged. Product/fixtures/sevencompositions/acceptance unchanged;lockstat unchanged,old/newspentrootsempty,attemptsabsent,noauthority. ONE Fable delta review next;ready locally,nothing sent. All19timing labels withheld;N1release,RG-02,CP-01 and all59obligations OPEN/HOLD.

2026-10-30 Codex: acted on user yes for CPU c41b8784 + conditional GPU914db0c1 under Fable ed19d38b. CPU entry ran once93.934s and stopped UNAVAILABLE at worker public-provider-surface (ValueError; original reason erased). Controller provider passed/exit0; worker forced termination/reap exit-9, empty owned census, cleanup complete:false.680samples/1262747648Bpeak. No GPU authority, lease, attempt or spend.25 raw files retained in fa512-backend-public-provider-stop-and-diagnostics-20261030; offline analysis equal under3interpreters. Independently reproduced pinned Torch classes synthetic-namespace refusal without Torch/native import; NOT attributed as actual first failure. Built concrete rev3 successor: explicit torch.ops/classes source-owner bytes/aliases/class+method-code pins, unknown dynamic modules still refuse; bounded step/ordinal/exact-literal diagnostics retained through COMPOSITION+STOP. Shader7compositions/23fixtures/two poisons/322calls/325requests/acceptance/limits unchanged. Final62 guarded jobs PASS,247system/248eachproof tests(743executions),common stdout equal; prooflabels samebinary. Initial9 failed jobs retained; fixed helper extraction and a caught truthiness-before-type-check regression. CPUc1609470/GPU017f8f90,HANDOFF82bd8cc2/HASHES31f5813b(499files),96405inputs/15629links,all96010parent rows unchanged. Parent362-file seal/Fable152-file review/lock inode189249476 unchanged; old/newGPU spent roots empty,rev3attempts absent. No native retry during repair. ONE combined Fable delta review next,ready locally,nothing sent. All19timing labels withheld;N1release,RG-02,CP-01 and all59obligations OPEN/HOLD.

2026-10-30 Codex: addressed Fable D4 review38616937 in sealed rev4 public-provider population repair. Fixed source-owner policy covers11 unconditional Torch objects plus optional xnnpack (12); pinned-source class-body fixtures exercise full population before-device/freeze/closing, availability branches, root/owner/alias/MRO/method/receiver changes and hostile hooks. Rev3 is the red control and remains unlaunched. Unknown dynamic subclasses still refuse; native extension population is not proven. Source census2845Pythonfiles/94candidatefiles; direct-raise diagnostics and bounded known-name/ordinal progress retained; root identity pinned; added-file diffs include /dev/null. Corrected original-stop caveats (incomplete discovery/memory,53.400ssampling,worker-only peak,STARTINGphase,unverified consent,entryValueError); original cause remains unknown.68guarded CPUjobs PASS,257system/258eachproof tests(773executions),9rebound/analysisjobs and3finalfullverifies PASS,commonstdout equal. Inherited cache replay imports Triton CPUextension with fake tools; no provider entry,Torch/Metal import,compiler executable,ps,lease,device orGPU work. CPU7ba7bab7/GPU63daed47,HANDOFFdaa305c6/HASHES42ff0af7,96832inputs/15629links,all96405parent rows unchanged. Parent499-file seal/Fable176-file review/25rawrecords andlockinode189249476 unchanged;rev2/3/4GPUspentrootsempty,rev3/4attemptsabsent. ONE Fable delta review next via fa512-backend-public-native-rev4-candidate-20261030/HANDOFF.txt;ready locally,nothing sent. Product source/test/gate digest unchanged;all19timing labels withheld;N1release,RG-02,CP-01 andall59obligations OPEN/HOLD.

2026-10-30 Codex: executed direct user yes for rev4 CPU7ba7bab7 and conditionalGPU63daed47 under Opus25f27629. Prelaunch fullverify passed,aggregate memory86%free. CPUentry once95.027s,UNAVAILABLE after31legacy checks/all13imports at loaded-origin-check:ValueError,reason null,module ordinal null. Controller provider passed;worker forced termination/reap-9,empty owned census,cleanup complete:false.744samples/1557168128Bpeak,58.094ssampling/72.281scontrollerUTC. No GPU authority/preflight/receipt/clearance/lease/attempt/spend.26rawfiles retained;3offline recounts match. Actual failing path/reason remains unknown; cleared module progress points toward following image check under unchanged-input assumption,not proof of specific cause. No guessed inventory expansion. Prepared sealed rev5 refusal-only owned-image diagnostic: fixed unbound/changed-provider reasons,optional sink reuses resolved path/membership,private bounded image path/index retained through COMPOSITION/STOP/controller;no extra census/resolve/byte read,no admission relaxation,success records unchanged. Additional path-retention scope needs explicit next decision after ONE focused Opus delta review,not Fable/full redesign.71guarded CPUjobs PASS(264system/265eachproof=794tests),3stoprecounts and3fullverifies PASS;commonstdout equal. Prior384-file seal/Opus197-file review/rawrecords/lockinode189249476 unchanged;allGPUspentrootsempty,newattemptsabsent. CPU0c2d419d/GPUde0465fe,HANDOFFe866a582/HASHESd6ec3e09. Inherited test imports Triton CPUextension with fake tools; no successor native entry or GPU run. Runtime cause remains unproved;all19timing labels withheld;N1release,RG-02,CP-01/all59obligations OPEN/HOLD.

2026-10-30 Codex: addressed Opus rev5 review 0c9ff11e with sealed rev6 exact libssl inventory repair. Pinned _ssl Mach-O declares /opt/homebrew/opt/openssl@3/lib/libssl.3.dylib; the resolved 888928-byte file SHA c6ee3732 was absent while _ssl/libcrypto were bound. Added exactly that non-evidence file plus the openssl@3 selection symlink in both CPU/GPU headers. No admission relaxation: old real checked/image_records refuse, new admits, wrong/missing digest and retargeted/uncovered selection still refuse. Runtime Python/snapshots byte-equal to rev5 after mechanical active-lane paths; seven compositions/23 fixtures/two poisons/322 public calls/325 requests/acceptance/caps unchanged. Recomputed inherited cost-account arithmetic for one additional link; no measured bound. Original rev4 stop cause remains unproved. Carried Opus diagnostic limits and explicit all-public-stage image-refusal sinks/stdlib-extension/Homebrew import decision scope into COMMANDS. 18 targeted guarded jobs PASS,59 tests per environment (177 total),3 full verifiers PASS with equal stdout; proof labels share a binary. Nine initial scratch-wrapper refusals retained, harness path corrected, only affected jobs rerun. Broader71 rev5 jobs reused by byte equivalence, not rerun. All97283 parent rows unchanged;97717 rows/15630 links;1290 delivery pins verified. Parent392-file seal/Opus188-file review/26 original raw records/lockinode189249476 unchanged; rev2–rev6 GPU spent roots empty,rev5/rev6 attempts absent. CPU517c4623/GPUee1a1cb4,HANDOFFb6836946/HASHES9362b3cb,246 packet files. Ready locally for ONE focused Opus delta review via fa512-backend-public-native-rev6-candidate-20261030/HANDOFF.txt; nothing sent, no new decision/native entry/framework/ps/lease/device/GPU work. Product source/test/gate digest unchanged;19 timing labels withheld;N1release,RG-02,CP-01/all59 obligations OPEN/HOLD.

2026-10-30 Codex: executed approved rev6 CPU517c4623 under Opus9ee44672; both provider children VERIFIED, exit0 and reaped, worker passed31 legacy records plus public surface and loaded libssl with pinned bytes. Parent controller then stopped UNAVAILABLE at closing-population: exactly7 late instrument imports, no removals/origin changes. CPUentry151.633s,controllerUTC118.030s,1097samples/1596768256Bpeak,finalownedcensusempty,cleanup noerrors;27rawfiles retained. No GPU preflight/authority/lease/attempt/spend. Diagnosed protocol.validate_result lazy public_stage import after parent freeze; existing tests preloaded it. Rev7 one functional runtime edit imports public_stage in controller_body before unchanged freeze. Cold actual-controller/protocol/Guard.check regression reproduces old exact7 additions, new success, unexpected-module and schema refusals; peers/images/freeze/authority explicitly doubled, no native qualification. 23 targeted jobs PASS;42 inherited tests/environment,4 cold variants/prooflabel; prooflabels same3.14binary, system3.9 cold model excluded with reason retained. Three final full verifiers PASS:98144rows/15630links,all97717parentrows unchanged. Parent246-file seal/Opus180-file review/rawrecords/lockinode189249476 unchanged; GPUspentroots empty,rev7attempts absent. CPUc24512d0/GPU14f1a2c9,HANDOFF5464bf12/HASHESc4259e71,246packetfiles. ONE focused Opus stop-and-delta review ready locally at fa512-backend-public-native-rev7-candidate-20261030/HANDOFF.txt; nothing sent, no rev7 decision or native retry. All7compositions/23fixtures/322calls/acceptance/caps retained.19timinglabels withheld;N1release,RG-02,CP-01/all59obligations OPEN/HOLD.

2026-10-30 Codex: closed Opus rev7 B1/N2 review3c4d571c in test-only rev8 successor. Prior targeted23-job scope missed stale check_selectors historical controller comparison; no waiver. New exact-one-line import normalizer keeps all other text drift visible, with missing/duplicate/alias/extra-import/extra-statement controls. Sealed rev7 test reproduces exactly one B1 failure under all3interpreters. Accepted cold population regression copied into CPU lane and normal full replay, fresh process per old/new/unexpected/schema case on both proof labels; real controller/protocol/Snapshots/Guard.check, freeze hashing/closing/native images/peers/authority doubled as disclosed; system3.9 excluded only from cold3.14model. Runtime unchanged except mechanical active-lane paths. Full79jobs PASS:71inherited+8cold,265system/266eachproof unittests(797executions);6extra delta/red jobs match expected verdicts. Three final fullverifiers PASS,98600inventoryfiles/15630links,all98144parentrows unchanged,2294deliverypins. CPU899f27f4/GPU7587c701,HANDOFFa812d653/HASHES482a2847,410sealedfiles. Parent246-file packet and Opus review/input pins unchanged;lockinode189249476 unchanged,rev2..8GPUspentroots empty,rev7/rev8attempts absent. Inherited cache test loads Triton CPUextension with fake tools;no framework/native entry/ps/compiler executable/lease/GPU run. ONE short Opus delta handoff ready locally at fa512-backend-public-native-rev8-candidate-20261030/HANDOFF.txt; nothing sent, no new approval. All7compositions/23fixtures/322calls/acceptance/caps retained;19timinglabels withheld;N1release,RG-02,CP-01/all59obligations OPEN/HOLD.

2026-10-30 Codex: executed direct user yes for rev8 CPU899f27f4 + conditionalGPU7587c701 under Opus3c4d571c/1ad48f3b. Prepared scope corrected900sCPU versus unchanged3060sGPU wording before consent; issuer delta retained. CPUentry163.706s exit0, both providers verified and parent closing passed. Fresh GPUauthority0.193s afterCPUcompletion,entry0.097s later,no secondpause. GPU COMPLETE_DIAGNOSTIC exit0 after845.649s,all322real public JIT/launcher/attention/dispatch calls/325requests,7compositions/23fixtures/two poisons,21specializations/14sources. All276candidate comparisons byte-identical toC;161poisonpairs agree;fixed saved-buffer acceptance reproduced,maxerror0.00016276041666662966. Workerexit0,reaped,cleanupcomplete/errors[],lease released,one spend,finalownedcensusempty;GPU7534samples/1699905536Bpeak,1.163smaxgap;CPU1181samples/1635368960Bpeak. CPUworker1216modules/580images toGPU2132/596:926added,10removed,10changed modules,16addedimages;allfileorigins/bytesmanifest-bound,source-owner delta retained;causal importer-chain notclaimed. Frozen equalsclosing. Retained448rawfiles/20967915bytes. Nine offline verify/analyze/check jobs PASS/equal across3interpreters;prooflabels onebinary,systemseparate;7corruption+2authority-time controls refuse. Full98600-file/15630link inventory,410-filepreparation andbothreviewseals unchanged;lockinode189249476 unchanged. Results HANDOFFb2b70ce9/HASHES88b5d521(33files),ANALYSIS0d44d6d9 in fa512-backend-public-native-rev8-results-20261030. ONE focused Opus results review next,ready locally,nothing sent. Worktree public-path fixture correctness only;keepallcompositions/defaultoff,then scoped performance work. No timing/installed/physical-command-count/fullcoverage/releasecredit;19timinglabels withheld;N1release,RG-02,CP-01/all59obligations OPEN/HOLD.

2026-10-31 Codex: accepted Opus rev8 results review5215076d (Q1-Q3 yes). Added append-only semantics/envelope clarification: all7compositions match the Triton-program oracle, not textbook attention for padded noncausal tails; uniformN1 gives1/32. Test-kernel tail coverage gap tracked separately; no source/oracle change. CPU public-performance boundary prototype now retains all guards and wraps observed public dispatch plus synchronization, explicitly including Python route-observer cost. Fake-clock controls exclude guard delay and reject invalid clocks/partial failures;15inherited+7new tests pass in each of3environments (prooflabels sharebinary);6final replayjobs pass/equal. Prior33-file result,410-file candidate and45-file review seals unchanged;lock identity unchanged. New lane fa512-public-performance-boundary-preparation-20261031,PROGRESS718d1288,HASHESaf009596(31files). Next: integrate exact timing records, balanced controls/schedule and predeclared reducer into ONE concrete timing instrument for combined review; no separate intermediate review. Keepall7compositions/defaultoff;no native/GPU timing performed,no timingcredit,all19labels withheld. N1release,RG-02,CP-01/all59obligations OPEN/HOLD.

2026-10-31 Codex: Blake requested a59-row release burn-down before more instrument work. Created UNSEALED engineering/BURNDOWN-WORKING.txt and .json beside liveledger, deterministic stdlib build_burndown.py and local OPUS-BURNDOWN-REQUEST.txt. Seven workloadgroups retained14/4/10/4/4/6/9; each51workload+8class row names protocol/results/review, experiment/reuse, dependencies/globalstaleness and limitation disclosure. Corrected historicalG5 versuscurrent qualification (varlen changedroute,backward nearparity), three-casehostevidence versusonecostfor14, G4wallexcess versusdevicecost, rev8noncausalworktree versuscausal/MLA/installedcoverage, and existingreviewedcoldincrement. Classclosure requires synthesis; documentation/waiver cannot make currentchecker evidence_ready. Proposed F0researchbaseline then selectedrepairs/FQsource-test-definitionfreeze and installedbuilds before finalreceipts; earlierG5diagnostics mayhelp butwouldstaleafterG1sourcechange. PreservecurrentFA512timingboundary inG3/all7compositions; no newinstrumentfamily untiloneOpusmapreview. Comparedlive59IDs toSeptember24snapshot:59stillopen,0new,0withdrawn,0closed;36historicalwarmrows matchpinneddoc,21planninginputpins verified. No ledgerdefinition/state/receipt,backend/test/native/GPU/ps/lease changes;no pages,nothingexternallysent. N1release,RG-02,CP-01 remainopen. Source/test/gate digest unchanged c8c2966f.

2026-10-31 Codex: resumed concrete G1 CPU implementation under Blake's instruction; map review is not a prerequisite to existing repair work. Implemented pure environment dependency check as an explicit standard-all short-circuit loop, retaining code/default checks, live/eager lookup order, custom-all lazy fallback and unchanged native delegation. Added10 regression cases;54 focused CPU tests pass under guarded Python3.14.4. Real ordinary-environment CPU component probe12alternating blocks x10000calls/arm: median paired saving658.6104ns,candidate/parent0.9065; no public/end-to-end/native-helper timing claim. Earlier immutable JSON-template cache v1-v3 retained but not landed: safeguarding mutable decoder state erased its small gain; no ceiling or rejection of composing gains inferred. Evidence shared-launch-small-snapshot-20261031/RESULTS.json,live engineering/HOST-PATH-PROGRESS.txt. Initial probe refused stdlib ctypes PyDLL(None) before measurement; successor permits current-process handle only, matching CPU launcher; no namedload,framework import,GPU,ps orlease. Product change only_environment_snapshot.py plus newtests; other preexisting backend changes retained. New source/test/gate digestde58ca5e(parentc8c2966f); oldsealedrecords remain historical,no current receipt or label promotion. Next concrete patch/results review with existing G1 comparison;retainall7FAcompositions and19withheldlabels. Counts59stillopen,0new,0withdrawn,0closed;N1release,RG-02,CP-01OPEN/HOLD.

2026-10-31 Codex: accepted Opus snapshot review; product patch unchanged. Added4 discriminating CPU tests;58 focused tests,4 in-memory negative controls and2 real contract-route witnesses pass, with stdout/stderr retained. Actual checker plus independent ordered-input reconstruction reproduce reviewed de58ca5e and parent c8c2966f exactly; reviewer alternatives remain unreproduced. Current digest d58bd366 reflects added tests. Current CPU witnesses and24 accepted historical G1 profiles agree on2 snapshots per pure hook+launcher route,1 resident; historical native profiles have0 Python snapshot calls. About1.3us pure-public contribution is a hypothesis only, no end-to-end/native speed credit. Three exotic divergences explicitly retained; no extra review gate. Follow-up: shared-launch-small-snapshot-20261031/review-followup/FOLLOWUP.txt. Next existing G1 comparison with separately bound baseline/candidate and profile counts separate from timing; preserveall7FAcompositions. No timing/GPU/framework/ps/lease work. Counts59stillopen,0new,0withdrawn,0closed;N1release,RG-02,CP-01OPEN.

2026-10-31 Codex: executed G1 parent/candidate pure-worktree comparison under Blake continuation approval, reusing accepted September25 worker/model/admission byte-for-byte and exclusive project lease. Two current110-file copies differ only in reviewed environment snapshot. Run completed52.01s,4workers exit0/reaped,no cleanup errors,lease released,lock inode189249476 preserved. All real pre/post correctness witnesses pass for add/softmax/matmul_ieee;1440 synchronized public samples,144 modeled batches/28800 measured calls,24 separate profiles. Modeled-public median paired savings add1.627us,softmax1.048us,matmul1.338us;positive per-repetition medians in all6 modeled boundaries;actualprofiles2snapshots public/1resident. Synchronized results MIXED: add+2.400us,matmul+5.957us,softmax-5.865us;softmax cost retained,not noise-dismissed or promoted.42direct-worker RSS samples peak1158.59MiB,not continuous/descendant bound. No native-helper/installed/release claim.9independent arithmetic checks,168worker/controller record matches,4negative controls pass;source/input pins unchanged. Results shared-launch-snapshot-comparison-20261031/FINDINGS.txt sha a8f82733;protocol admission wording narrowed to untimedwitnesses (timedcalls restored production),initial rounding draft retained. Keep accepted small gain and all7FAcompositions;next target shared native/pure overhead with softmax fullcall cost open,not repeated component timing until success.59stillopen,0new,0withdrawn,0closed;N1release,RG-02,CP-01OPEN.

2026-10-31 Codex: continued implementation and real G1 execution. Added shared driver tensor-classification loop retaining live read/order/custom-all fallback and PEP479 cause;15 classifier controls and4000 old/new differential cases pass,4mutants caught.71 focused cases passed before final exception repair;35 native-helper CPU controls pass with3locally built current-source images selected explicitly. Direct device-type prototype retained but not landed (43ns marginal gain); production initial loop CPU probe128ns. Composed snapshot+launcher changes tested in BOTH native and pure worktree copies, exact final repaired sources,2880 synchronized public samples,288modeled batches,48profiles,all pre/post correctness witnesses pass,8workers exit0/reaped/leasesreleased;earlier2880-sample composed run retained. Final modeled public savings pure add1.548us/softmax1.773us/matmul1.924us;native .268/.272/.168us with native softmax repetition conflict. Native resident .198/.124/.122us positive bothreps. Fullcalls remainmixed: native softmax4.640us cost;prior snapshot-only softmaxcost notrelabeled. Independent18summarychecks,336rawrecordmatches,8negativecontrols pass. Sourcepackagepinsverified; inherited informational INPUTS sourcedigest stale d58 corrected by WORKTREE-STATE sidecar, notusedbyruntimegate;actualdigest 46f3b582. Results shared-launch-composed-host-rev2-comparison-20261031/FINDINGS.txt;no installed-wheel/significance/releaseclaim,no commitpush. Keepcomposablegains/7FAarms, next larger shared-host bottleneck rather than repeat tinytimingsuntilpositive.59stillopen,0new,0withdrawn,0closed;N1release,RG-02,CP-01OPEN.

2026-10-31 Codex: completed actual installed artifact qualification from composed candidate 46f3b582. Frozen625files;native/pure wheels from one sdist,47offline pinned dependencies,16build/install commands pass,four fresh/relocated strict smokes,all installed wheel bytes/RECORDs rechecked. Native885pass6skip,pure793pass98skip under retained CPU guard;3native binder cold orders and separate actual-pure cold pass. Strict focused83native/59pure and18artifact controls pass. Native initial plugin duplication and86sandbox OS-query refusals retained and repaired by launch settings/authorized escalation,with unchanged checks. NoGPU,lease,sourcepatch,publication. Concrete artifact/native_and_pure review request ready;independent review pending,not self-issued. Evidence composed-installed-artifacts-20261031/FINDINGS.txt sha5a696636;1110-file seal. Preserve all G1 fullcall costs/FAwithheldlabels.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: G5 installed native/pure correctness completed for varlen48/32 and attentionforwardbackward:92publiccalls184guardedcustomcommands,30retainedarrays pass unchanged oracles and byte-equal modes/repeats;independent FP64 errors<8e-7,52dispatch/30output negatives pass. Harness import-order and route-helper assertion failures retained and repaired without product changes. Working map corrected to September16 fullmatrix provenance;September25 elevenportable comparison excludes G5. Static proof finds public-main varlen unmasked Q/K/V staging reads out of bounds;current source has exact guards and is unaffected. Historicalmain denominator quarantined,no unsafe rerun. Existing artifact review remains pending; no release receipts. Source46f3b582 unchanged;59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: finished G5 installed correctness plus matched warm comparison. Four current native/pure cases pass;92calls184guardedcommands and30arrays byte-equal across modes. Matched run against intact compatible September24 wheel payloads completed87.588s,3840timedrequests,224correctnesswitnesses,472arrays independently checked,16workers exit0/reaped and lease released.4048rawworker/controller records and exact-rational interval endpoints reproduce;peakcombinedownedRSS960.7MiB. All4fixed-rule timinglabels UNRESOLVED; no unchangedrerun to chasepass. Historical public-main varlen is statically OOB and never rerun;exact871 identity refusal before samples retained,notbypassed. Productsource46f3b582unchanged. Artifact review remains pending;59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: resumed after reauthentication and completed isolated G1 guard-proof composition prototype. Both parent/candidate71identity tests pass;960000CPU contract calls in2freshprocesses show paired savings0.287us/0.235us,below declared0.5us threshold. Added native-observer mutation control catches3calls becoming1,so prototype NOTLANDED; no production defect/newobjective. Retained lead for later composition with observer fix,not discarded for lacking end-to-end win. Wrapper-only query/libSystem/NumPy setup failures repaired and retained. Product46f3b582 unchanged;current artifact/G5evidence remains current. Independent artifact results review pending.59stillopen,0new,0withdrawn,0closed.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/shared-launch-guard-batch-prototype-20261031/FINDINGS.txt

2026-10-31 Codex: completed remaining G1 installed correctness in native/pure for layernorm1024x1024,quantdecode512x1024,bandwidth16M,assert_valid. Eightworkers0/reaped,184publiccalls/guardedcustomcommands,24savedarrays pass unchanged independent references and are byte-equal modes/repeats. Native3helpers,pure0;inputs unchanged. Run70.207s,sampledpeak1789.25MiB,lease released. First wrapper false-alias failure retained/repaired using storage identity instead of opaque MPS-handle interval arithmetic;no product defect. Source46f3b582 unchanged. Results g1-installed-continuity-20261031/FINDINGS.txt sha da8176e7,361sealedfiles. Paired warm comparison now running on same four existing rows/mode under inherited fixed rule;no newobjectives,review/receipt pending.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed existing G1 remaining-four matched warm comparison in native/pure against intact September24 installed payloads on same dependencies.32workers0/reaped,320.532s,7680samples,448correctnesswitnesschecks/96savedarrays all pass and byte-equal arms/modes/repetitions.8096rawresponses independently matched,96corruptioncontrols,208combinedRSSwindows peak3613.98MiB below4GiB,lease released. All8fixed labels UNRESOLVED with intervals crossing0;no trimming/relabeling/repeatuntilpass,no performancecredit. Tiny hostgains andpriorsoftmaxcosts retained. Source46f3 unchanged. Evidence g1-remaining-matched-warm-20261031/FINDINGS.txt. Existing artifactresultsreview still pending;no self-issued receipt.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed decisive existing G1 public/runtime boundary diagnostic on assert_valid+layernorm1024x1024 with both actual installed modes. Same shader/tensors/geometry/sync, direct route separately poisoned and verified before timing and every block.16workers0/reaped,3840samples,232witnesschecks/56savedarrays correct and byte-equal routes/modes/repetitions;4048rawresponses recounted,56corruptioncontrols,129.751s,lease released. All4fixed contrasts positive: public-path premium29.48-46.14us native,56.08-74.90us pure;wide layernorm intervals retained. This is fullcall route attribution,not exactCPU/GPU split or a production bypass/speedup. Source46f3 unchanged;all previous unresolved/cost labels preserved. Evidence g1-public-runtime-boundary-20261031/FINDINGS.txt sha e466c31f,584sealedfiles. Retained12profile account ties shared launch/identity/descriptor work to this next optimization target;no unsafe JSON shortcut landed. Independent artifactresultsreview pending.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: advanced G1 through implementation and actual public comparison. Native-assisted JSON snapshot cache regressed1.16-1.20us despite40live checks; retained,not landed. Repaired prior guard-batching lost-observer bug without new native code:71tests,15differential observer cases,mid-check replacement and red3-vs1control pass. Final960000CPU calls save61.7/96.7ns percontract check. Actual public instrument completed44.19s,4workers0/reaped,lease released;1440completed-call samples,28800modeledcalls,36savedarrays all pass independent source-aware oracles and byte-equal arms/stages/reps,36corruptioncontrols,96inputwitnesses,1692rawresponsesmatched. Modeledpublic paired medians save0.405/0.480/0.159us(add/softmax/matmul),bothrepspositive,2hits/public and1/resident confirmed. Fullcall add/matmul lean faster;softmax costs0.600us descriptively,all3ranges cross0,matmulresident conflicts. Retain this composable CPU lead and every adverse/unresolved result;no performance/release credit or productpromotion. Product remains46f3. Evidence shared-launch-guard-public-comparison-20261031/FINDINGS.txt sha b432aecae,833files;repair lane8124c21a,651files. Next address larger repeated launch mapping/selection with this repair retained for composition,using the same existing panel. Independent artifactresultsreview permission remains pending.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: implemented isolated shared-census native binder and tested composition with retained guard repair.147CPUtests pass17Metal-onlyskips;mutation/callback/partial/pure/cold/watchercapacity controls pass. Matchedclang CPUwholebinder saves218-295ns across2processes;initial unmatchedbinarycomparison retained separately. Matchedpublic4workers complete,36outputs correct/byte-equal,1440samples+28800modeledcalls; modeledpublic savings0.322/0.233/0.259us(add/softmax/matmul),positivebothreps;fullcall mixed. Crucial same-source null reproduces36/36blocks favoringsecondarm,19.7-27.4usmedianpositioneffect. Position-balancedrepairalone failsall3prospectivecoarsenulls;recorded20warmups/arm/block thenaddpasses,softmax+matmulstillfail unchanged5usmedian/10usblockscreen. Firstsampleaccount explains~20usofblockcontrast;no samplestrimmed/relabelled,nullsubtracted,oldcostserased. All5boundedpublicruns complete20workersexit0/reaped/leasesreleased,180savedarrayscorrect;7200measuredsamples+144000modeledcalls,1440additionaldeclaredwarmups,lasttestraw3132responsesrecounted. Product46f3unchanged;candidatekeptforcompositionnotpromoted. Next repair/qualifytheexistingG1warm timingboundary (resource/model/IPC intermissionsvswarmstream) beforemorepatchcredit,notanothermicrooptimizationorunchangedrerun. Independentartifactreviewpermissionpending.59stillopen,0new,0withdrawn,0closed.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/shared-launch-binder-census-20261031/FINDINGS.txt; /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/g1-public-warm-balanced-null-20261031/FINDINGS.txt

2026-10-31 Codex: completed two causal G1 timing-instrument repairs and actual same-source GPU controls. Contiguous worker-local windows remove per-call IPC; first window still faster in31/36 blocks. Symmetric within-block ABBA/BAAB then balances positions exactly, but all3 fixed coarse nulls still fail in both instruments. No thresholds changed, samples trimmed, null subtracted or earlier labels relabelled. Across both:8workers exit0/reaped/leases released,2880 measured calls+4320 retained warmups,57600 modeled calls,48profiles,72savedarrays independently correct and byte-equal;720rawresponses matched. Stop this GPU-null sequence: instrument cannot qualify the small binder gain. Preserve guard/binder composition and CPU evidence; next address larger repeated identity/descriptor launch costs using retained profiles, with live values/custom callbacks intact. Product46f3 unchanged, no newobjective or releasecredit. Independentartifactreview not started (permission unresolved).59stillopen,0new,0withdrawn,0closed.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/g1-public-inprocess-window-null-20261031/FINDINGS.txt; /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/g1-public-window-crossover-null-20261031/FINDINGS.txt

2026-10-31 Codex: recovered and completed pending native identity representation experiments. Module-attribute specialization passes1023 differential cases/282 candidate CPU tests but actual four hot programs contain zero eligible module attributes and first paired median regresses14.77185ns; rejected without second run. Type-tag predecode passes282 inherited CPU tests,19 differential cases and a compiled stale-tag mutant; two fixed fresh processes save only1.717675ns/0.2771ns per134-probe group,7/12positiveblocks each. Retained but not promoted: nominally positive is not useful evidence for8KiB extra cache. Exact-rational recount verifies timings/binaries;11candidate CPUguard receipts show no device attempts and correct loaded paths. Both experiments retain all failures/builds,330/290-file seals. Workload-specific retained profile account separates add/softmax JSON from45-node matmul checked-copy; next target is larger live descriptor/launch work with original callback order, not more tag-cache tuning or another failed-null GPU rerun. Product46f3 unchanged; prior binder/guard gains, fullcall softmax costs, allFA compositions and withheld labels retained. Artifact independent review not started; it does not block implementation.59stillopen,0new,0withdrawn,0closed.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/shared-launch-module-probe-20261031/FINDINGS.txt; /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/shared-launch-type-tag-program-20261031/FINDINGS.txt

2026-10-31 Codex: completed a larger checked small-descriptor implementation after native identity experiments. It removes both JSON traversals only behind99live provider predicates and exact small-scalar snapshot checks, retaining full original fallback.1000generatedcases,11provider mutations,latecallback/error and alias tests pass; actual broken-gate negative control demonstrates why observation cannot be omitted. Fixed first-process whole-boundary comparison regresses4.17-4.48us across all4cases,so rejected with no secondprocess/noGPU/no productchange.17-file sealed evidence shared-launch-small-descriptor-prototype-20261031/FINDINGS.txt e291cd7d. This rejects the guarded Python shortcut, not all descriptor representations. Native module/type experiments are separately sealed330/290files; no more cache-only microtweaks or unchanged failed-null GPU reruns. Next target remains repeated launch preparation under existing G1, preserving live values/callbacks, binder/guard leads, adverse softmax outcomes and allFAcompositions. Independentartifactreview remains pending and is not global implementation blocker. Product46f3 unchanged;59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed actual-installed G1 exceptional correctness for native/pure add+softmax,using unchanged accepted source/geometry and four fresh workers.92publiccalls/guardedcommands,12retainedoutputs all pass independent FP64 add and per-row exceptional-aware FP64softmax;allstages/modes byte-equal. Addfinitebits/signedzeros exact;23arithmeticpatterns includeoverflow,rounding,NaN/infinity. Softmax16patterns,48NaNrows and80finite rows checkedseparately,maxfiniteerror1.4901161193847656e-08 atunchanged1e-6/1e-5tolerance.26prelaunch+24finalcorruptioncontrols pass;8rawresponsesmatched,allartifact/sourcepinsunchanged,4workers0/reaped,38.348slease released,peakownedRSS486309888bytes/16samples. Preparationcorruptionthatroundedaway wascaught/repaired beforeGPU. No timingcredit,denormalcoverage,NaNpayloadguarantee or newobjective. Product46f3 unchanged;previoushostgains and adverse/unresolved performance remain. Evidence g1-installed-exceptional-20261031/FINDINGS.txt sha516b4a01,185-file seal. Independentartifactreview permission requested with explicitrepo/scope;no reviewer started without answer. Continue existingG1 implementation/qualification;59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed G1 kernel-handle reuse investigation and real GPU diagnostic. Pinned Torch source confirms fresh native wrapper creation on each lookup despite cached pipelines. Isolated guarded reuse passes CPU mutation/error controls and1052valid GPUcommands in4freshworkers;112retained exceptional add/softmax outputs pass independent FP64 recount,108withinworker byte comparisons,8corruption controls; concurrent ordinary callers,alternating libraries and refusal recovery pass.96000lookups save only18-56ns;960synchronized direct dispatches are mixed across environments,6/12positiveblocks each. Do not promote this negligible cache or infer a public speedup.4.149slease,allchildren0/reaped,16RSSsamples peak237355008bytes,8rawresponsesmatch,lockinode unchanged.157sealedfiles, FINDINGS sha2de30748 in g1-kernel-handle-reuse-20261031. Product46f3 unchanged; earlier binder/guard gains,public regressions,failednulls and allFAcompositions retained. Independentartifactreview not started without delegation approval. Continue larger repeated public-launch work; native wrapper allocation is now ruled out as a material explanation of the29-75uspublic premium.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed actual G1 elementwise loop-expansion and compiler-limit composition experiments, preserving the user's requirement to test combinations after an isolated regression. Current add/bandwidth source C, explicit eight-body X, and both with maximum-threadgroup128 (actual dispatch unchanged128). Exactbody/indexproof13073cases+2mutants; offlinehint-onlyU retained/unrun. First C/X GPUrun1096commands, expansion bandwidthcost121.700/207.223us. Separate predeclared four-arm balancedposition/neighbor run2192commands testsC/X/C128/X128; no contrast wins inbothworkers, so no default/productpromotion or speedcredit. Combined4workers3288commands,576logical output witnesses, independentFP64/finitebits/signedzero/NaN/inf/tailguard/input checks allpass;12independentcorruptionscaught. Allchildren0/reaped,leasesreleased,lockinode unchanged,workerpeaks<1.5GiB. CPUattribute-placement failure and prematureprelaunchprint retained/repaired beforeGPU. Four-arm reporting omittedC/Xpipelinevalues despiteunconditional>=128checks; real-loop fakeprobe reproduces, reporting-onlyrepairpassesandremainsunexecuted, no timingrerun. Evidence289/549files, findings fa1dd32d/257b7d72 in g1-elementwise-unroll[-composition]-20261031. Product46f3 and59ledgerdefinitions unchanged. Prior public null failures,costs,binder/guard leads andallFAcompositions retained. Native-generation batching notimplemented because ordinary/callback admission stillneedsproof. ContinueexistingG1work, no newobjectives orselfreview.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed causal G1 scheduling diagnostic under existing continuation. Same-source crossover now contrasts forcedUTILITY with inherited requestedclass33, in four fixed UTILITY/INHERITED/INHERITED/UTILITY phases.8workers0/reaped,72.61655slease,lockinode189249476unchanged,peakdirectworkerRSS1069154304bytes.2880timedsamples+5760warmups+57600modeledcalls,48profiles;72retainedoutputs independently correct/byteequal,72corruptioncontrols,192inputwitnesses,792rawresponsesmatch. Everycase fails unchanged5usmedian/10usallblock coarse screen in BOTH conditions; loweringpriority is not sufficient explanation, no productQoSchange/no rerununchanged. Querybefore/close confirmsactualrequestedclasses;no core/thermal/exacthostGPUattribution. Protocol/collector unchanged except fixedfour-phase schedule and own-threadclassobservation. Existing shaderexpansion/limit regressions,guard/bindercomponent gains,softmaxcosts and allFAcompositions retained. Evidence 1206files,g1-public-scheduling-null-20261031/FINDINGS.txt shae45d0fda. Product46f3 and59definitions unchanged. Next primary implementation target remains repeated public-launch preparation; scalar eligibility rewalk is now located at driver.py:663, post-hook live proofs mustremain. Independentartifactreview notstarted without delegation approval and not a globalengineeringblock.59stillopen,0new,0withdrawn,0closed.

2026-10-31 Codex: completed G1 scalar-eligibility composition, including actual public execution. Retained guard+binder gains in botharms, live nativei32/standardpointer shortcut only incandidate; unchanged fullPython fallback. Fixed lazyimport and active-monitor recovery beforeGPU, plus atomiccoldconfiguration. Parent/finalcandidate228CPUtests17skips each,2038differentials,9extracontrols,8local/globalmonitoring recoverycases,7compiledmutants,63guardreceipts with0deviceattempts. TwofixedCPUprocesses save151-153nsaddlike,621-624nssoftmaxlike,3.24-3.28us16scalar; customfallbackcost31.6ns retained. Realpublic4workers0/reaped,30.99745slease released,lockinodeunchanged,1440timed+2880warmups+28800modeledcalls/24profiles.36savedoutputs independentlycorrect/byteequal,36corruptions,96inputwitnesses,396rawresponses,4countermutants. Actualscalarcounts12276hits/800profilefallbacks,matmul0; profilingrecovery verified inrealpath. Allpublictiming remainsUNQUALIFIED: earlierfailednulls stand, and unaffectedmatmul alsohasmodeledsaving,so do notattribute allinterworker differences toscalar.113packagefiles/arm unchanged,peakdirectRSS1066303488bytes. Candidate retainednotpromoted,pinnedCPython3.14internalmonitoringheaderexplicit; no receipt/selfreview/newobjective. Product46f3 unchanged; allpriorcosts/compositions retained. Evidence g1-scalar-eligibility-composition-20261031/FINDINGS.txt sha315afe5a25d045849517ea6778402ff4085bf46ebaf8009572a890d4668754d1. Continue largerexistingG1 work; independentartifact/compositionreviewpending withoutdelegationapproval, notglobalblock.59stillopen,0new,0withdrawn,0closed.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/g1-scalar-eligibility-composition-20261031/FINDINGS.txt

2026-10-31 Codex: completed the composed G1 public-throughput comparison against current product 46f3, with two identical-source worker controls per implementation. Two fresh four-worker panels, fixed Williams positions/neighbors, 192 batches of 256 public calls after 16 warmups. All 52,800 native binder admissions succeeded; composed scalar path had 17,600 hits and no misses. All 264 retained outputs independently correct and byte-equal; 264 input witnesses, 240 raw responses, six exact-rational reductions and corruption controls pass. All three throughput labels are UNRESOLVED under the rule fixed before execution: identical-worker variation exceeds the composition estimates. No rule widening, control subtraction, unchanged rerun or replacement of synchronous warm criteria. Eight workers exited 0/reaped, 67.474893 s lease released, peak sampled direct-worker RSS 2,124,169,216 bytes, lock unchanged. Preparation harness mistakes retained and repaired before GPU. Evidence has 1,250 sealed files, FINDINGS SHA256 5e7bfe5683a36834984a8c1c9e0597b6f6f4e05f3b9bd7b8fa07465660219c31. Product source and all 59 definitions unchanged; every prior gain/cost/composition retained. Blake explicitly approved one independent artifact/native_and_pure review, now running read-only with no GPU or ledger authority. Address any concrete findings, then issue a receipt only if that existing row is independently supported. 59 still open, 0 new, 0 withdrawn, 0 closed pending review.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/g1-composed-throughput-20261031/FINDINGS.txt

2026-10-31 Codex: artifact/native_and_pure is now evidence_ready under independently reviewed source-bound receipt /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/artifact-native-pure-closure-20261031/RECEIPT.json. Review a6ce271c passed after independently verifying the separate six-fixture source-continuity repair. All original artifact records and failures retained; 1,110 artifact files and 636 continuity files reverified. Product/source/test/gate digest remains 46f3b58223bf2f1e575b2ed648af20796b02833f05d0bdd2128dff5122ee4c19. No acceptance definitions, IDs, GPU labels or release bar changed. 58 still open, 0 new, 0 withdrawn, 1 closed (evidence_ready). Continue existing G1 work: guard/binder/scalar composition retained; all three composed-throughput labels remain UNRESOLVED because identical-source variation exceeds estimated gains. Do not rerun that unchanged instrument or widen its rule; investigate remaining launch-preparation cost with semantics-preserving differential controls. No additional delegation, publication or release approval inferred.

2026-10-31 Codex: completed packed-copy lookup reuse on the retained guard+binder+scalar composition, through CPU qualification and real public GPU execution. CPython3.14 Unicode-only live dictionaries reuse four module-owned names; custom/general keys retain original fresh identity, all live provider lookups and recursive edges unchanged. Parent/candidate136tests each,2013differential controls, two compiled mutants caught, namespace collection/trace/profile/subclass controls pass. Two CPU processes save0.21-0.22us flat and0.70-0.77us nested; leaf/fallback costs retained. Actual matmul uses45nativecopy calls/launch and modeled-public saving1.224/1.249us; synchronized results mixed. Add/softmax do not use copier yet synchronized costs observed, preserved without attribution. All timing remains UNQUALIFIED under prior failed same-source controls; no threshold widening or unchanged rerun. Fourworkers exit0/reaped,43.362889slease released,peak sampled directRSS1065238528bytes,lock unchanged.36savedoutputs independently correct/byteequal,96inputwitnesses,396rawresponses and reduction controls pass. Premature offline analysis missing finalstate retained; same livehandle completed, no GPU retry. Evidence /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/g1-packed-key-reuse-20261031/FINDINGS.txt SHA256 21df91013d2a1f462c4b6d0d888e3ee02ef6db4f0ee9810b9a95435f11f2ac3f; 1724sealedfiles. Product/source46f3 and artifact receipt unchanged. 58stillopen,0new,0withdrawn,1evidence_ready. Continue larger existing G1 launch-preparation work with all gains and unresolved regressions retained; no new objective or self-issued review.

2026-11-01 Codex: implemented and executed the existing TriFast external-consumer outcome in isolated candidate /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-external-cooperative-20261101/candidate. Default-off cooperative lowering uses the ordinary source proof/compiler/cache; portable export command and ABI documentation added. Original reporter frontend equals retained IR source. Two real native workers complete/reap under shared project leases: 192 correctness commands plus64warmups/192timed commands;384output/LSE blobs independently recounted against directFP64 oracle, errors<1e-6, canaries/inputs intact. Bothbranches N32/64/128/256 atH=batch1. MaskedN32 warm external submit/wait327.780->183.399us and GPU193.036->56.451us pass predefined development screen; other3cells remainUNRESOLVED due same-source variation. NoM1/fullXLA/Julia/Python-JIT/release claim. Final244CPUtests pass, actual export command compiles bothbranches,14file patch applies. Product source46f3 and artifact receipt unchanged; candidate not promoted without independent review/artifact qualification. Findings /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-external-cooperative-20261101/FINDINGS.txt SHA256 e268db42dd24dac9f8b78eeedc6c44c74393537c31eb6ee3f1e27e8827cf839f. Preserve all prior compositions; next existing objective is installed/external reporter integration, not another instrument design. 58open,0new,0withdrawn,1evidence_ready.

2026-11-01 Codex: installed TriFast cooperative external continuation completed. Fresh private native/pure wheels built offline from exact candidate patch, ordinary artifact checker and pip check pass. Separate cloned dependency environments, backend force-reinstalled;244CPUtests each, allbackend origins in respective site-packages,3native/0purehelpers. Original reporter frontend compiles bothbranches through INSTALLED backend. Standalone Objective-C Metal consumer (no Python runtime) executes64cases/installation,128GPUcommands total;256output/LSEblobs independently correct and byteequal to sourcecandidate, all input witnesses/canaries intact. N32/64/128/256,H=batch1,D32. Native/pure childrenexit0/reaped;peak182MB;bothprojectleasesreleased. Exact source remains46f3 and oldartifactreceipt unchanged. No newtiming, noJulia/XLA orM1claim, no publication/selfissuedreview. Missing Julia executable in checked local locations; actual workflow/hardware qualification remains. Findings /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-installed-external-20261101/FINDINGS.txt SHA256 4e7b40b2e582d7ed12150eef0d5d6daf209ed035e7aa30670428214a9908165f. Existing59definitions unchanged:58open,0new,0withdrawn,1evidence_ready.


2026-11-01 Codex: actual reporter Julia forward workflow now passes. Pinned Janssena LuxTriangleAttention and KernelDispatch sources; isolated official Julia1.12.7, Metal1.10.0/Reactant0.2.304. Six real TriFast adapter cases N32/64 plus six full TriangleAttention layer cases (starting/ending, no mask/holes/all masked) pass independent FP64/host comparisons. Maximum layer difference 8.94e-7; inputs unchanged. Ten upstream CPU custom-call IR tests pass. Both shader branches compile as Metal3.0; full layer uses that binary on M4. Finite mask sentinel preserved. Two GPU children exit0/reaped and leases released; CPU trace child exit0. No timing, backward, XLA GPU or M1 execution claim. No main product source change or artifact receipt change. Findings /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-reporter-workflow-inquiry-20261101/FINDINGS.txt SHA256 51515535dbf3e979e6ee6f7b8f4c164d338fc20f69f02a74ab95e4efac9bb0fd. Existing59 definitions unchanged:58open,0new,0withdrawn,1evidence_ready.


2026-11-01 Codex: M4 backward/XLA continuation completed with mixed results retained. Real MPS PJRT full TriangleAttention baseline passes six cases after a process-local broadcast-axis normalization; maxerror1.073e-6, unchanged tolerance, no timing. Unchanged plugin fails before attention; small repro and exact swapped-axis model identify unsorted broadcast handling, reporter already has a public fix branch. All39 normalized broadcasts in9graphs independently inverted, no other graph changes;80indexmodels and38redcontrols. Custom mps.metal_kernel_lib remains absent from published plugin/three inspected reporter branches; custom build requested. All3 native backward kernels safely refuse derived-stride address resolution before Metal generation. Local backward ABI repair matches actual pinned source; original fully-masked dV defect reproduced and local mask repair passes36CPUinterpreter checks, FP64autograd independently confirms oracle. No native backward qualification. Seven GPU-leased children reaped (3expectedfailure,4exit0), leases released;203offline checks pass.990sealedfiles,294inputpins. No main product or artifact receipt change. Findings /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-m4-backward-xla-20261101/FINDINGS.txt SHA256 371ac5660359c83832cadb4d0787ade2e9ee781ab82dce7d2b254ad010cdbd0b. Existing59definitions:58open,0new,0withdrawn,1evidence_ready.


2026-11-02 Codex: repaired TriFast backward address resolution in isolated candidate and executed real M4 forward/backward via Metal.jl/KernelDispatch. Derived strides, bool-pointer address and rebuilt dbias loop supported without replacing value/bounds proofs. Source-paired ABI, canonical K transpose, explicit fully-masked dV correction and multi-head layout packing composed. Final9cases/45quantities pass independentFP64 gradients, max3.094454e-7; all originalH1 results byte-equal after packing. H2B2 full TriangleAttention forward6cases also pass, max9.537e-7. Initial3native dV failures retained.141CPUregressionspass/4GPUskips,32stridechecks/3realparentrefusals,377offlinechecks/45corruptioncontrols. Five leased nativechildren0/reaped; no timing. Missing customXLAhandler and physicalM1 remain unavailable.913sealedfiles/408inputpins. Main46f3 source and artifactreceipt unchanged; local patch ready for independent review and artifact qualification, no productpromotion/selfissuedreceipt. Findings /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-backward-repair-20261102/FINDINGS.txt SHA256 f9b32d29077ebc6640bdfd9f85e01f78b47d1512483510614c8fd0a6bba1d3d9. Existing59definitions:58open,0new,0withdrawn,1evidence_ready.

2026-11-02 Codex: implemented missing mps.metal_kernel_lib in private jax-mps candidate and built pinned native dependencies. Actual reporter Reactant/PJRT custom-kernel forward and explicit backward now pass on M4: nine cases/45 quantities match independent FP64 oracles (max3.094454e-7), byte-equal to eager Metal; full H2/B2 TriangleAttention layer six cases pass (max9.536743e-7). Native scalar/layout/zero-fill/snapshot/error controls and unsupported ABI-type refusals pass. Corrected initial layout diagnosis: original reporter binder retained; failed experimental override and harness failures preserved. Final relocated workflow bundle passes all15 multihead quantities, 275 pins verified; actual M1 still requires hardware and automatic training AD/timing remain unqualified. Main source46f3 and artifact receipt unchanged; no source promotion, publication or independent-review receipt. Findings /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/trifast-xla-plugin-20261102/FINDINGS.txt SHA256 4bb634a3828d9f5bd4d99fb3354508036aa2dfbf467ed5b4804019e565ec3013. Existing59definitions:58open,0new,0withdrawn,1evidence_ready.

2026-11-03 — integrated 0.4.0rc1 candidate completed (local, unpublished)
One candidate branch carries development work, cooperative exporter and backward
stride repairs. Native/pure wheels built from one sdist; 1306/1214 CPU passes,
79 distinct public workflow tests per mode,256 external output/LSE comparisons,
30 Julia eager/XLA output-gradient comparisons and 6 whole-layer cases pass on M4.
All seven attention modes execute and agree byte-for-byte. Fixed missing Metal
toolchain, installed-test source lookup and missing SciPy test dependency; failed
runs retained. No tolerance/guard relaxation, performance credit or M1 claim.
Evidence: /Users/bledden/Documents/triton-validation-evidence/2026-09-19-0.4.0-start/release-candidate-integration-20261103
Source: docs/RELEASE_CANDIDATE_0.4.0.md. Release criteria and IDs unchanged.
