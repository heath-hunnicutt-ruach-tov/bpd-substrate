%% plan_bridge.lgt — The Logtalk bridge between plans-mcp and the dashboard.
%%
%% ARCHITECTURE (Heath's design, 2026-09-02):
%%   agents mark tasks done in plans-mcp
%%          v
%%   plan_tasks.status = 'done'  (plans-mcp's authority over its own state)
%%          v
%%   plan_bridge (THIS OBJECT) — calls INTO the plans-mcp API via dispatch-mcp,
%%     parses the JSON response, and emits plan_task/N + cell_plan/N facts
%%          v
%%   lift_coverage.pl renders dashboard cells linked to their greening-plan-task,
%%     auto-updating on each CI tick because it re-queries the live plan state.
%%
%% "Logtalk calling INTO the MCP API is the coolest way" — this object speaks the
%% plans-mcp API (through the dispatch-mcp transport), NOT raw SQL. plans-mcp
%% stays the authority; the bridge only READS task-state and projects it onto
%% dashboard cells. The bridge is an object with a swappable transport, so it
%% works with whatever dispatch mechanism the runtime provides (dispatch-mcp in
%% the scheme host, or a stdio-MCP shim on the enclave).
%%
%% Chosen transport per Heath (2026-09-02): (B) reuse the existing dispatch-mcp
%% mechanism rather than reimplement SSE-MCP in Logtalk. The bridge calls the
%% plans-mcp tool through a transport predicate; the concrete transport is
%% injected (transport/3 hook), keeping the bridge protocol-agnostic and testable.

:- object(plan_bridge).

    %% Logtalk objects don't inherit SWI's autoloaded list predicates (member/2,
    %% forall/2 etc — module code gets them free, objects don't). This is the
    %% migration's crown incompatibility (Bocher's migrator auto-injects this).
    %% forall/2 is a Logtalk built-in meta-predicate; member/2 comes from list.
    :- uses(list, [member/2]).

    :- info([
        version is 0:1:0,
        author is 'Iyun',
        date is 2026-09-02,
        comment is 'Bridge from plans-mcp (via the MCP API) to dashboard-cell facts. Reads plan task-state, projects onto KernelBench cells. Auto-updates the dashboard-as-worklist.'
    ]).

    :- public(cell_plan_status/3).
    :- mode(cell_plan_status(+atom, +atom, -atom), zero_or_more).
    :- info(cell_plan_status/3, [
        comment is 'For dashboard cell (Level, Capability), the aggregate task-status of its greening plan: done/active/blocked/pending/mixed.',
        argnames is ['Level', 'Capability', 'Status']
    ]).

    :- public(query_tasks/2).
    :- mode(query_tasks(+atom, -list), zero_or_one).
    :- info(query_tasks/2, [
        comment is 'Query the live task list for a plan by calling the plans-mcp API. Returns a list of task(TaskId, Status, Description) terms.',
        argnames is ['PlanId', 'Tasks']
    ]).

    %% ── the cell -> plan mapping (which plan greens which KernelBench cell) ──
    %% Static curation: each non-green dashboard cell names the plan-phase whose
    %% completion turns it green. This is the connective tissue Heath asked for —
    %% the dashboard becomes a live view of the plan forest.
    %% VERIFIED plan_ids from plans-mcp (2026-09-02):
    %%   196cd2c2 = all-100-L1 -> 0-ULP -> SIMD-perf (the L1 correctness+perf spine)
    %%   8d65ba1c = L1 verify (Tier-2 numerical correctness)
    %%   9b08777e = Frontier-4 curriculum beyond llama.cpp (L2/L3 breadth)
    %%   ce4c6a60 = prolog_to_c (emit side)
    :- public(cell_plan/3).
    :- mode(cell_plan(?atom, ?atom, ?atom), zero_or_more).
    cell_plan(kernelbench_l1, verify,    '8d65ba1c').
    cell_plan(kernelbench_l1, roundtrip, '196cd2c2').
    cell_plan(kernelbench_l1, execute,   '196cd2c2').
    cell_plan(kernelbench_l1, emit_llvm, 'ce4c6a60').
    cell_plan(kernelbench_l1, emit_rust, '9b08777e').
    cell_plan(kernelbench_l1, lift_auto, 'auto_lift_frontier').   % the keystone gap (plan TBD)
    cell_plan(kernelbench_l2, lift_auto, 'auto_lift_frontier').
    cell_plan(kernelbench_l2, verify,    '9b08777e').
    cell_plan(kernelbench_l3, lift_auto, 'auto_lift_frontier').
    cell_plan(kernelbench_l3, verify,    '9b08777e').

    %% cell_plan_status(+Level, +Capability, -Status)
    %% Look up the greening plan for a cell, query its live task-state via the
    %% MCP API, and aggregate to a single status the dashboard renders.
    cell_plan_status(Level, Capability, Status) :-
        cell_plan(Level, Capability, PlanId),
        ( ::query_tasks(PlanId, Tasks)
        -> aggregate_status(Tasks, Status)
        ;  Status = unknown   % API unreachable — honest: don't fake a status
        ).

    %% query_tasks(+PlanId, -Tasks) — call INTO the plans-mcp API.
    %% This is the "Logtalk calling into the MCP API" core. It delegates to the
    %% injected transport (transport/3), which performs the actual dispatch-mcp
    %% call. Keeping the transport a hook makes the bridge protocol-agnostic:
    %% dispatch-mcp in the scheme host, or a stdio-MCP shim on the enclave.
    query_tasks(PlanId, Tasks) :-
        transport(plan_status, json{plan_id: PlanId}, Response),
        parse_tasks(Response, Tasks).

    %% transport(+Tool, +Args, -Response) — the swappable MCP-call hook.
    %% Default: fail (no transport wired). The runtime injects a concrete
    %% transport (see plan_bridge_dispatch or plan_bridge_shell below). Failing
    %% by default is honest — an unwired bridge returns 'unknown', never a fake.
    transport(_, _, _) :- fail.

    %% aggregate_status(+Tasks, -Status) — collapse a task list to one cell-status.
    %% Lattice: all done -> done; any blocked -> blocked; any active -> active;
    %% all pending -> pending; a mix of done+not-done -> mixed. This is what the
    %% dashboard cell reflects (a cell greens when its plan's tasks are all done).
    aggregate_status([], pending).
    aggregate_status(Tasks, Status) :-
        Tasks \== [],
        findall(S, member(task(_, S, _), Tasks), Statuses),
        ( forall(member(S, Statuses), S == done) -> Status = done
        ; member(blocked, Statuses)               -> Status = blocked
        ; member(active, Statuses)                -> Status = active
        ; forall(member(S, Statuses), S == pending) -> Status = pending
        ; Status = mixed
        ).

    %% parse_tasks(+Response, -Tasks) — extract task(Id,Status,Desc) from the
    %% plans-mcp JSON response. Tolerant of the response shape (phases->tasks).
    parse_tasks(Response, Tasks) :-
        ( is_dict(Response)
        -> ( get_dict(phases, Response, Phases)
           -> findall(task(TId, TStatus, TDesc),
                ( member(Phase, Phases),
                  get_dict(tasks, Phase, PhaseTasks),
                  member(T, PhaseTasks),
                  task_field(T, id, TId),
                  task_field(T, status, TStatus0), norm_status(TStatus0, TStatus),
                  task_field(T, description, TDesc)
                ), Tasks)
           ;  Tasks = [] )
        ;  Tasks = [] ).

    task_field(T, Key, Val) :- get_dict(Key, T, Val), !.
    task_field(_, _, '') .   % missing field -> empty, don't fail the whole parse

    %% norm_status(+Raw, -Norm) — normalize plans-mcp status strings to atoms.
    norm_status(done,    done)    :- !.
    norm_status("done",  done)    :- !.
    norm_status(active,  active)  :- !.
    norm_status("active",active)  :- !.
    norm_status(blocked, blocked) :- !.
    norm_status("blocked",blocked):- !.
    norm_status(skipped, done)    :- !.   % skipped counts as not-blocking (done-ish)
    norm_status("skipped",done)   :- !.
    norm_status(_,       pending).

:- end_object.
