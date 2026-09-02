:- protocol(dispatchp).
    :- public(dispatch/3).
    :- public(dispatch_unchecked/3).
    :- public(register_graph/2).
    :- public(set_dispatch_policy/1).
    :- public(dispatch_policy/1).
:- end_protocol.

:- object(dispatch,
    implements(dispatchp)).

    :- uses(compute_graph_invariants, [assert_tensor/7, assert_op/4, check_all_invariants/1, check_view_containment/2, check_view_alignment/2, check_stride_consistency/2, check_write_read_coherence/3, check_temporal_ordering/3, check_shape_compatibility/3, diagnose_graph/1, clear_graph/0, check_dtype_coherence/3, check_dtype_flow/3, check_dtype_chain/2, trace_dtype_chain/2, check_scale_coherence/3, assert_scale_convention/2, scale_application_path/2, scale_matches_oracle/2]).


    :- uses(list, [member/2, length/2]).
    :- uses(meta, [include/3]).




:- dynamic dispatch_policy/1.
:- dynamic registered_graph/2.

%% Default policy: always check invariants before dispatch
dispatch_policy(check_always).

%! set_dispatch_policy(+Policy) is det.
%  Policy is one of:
%    check_always    — run all invariants before every dispatch (safest)
%    check_once      — run invariants on first dispatch, cache result
%    check_never     — skip invariant checks (fastest, UNSAFE)
%    check_errors    — only check error-level invariants, skip warnings
set_dispatch_policy(Policy) :-
    retractall(dispatch_policy(_)),
    assertz(dispatch_policy(Policy)).

%! register_graph(+KernelName, +GraphSetupGoal) is det.
%  Register a compute graph for a kernel. GraphSetupGoal is a Prolog
%  goal that asserts the tensor and op facts for this kernel's graph.
register_graph(KernelName, GraphSetupGoal) :-
    retractall(registered_graph(KernelName, _)),
    assertz(registered_graph(KernelName, GraphSetupGoal)).

%! dispatch(+KernelName, +Config, -Result) is det.
%  Safe dispatch: check invariants, then launch.
%  Refuses to launch if any error-level invariant fails.
dispatch(KernelName, Config, Result) :-
    dispatch_policy(Policy),
    (Policy = check_never ->
        dispatch_unchecked(KernelName, Config, Result)
    ;
        %% Set up the graph if registered
        (registered_graph(KernelName, SetupGoal) ->
            clear_graph,
            call(SetupGoal)
        ; true),
        
        %% Run invariant checks
        check_all_invariants(Diagnostics),
        
        %% Separate errors from warnings
        include(is_error_diag, Diagnostics, Errors),
        include(is_warning_diag, Diagnostics, Warnings),
        
        %% Report warnings
        (Warnings \= [] ->
            length(Warnings, NW),
            format('[dispatch] ~w warnings for ~w:~n', [NW, KernelName]),
            forall(member(W, Warnings),
                (W = diag(Class, Subject, warning(Type, Msg)) ->
                    format('  [WARN] ~w/~w: ~w~n', [Class, Subject, Msg])
                ;
                    format('  [WARN] ~w~n', [W])
                ))
        ; true),
        
        %% Check errors
        (Errors \= [] ->
            %% REFUSE TO LAUNCH
            length(Errors, NE),
            format('[dispatch] BLOCKED: ~w error(s) for ~w:~n', [NE, KernelName]),
            forall(member(E, Errors),
                (E = diag(Class, Subject, error(Type, Msg)) ->
                    format('  [ERROR] ~w/~w: ~w~n', [Class, Subject, Msg])
                ;
                    format('  [ERROR] ~w~n', [E])
                )),
            Result = error(pre_dispatch_check_failed, Errors)
        ;
            (Policy = check_errors ->
                dispatch_unchecked(KernelName, Config, Result)
            ;
                %% All clear — dispatch
                format('[dispatch] All invariants passed for ~w. Launching.~n', [KernelName]),
                dispatch_unchecked(KernelName, Config, Result)
            )
        )
    ).

%! dispatch_unchecked(+KernelName, +Config, -Result) is det.
%  Launch without invariant checks. Used internally after checks pass,
%  or when policy is check_never.
dispatch_unchecked(KernelName, Config, Result) :-
    format('[dispatch] Launching ~w with ~w~n', [KernelName, Config]),
    %% Actual kernel launch would go here — via ctypes, FFI, or system call
    %% For now, return a placeholder
    Result = ok(launched(KernelName, Config)).

%% Helper: classify diagnostics
is_error_diag(diag(_, _, error(_, _))).
is_warning_diag(diag(_, _, warning(_, _))).

:- end_object.
