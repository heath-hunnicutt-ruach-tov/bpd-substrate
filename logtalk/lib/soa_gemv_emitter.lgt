:- protocol(soa_gemv_emitterp).
    :- public(emit_soa_gemv/1).
:- end_protocol.

:- object(soa_gemv_emitter,
    implements(soa_gemv_emitterp)).



emit_soa_gemv(OutFile) :-
    open(OutFile, write, S),
    format(S, '; Q8_0 SoA gemv: dst[r] = dot(W_row_r, activation), over nrows rows.~n', []),
    format(S, '; Composes @bpd_q8_0_dot (rung-1, 0-ULP gated).~n~n', []),
    format(S, 'declare float @bpd_q8_0_dot(ptr %wq, ptr %wd, ptr %aq, ptr %ad, i32 %nb)~n~n', []),
    emit_gemv_function(S),
    close(S),
    format("Emitted Q8_0 SoA gemv to ~w~n", [OutFile]).

emit_gemv_function(S) :-
    format(S, 'define void @bpd_soa_gemv_q8_0(ptr %wq, ptr %wd, ptr %aq, ptr %ad, ptr %dst, i32 %nrows, i32 %nb, i32 %ncols_dst) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %nr_gt = icmp sgt i32 %nrows, 0~n', []),
    format(S, '  br i1 %nr_gt, label %row.head, label %done~n~n', []),
    format(S, 'row.head:~n', []),
    format(S, '  %r = phi i32 [ 0, %entry ], [ %r.next, %row.tail ]~n', []),
    %% weight row r: wq offset = r*nb*32, wd offset = r*nb
    format(S, '  %wq.qstride = mul i32 %r, %nb~n', []),
    format(S, '  %wq.qoff = mul i32 %wq.qstride, 32~n', []),
    format(S, '  %wq.row = getelementptr i8, ptr %wq, i32 %wq.qoff~n', []),
    format(S, '  %wd.row = getelementptr float, ptr %wd, i32 %wq.qstride~n', []),
    %% dst[r] = dot(W_row_r, activation)
    format(S, '  %d = call float @bpd_q8_0_dot(ptr %wq.row, ptr %wd.row, ptr %aq, ptr %ad, i32 %nb)~n', []),
    format(S, '  %dst.r = getelementptr float, ptr %dst, i32 %r~n', []),
    format(S, '  store float %d, ptr %dst.r~n', []),
    format(S, '  br label %row.tail~n~n', []),
    format(S, 'row.tail:~n', []),
    format(S, '  %r.next = add i32 %r, 1~n', []),
    format(S, '  %r.lt = icmp slt i32 %r.next, %nrows~n', []),
    format(S, '  br i1 %r.lt, label %row.head, label %done~n~n', []),
    format(S, 'done:~n', []),
    format(S, '  ret void~n', []),
    format(S, '}~n', []).

:- end_object.
