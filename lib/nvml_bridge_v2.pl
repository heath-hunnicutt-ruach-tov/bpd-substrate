%% nvml_bridge_v2.pl — NVML bridge using declarative llvm_emit infrastructure.
%%
%% The bridge specification is FACTS. The emission is RULES from llvm_emit.pl.
%% Compare to nvml_bridge_emitter.pl where everything was format strings.
%%
%% Usage:
%%   swipl -g 'emit_bridge("nvml_bridge.ll"), halt' nvml_bridge_v2.pl
%%   llc -relocation-model=pic -filetype=obj nvml_bridge.ll -o nvml_bridge.o
%%   gcc -shared -o nvml_bridge.so nvml_bridge.o -ldl
%%
%% Author: mavchin (2026-06-01)

:- module(nvml_bridge_v2, [emit_bridge/1]).

:- use_module(llvm_emit).

%% ============================================================
%% Bridge specification — PURE FACTS
%% ============================================================

%% The library we bridge to
bridge_library('libnvidia-ml.so.1').

%% NVML functions we need (resolved via dlsym at runtime)
%% nvml_func(PrologName, NvmlSymbol, RetType, ArgTypes, FnPtrGlobal)
nvml_func(init,       'nvmlInit_v2',                         i32, [],                fn_init).
nvml_func(shutdown,   'nvmlShutdown',                        i32, [],                fn_shutdown).
nvml_func(gethandle,  'nvmlDeviceGetHandleByIndex_v2',       i32, [i32, ptr],         fn_gethandle).
nvml_func(getname,    'nvmlDeviceGetName',                   i32, [ptr, ptr, i32],    fn_getname).
nvml_func(gettemp,    'nvmlDeviceGetTemperature',            i32, [ptr, i32, ptr],    fn_gettemp).
nvml_func(getpower,   'nvmlDeviceGetPowerUsage',             i32, [ptr, ptr],         fn_getpower).
nvml_func(getclock,   'nvmlDeviceGetClockInfo',              i32, [ptr, i32, ptr],    fn_getclock).
nvml_func(getmeminfo, 'nvmlDeviceGetMemoryInfo',             i32, [ptr, ptr],         fn_getmeminfo).
nvml_func(getutil,    'nvmlDeviceGetUtilizationRates',       i32, [ptr, ptr],         fn_getutil).
nvml_func(getpciegen, 'nvmlDeviceGetCurrPcieLinkGeneration', i32, [ptr, ptr],         fn_getpciegen).
nvml_func(getpciewid, 'nvmlDeviceGetCurrPcieLinkWidth',      i32, [ptr, ptr],         fn_getpciewid).

%% Prolog predicates exposed by this bridge
%% plf_pred(PrologName, Arity, ImplFunction)
plf_pred(gpu_temperature, 1, pl_gpu_temperature).
plf_pred(gpu_power,       1, pl_gpu_power).
plf_pred(gpu_clock,       2, pl_gpu_clock).
plf_pred(gpu_memory,      3, pl_gpu_memory).
plf_pred(gpu_utilization, 2, pl_gpu_utilization).
plf_pred(gpu_pcie,        2, pl_gpu_pcie).
plf_pred(gpu_name,        1, pl_gpu_name).

%% ============================================================
%% Bridge emission — uses llvm_emit rules
%% ============================================================

emit_bridge(OutFile) :-
    open(OutFile, write, S),
    %% Header
    emit_header(S, "x86_64-unknown-linux-gnu"),
    %% External function declarations (from llvm_emit facts)
    format(S, '; --- External functions (from llvm_extern/3 facts) ---~n', []),
    emit_declares(S),
    format(S, '~n', []),
    %% Global state
    emit_bridge_globals(S),
    %% Helper: resolve_sym
    emit_resolve_sym(S),
    %% Lazy init
    emit_nvml_lazy_init(S),
    %% Predicate implementations
    emit_pred_gpu_temperature(S),
    emit_pred_gpu_power(S),
    emit_pred_gpu_clock(S),
    emit_pred_gpu_memory(S),
    emit_pred_gpu_utilization(S),
    emit_pred_gpu_pcie(S),
    emit_pred_gpu_name(S),
    %% Install
    emit_bridge_install(S),
    close(S),
    format("Emitted NVML bridge v2 to ~w~n", [OutFile]).

%% ============================================================
%% Globals — generated from nvml_func/5 and plf_pred/3 facts
%% ============================================================

emit_bridge_globals(S) :-
    format(S, '; --- Global state (generated from nvml_func/5 facts) ---~n', []),
    format(S, '@lib_handle = internal global ptr null~n', []),
    format(S, '@nvml_dev = internal global ptr null~n', []),
    format(S, '@lib_initialized = internal global i32 0~n', []),
    format(S, '~n', []),
    %% Function pointer globals — one per nvml_func
    forall(nvml_func(_, _, _, _, FnPtr),
           format(S, '@~w = internal global ptr null~n', [FnPtr])),
    format(S, '~n', []),
    %% String constants for NVML symbol names
    bridge_library(LibName),
    llvm_emit:emit_string(S, str_libname, LibName),
    forall(nvml_func(_, NvmlSym, _, _, FnPtr),
           (format(atom(StrName), 'str_~w', [FnPtr]),
            llvm_emit:emit_string(S, StrName, NvmlSym))),
    format(S, '~n', []),
    %% String constants for Prolog predicate names
    forall(plf_pred(PrologName, _, _),
           (format(atom(StrName), 'str_pred_~w', [PrologName]),
            llvm_emit:emit_string(S, StrName, PrologName))),
    format(S, '~n', []).

%% ============================================================
%% resolve_sym helper
%% ============================================================

emit_resolve_sym(S) :-
    format(S, 'define internal ptr @resolve_sym(ptr %lib, ptr %name) {~n', []),
    format(S, '  %fn = call ptr @dlsym(ptr %lib, ptr %name)~n', []),
    format(S, '  ret ptr %fn~n', []),
    format(S, '}~n~n', []).

%% ============================================================
%% Lazy init — generated from nvml_func/5 facts
%% ============================================================

emit_nvml_lazy_init(S) :-
    format(S, '; --- ensure_nvml_init (generated from nvml_func/5 facts) ---~n', []),
    format(S, 'define internal i32 @ensure_nvml_init() {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %already = load i32, ptr @lib_initialized~n', []),
    format(S, '  %done = icmp ne i32 %already, 0~n', []),
    format(S, '  br i1 %done, label %ret_ok, label %do_init~n~n', []),
    format(S, 'do_init:~n', []),
    format(S, '  %lib = call ptr @dlopen(ptr @str_libname, i32 2)~n', []),
    format(S, '  %lib_null = icmp eq ptr %lib, null~n', []),
    format(S, '  br i1 %lib_null, label %ret_fail, label %resolve~n~n', []),
    format(S, 'resolve:~n', []),
    format(S, '  store ptr %lib, ptr @lib_handle~n', []),
    %% Resolve each function pointer — GENERATED from facts
    forall(nvml_func(_, _, _, _, FnPtr),
           (format(atom(StrName), 'str_~w', [FnPtr]),
            format(S, '  %~w_p = call ptr @resolve_sym(ptr %lib, ptr @~w)~n', [FnPtr, StrName]),
            format(S, '  store ptr %~w_p, ptr @~w~n', [FnPtr, FnPtr]))),
    format(S, '~n', []),
    %% Call nvmlInit
    format(S, '  %init_fn = load ptr, ptr @fn_init~n', []),
    format(S, '  %init_ret = call i32 %init_fn()~n', []),
    format(S, '  %init_fail = icmp ne i32 %init_ret, 0~n', []),
    format(S, '  br i1 %init_fail, label %ret_fail, label %get_device~n~n', []),
    %% Get device handle
    format(S, 'get_device:~n', []),
    format(S, '  %devptr = alloca ptr~n', []),
    format(S, '  %gh_fn = load ptr, ptr @fn_gethandle~n', []),
    format(S, '  %gh_ret = call i32 %gh_fn(i32 0, ptr %devptr)~n', []),
    format(S, '  %dev = load ptr, ptr %devptr~n', []),
    format(S, '  store ptr %dev, ptr @nvml_dev~n', []),
    format(S, '  store i32 1, ptr @lib_initialized~n', []),
    format(S, '  br label %ret_ok~n~n', []),
    format(S, 'ret_ok:~n  ret i32 0~n~n', []),
    format(S, 'ret_fail:~n  ret i32 1~n', []),
    format(S, '}~n~n', []).

%% ============================================================
%% Predicate pattern: read-one-i32 (temperature, power)
%% ============================================================

%% Generic pattern: init → load dev → alloca i32 → call fn → unify
emit_read_one_i32(S, FuncName, FnPtr, ExtraArgs) :-
    format(S, 'define i32 @~w(i64 %t0, i32 %arity, ptr %control) {~n', [FuncName]),
    format(S, 'entry:~n', []),
    format(S, '  %ir = call i32 @ensure_nvml_init()~n', []),
    format(S, '  %fail = icmp ne i32 %ir, 0~n', []),
    format(S, '  br i1 %fail, label %err, label %read~n~n', []),
    format(S, 'read:~n', []),
    format(S, '  %dev = load ptr, ptr @nvml_dev~n', []),
    format(S, '  %val_ptr = alloca i32~n', []),
    format(S, '  %fn = load ptr, ptr @~w~n', [FnPtr]),
    format(S, '  call i32 %fn(ptr %dev~w, ptr %val_ptr)~n', [ExtraArgs]),
    format(S, '  %val = load i32, ptr %val_ptr~n', []),
    format(S, '  %val64 = sext i32 %val to i64~n', []),
    format(S, '  %ok = call i32 @PL_unify_integer(i64 %t0, i64 %val64)~n', []),
    format(S, '  ret i32 %ok~n~n', []),
    format(S, 'err:~n  ret i32 0~n', []),
    format(S, '}~n~n', []).

%% Generic pattern: read-two-i32 (clock, pcie, utilization)
emit_read_two_i32(S, FuncName, FnPtr1, Extra1, FnPtr2, Extra2) :-
    format(S, 'define i32 @~w(i64 %t0, i32 %arity, ptr %control) {~n', [FuncName]),
    format(S, 'entry:~n', []),
    format(S, '  %ir = call i32 @ensure_nvml_init()~n', []),
    format(S, '  %fail = icmp ne i32 %ir, 0~n', []),
    format(S, '  br i1 %fail, label %err, label %read~n~n', []),
    format(S, 'read:~n', []),
    format(S, '  %t1 = add i64 %t0, 1~n', []),
    format(S, '  %dev = load ptr, ptr @nvml_dev~n', []),
    format(S, '  %a_ptr = alloca i32~n', []),
    format(S, '  %b_ptr = alloca i32~n', []),
    format(S, '  %fn1 = load ptr, ptr @~w~n', [FnPtr1]),
    format(S, '  %fn2 = load ptr, ptr @~w~n', [FnPtr2]),
    format(S, '  call i32 %fn1(ptr %dev~w, ptr %a_ptr)~n', [Extra1]),
    format(S, '  call i32 %fn2(ptr %dev~w, ptr %b_ptr)~n', [Extra2]),
    format(S, '  %a = load i32, ptr %a_ptr~n', []),
    format(S, '  %b = load i32, ptr %b_ptr~n', []),
    format(S, '  %a64 = sext i32 %a to i64~n', []),
    format(S, '  %b64 = sext i32 %b to i64~n', []),
    format(S, '  %ok1 = call i32 @PL_unify_integer(i64 %t0, i64 %a64)~n', []),
    format(S, '  %ok2 = call i32 @PL_unify_integer(i64 %t1, i64 %b64)~n', []),
    format(S, '  %ok = and i32 %ok1, %ok2~n', []),
    format(S, '  ret i32 %ok~n~n', []),
    format(S, 'err:~n  ret i32 0~n', []),
    format(S, '}~n~n', []).

%% ============================================================
%% Concrete predicates — using the patterns
%% ============================================================

emit_pred_gpu_temperature(S) :-
    format(S, '; --- gpu_temperature(T) → PTHERM ---~n', []),
    emit_read_one_i32(S, pl_gpu_temperature, fn_gettemp, ', i32 0').

emit_pred_gpu_power(S) :-
    format(S, '; --- gpu_power(mW) → power sensor ---~n', []),
    emit_read_one_i32(S, pl_gpu_power, fn_getpower, '').

emit_pred_gpu_clock(S) :-
    format(S, '; --- gpu_clock(SM, Mem) → PCLOCK ---~n', []),
    emit_read_two_i32(S, pl_gpu_clock, fn_getclock, ', i32 0', fn_getclock, ', i32 1').

emit_pred_gpu_utilization(S) :-
    format(S, '; --- gpu_utilization(Gpu%, Mem%) → PCOUNTER ---~n', []),
    %% utilization uses a struct {gpu:i32, mem:i32}
    format(S, 'define i32 @pl_gpu_utilization(i64 %t0, i32 %arity, ptr %control) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %ir = call i32 @ensure_nvml_init()~n', []),
    format(S, '  %fail = icmp ne i32 %ir, 0~n', []),
    format(S, '  br i1 %fail, label %err, label %read~n~nread:~n', []),
    format(S, '  %t1 = add i64 %t0, 1~n', []),
    format(S, '  %dev = load ptr, ptr @nvml_dev~n', []),
    format(S, '  %util = alloca [2 x i32]~n', []),
    format(S, '  %fn = load ptr, ptr @fn_getutil~n', []),
    format(S, '  call i32 %fn(ptr %dev, ptr %util)~n', []),
    format(S, '  %gp = getelementptr [2 x i32], ptr %util, i32 0, i32 0~n', []),
    format(S, '  %mp = getelementptr [2 x i32], ptr %util, i32 0, i32 1~n', []),
    format(S, '  %g = load i32, ptr %gp~n  %m = load i32, ptr %mp~n', []),
    format(S, '  %g64 = sext i32 %g to i64~n  %m64 = sext i32 %m to i64~n', []),
    format(S, '  %ok1 = call i32 @PL_unify_integer(i64 %t0, i64 %g64)~n', []),
    format(S, '  %ok2 = call i32 @PL_unify_integer(i64 %t1, i64 %m64)~n', []),
    format(S, '  %ok = and i32 %ok1, %ok2~n  ret i32 %ok~n~nerr:~n  ret i32 0~n}~n~n', []).

emit_pred_gpu_pcie(S) :-
    format(S, '; --- gpu_pcie(Gen, Width) → PBUS ---~n', []),
    emit_read_two_i32(S, pl_gpu_pcie, fn_getpciegen, '', fn_getpciewid, '').

emit_pred_gpu_memory(S) :-
    format(S, '; --- gpu_memory(Total, Used, Free) → PFB ---~n', []),
    format(S, 'define i32 @pl_gpu_memory(i64 %t0, i32 %arity, ptr %control) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %ir = call i32 @ensure_nvml_init()~n', []),
    format(S, '  %fail = icmp ne i32 %ir, 0~n', []),
    format(S, '  br i1 %fail, label %err, label %read~n~nread:~n', []),
    format(S, '  %t1 = add i64 %t0, 1~n  %t2 = add i64 %t0, 2~n', []),
    format(S, '  %dev = load ptr, ptr @nvml_dev~n', []),
    format(S, '  %mi = alloca [3 x i64]~n', []),
    format(S, '  %fn = load ptr, ptr @fn_getmeminfo~n', []),
    format(S, '  call i32 %fn(ptr %dev, ptr %mi)~n', []),
    format(S, '  %tp = getelementptr [3 x i64], ptr %mi, i32 0, i32 0~n', []),
    format(S, '  %fp = getelementptr [3 x i64], ptr %mi, i32 0, i32 1~n', []),
    format(S, '  %up = getelementptr [3 x i64], ptr %mi, i32 0, i32 2~n', []),
    format(S, '  %t = load i64, ptr %tp~n  %f = load i64, ptr %fp~n  %u = load i64, ptr %up~n', []),
    format(S, '  %mb = add i64 1048576, 0~n', []),
    format(S, '  %tm = udiv i64 %t, %mb~n  %um = udiv i64 %u, %mb~n  %fm = udiv i64 %f, %mb~n', []),
    format(S, '  %ok1 = call i32 @PL_unify_integer(i64 %t0, i64 %tm)~n', []),
    format(S, '  %ok2 = call i32 @PL_unify_integer(i64 %t1, i64 %um)~n', []),
    format(S, '  %ok3 = call i32 @PL_unify_integer(i64 %t2, i64 %fm)~n', []),
    format(S, '  %a = and i32 %ok1, %ok2~n  %ok = and i32 %a, %ok3~n  ret i32 %ok~n~nerr:~n  ret i32 0~n}~n~n', []).

emit_pred_gpu_name(S) :-
    format(S, '; --- gpu_name(Name) → PMC ---~n', []),
    format(S, 'define i32 @pl_gpu_name(i64 %t0, i32 %arity, ptr %control) {~n', []),
    format(S, 'entry:~n', []),
    format(S, '  %ir = call i32 @ensure_nvml_init()~n', []),
    format(S, '  %fail = icmp ne i32 %ir, 0~n', []),
    format(S, '  br i1 %fail, label %err, label %read~n~nread:~n', []),
    format(S, '  %dev = load ptr, ptr @nvml_dev~n', []),
    format(S, '  %buf = alloca [256 x i8]~n', []),
    format(S, '  %fn = load ptr, ptr @fn_getname~n', []),
    format(S, '  call i32 %fn(ptr %dev, ptr %buf, i32 256)~n', []),
    format(S, '  %ok = call i32 @PL_unify_atom_chars(i64 %t0, ptr %buf)~n', []),
    format(S, '  ret i32 %ok~n~nerr:~n  ret i32 0~n}~n~n', []).

%% ============================================================
%% Install — generated from plf_pred/3 facts
%% ============================================================

emit_bridge_install(S) :-
    format(S, '; --- install (generated from plf_pred/3 facts) ---~n', []),
    format(S, 'define void @install() {~n', []),
    forall(plf_pred(PrologName, Arity, ImplFunc),
           (format(atom(StrName), 'str_pred_~w', [PrologName]),
            format(S, '  call i32 @PL_register_foreign(ptr @~w, i32 ~w, ptr @~w, i32 0)~n',
                   [StrName, Arity, ImplFunc]))),
    format(S, '  ret void~n}~n', []).
