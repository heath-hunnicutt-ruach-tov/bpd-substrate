%% gguf_native_reader.pl — Native Prolog GGUF parser using safe_read.pl
%%
%% Reads GGUF files directly in Prolog with byte-ownership tracking.
%% No shell-out. No Python. No C. Pure Prolog binary I/O.
%%
%% Usage:
%%   ?- gguf_read(Path, Header, Metadata, TensorInfos).
%%   ?- gguf_architecture_native(Path, Arch).
%%   ?- gguf_tensor_list(Path, Tensors).

:- module(gguf_native_reader, [
    gguf_read/4,
    gguf_architecture_native/2,
    gguf_tensor_list/2
]).

:- use_module('lib/safe_read').

%% ═══════════════════════════════════════════════════════════════
%% Top-level reader
%% ═══════════════════════════════════════════════════════════════

gguf_read(Path, header(Version, TensorCount, KVCount), Metadata, TensorInfos) :-
    safe_open(Path, H0),
    catch(
        gguf_read_inner(H0, Version, TensorCount, KVCount, Metadata, TensorInfos, HN),
        Error,
        (safe_close(H0), throw(Error))
    ),
    safe_close(HN).

gguf_read_inner(H0, Version, TensorCount, KVCount, Metadata, TensorInfos, H6) :-
    %% Magic: "GGUF" = 0x46554747 little-endian
    safe_read_uint32_le(H0, Magic, H1),
    (Magic =:= 0x46554747 -> true ; throw(error(bad_gguf_magic(Magic), _))),
    safe_read_uint32_le(H1, Version, H2),
    safe_read_uint64_le(H2, TensorCount, H3),
    safe_read_uint64_le(H3, KVCount, H4),
    %% Metadata KV pairs
    read_kv_pairs(H4, KVCount, Metadata, H5),
    %% Tensor info entries
    read_tensor_infos(H5, TensorCount, TensorInfos, H6).

%% ═══════════════════════════════════════════════════════════════
%% KV pair reader
%% ═══════════════════════════════════════════════════════════════

read_kv_pairs(H, 0, [], H) :- !.
read_kv_pairs(H0, N, [Key-Value | Rest], HN) :-
    N > 0,
    safe_read_string(H0, Key, H1),
    safe_read_uint32_le(H1, Type, H2),
    read_kv_value(Type, H2, Value, H3),
    N1 is N - 1,
    read_kv_pairs(H3, N1, Rest, HN).

%% Type dispatch (GGUF spec: 13 value types)
read_kv_value(0, H0, Value, H1) :- safe_read_uint8(H0, Value, H1).      % uint8
read_kv_value(1, H0, Value, H1) :- safe_read_int8(H0, Value, H1).       % int8
read_kv_value(2, H0, Value, H1) :- safe_read_uint16_le(H0, Value, H1).  % uint16
read_kv_value(3, H0, Value, H1) :- safe_read_uint16_le(H0, V, H1),      % int16
    (V > 32767 -> Value is V - 65536 ; Value = V).
read_kv_value(4, H0, Value, H1) :- safe_read_uint32_le(H0, Value, H1).  % uint32
read_kv_value(5, H0, Value, H1) :- safe_read_int32_le(H0, Value, H1).   % int32
read_kv_value(6, H0, Value, H1) :- safe_read_float32_le(H0, Value, H1). % float32
read_kv_value(7, H0, Value, H1) :- safe_read_bool(H0, Value, H1).       % bool
read_kv_value(8, H0, Value, H1) :- safe_read_string(H0, Value, H1).     % string
read_kv_value(9, H0, array(ElemType, Values), HN) :-                     % array
    safe_read_uint32_le(H0, ElemType, H1),
    safe_read_uint64_le(H1, Count, H2),
    read_array_elements(ElemType, Count, H2, Values, HN).
read_kv_value(10, H0, Value, H1) :- safe_read_uint64_le(H0, Value, H1). % uint64
read_kv_value(11, H0, Value, H1) :-                                      % int64
    safe_read_uint64_le(H0, V, H1),
    (V > 9223372036854775807 -> Value is V - 18446744073709551616 ; Value = V).
read_kv_value(12, H0, Value, H1) :-                                      % float64
    safe_read_bytes(H0, 8, _Bytes, H1), Value = 0.0. % TODO: decode f64
read_kv_value(Type, H, unknown_type(Type), H).

read_array_elements(_, 0, H, [], H) :- !.
read_array_elements(Type, N, H0, [V|Rest], HN) :-
    N > 0,
    read_kv_value(Type, H0, V, H1),
    N1 is N - 1,
    read_array_elements(Type, N1, H1, Rest, HN).

%% ═══════════════════════════════════════════════════════════════
%% Tensor info reader
%% ═══════════════════════════════════════════════════════════════

read_tensor_infos(H, 0, [], H) :- !.
read_tensor_infos(H0, N, [tensor_info(Name, Dims, Type, Offset) | Rest], HN) :-
    N > 0,
    safe_read_string(H0, Name, H1),
    safe_read_uint32_le(H1, NDims, H2),
    read_dims(H2, NDims, Dims, H3),
    safe_read_uint32_le(H3, Type, H4),
    safe_read_uint64_le(H4, Offset, H5),
    N1 is N - 1,
    read_tensor_infos(H5, N1, Rest, HN).

read_dims(H, 0, [], H) :- !.
read_dims(H0, N, [D|Rest], HN) :-
    N > 0,
    safe_read_uint64_le(H0, D, H1),
    N1 is N - 1,
    read_dims(H1, N1, Rest, HN).

%% ═══════════════════════════════════════════════════════════════
%% Convenience predicates
%% ═══════════════════════════════════════════════════════════════

%% Extract architecture from GGUF metadata
gguf_architecture_native(Path, Arch) :-
    gguf_read(Path, _, Metadata, _),
    member('general.architecture'-ArchRaw, Metadata),
    %% Normalize: llama.cpp uses dashes, we use underscores
    atom_chars(ArchRaw, Chars),
    maplist([C, D]>>(C = '-' -> D = '_' ; D = C), Chars, NewChars),
    atom_chars(Arch, NewChars).

%% List all tensors with their shapes and types
gguf_tensor_list(Path, Tensors) :-
    gguf_read(Path, _, _, TensorInfos),
    maplist([tensor_info(N,D,T,O), tensor(N,D,T,O)]>>true, TensorInfos, Tensors).
