:- protocol(tensor_loader_adapterp).
    :- public(load_tensor_from_manifest/3).
    :- public(load_tensor_from_manifest/4).
    :- public(tensor_element_count/2).
    :- public(tensor_dtype_string/2).
    :- public(tensor_shape_list/2).
:- end_protocol.

:- object(tensor_loader_adapter,
    implements(tensor_loader_adapterp)).



:- use_module(library(janus)).

%% ────────────────────────────────────────────────────────────────────
%% load_tensor_from_manifest(+ModelPath, +TensorFact, +Mode, -Tensor)
%% ────────────────────────────────────────────────────────────────────
%%
%% Given a tensor/9 fact from the GGUF manifest and the model file path,
%% load the tensor's data and return a torch.Tensor handle.
%%
%% ModelPath: absolute path to the GGUF file
%% TensorFact: a fact of shape:
%%     tensor(name(NameStr), dimensions(Dims), type_code(_),
%%            type_name(TypeAtom), numpy_dtype(DtypeAtom),
%%            file_offset(Offset), byte_size(_),
%%            relative_offset(_))
%% Mode: 'fp32' (use f32 directly) or 'fp32_from_f16' (upcast F16→F32)
%% Tensor: the resulting torch.Tensor handle (single Python object)
%%
%% For P4 hardware: F16 tensors should be upcast to F32 since Pascal
%% doesn't accelerate F16 arithmetic.

load_tensor_from_manifest(ModelPath, TensorFact, fp32, Tensor) :-
    TensorFact = tensor(name(_), dimensions(Dims), type_code(_),
                        type_name(_), numpy_dtype(DtypeAtom),
                        file_offset(Offset), byte_size(_),
                        relative_offset(_)),
    tensor_element_count(Dims, Count),
    tensor_dtype_string(DtypeAtom, DtypeStr),
    tensor_shape_list(Dims, ShapeList),
    py_call(llamatov_loader:load_tensor(ModelPath, Offset, Count,
                                          DtypeStr, ShapeList),
            Tensor).

load_tensor_from_manifest(ModelPath, TensorFact, fp32_from_f16, Tensor) :-
    %% Only valid if the tensor is actually F16
    TensorFact = tensor(name(_), dimensions(Dims), type_code(_),
                        type_name('F16'), numpy_dtype(_),
                        file_offset(Offset), byte_size(_),
                        relative_offset(_)),
    tensor_element_count(Dims, Count),
    tensor_shape_list(Dims, ShapeList),
    py_call(llamatov_loader:load_tensor_f16(ModelPath, Offset, Count,
                                              ShapeList),
            Tensor).

%% load_tensor_from_manifest/4 — defaults to fp32 mode
load_tensor_from_manifest(ModelPath, TensorFact, Tensor) :-
    TensorFact = tensor(name(_), _, _, type_name(TypeName), _, _, _, _),
    ( TypeName == 'F16' -> Mode = fp32_from_f16 ; Mode = fp32 ),
    load_tensor_from_manifest(ModelPath, TensorFact, Mode, Tensor).

%% ────────────────────────────────────────────────────────────────────
%% Helpers: element count, dtype string, shape list
%% ────────────────────────────────────────────────────────────────────

%% tensor_element_count(+Dims, -Count)
%%   The product of all dimension sizes (e.g., [768, 2304] → 1769472).
tensor_element_count([], 1).
tensor_element_count([D | Rest], Count) :-
    tensor_element_count(Rest, RestCount),
    Count is D * RestCount.

%% tensor_dtype_string(+DtypeAtom, -DtypeStr)
%%   Convert numpy_dtype atom to the string form numpy.fromfile expects.
tensor_dtype_string(float32, 'float32').
tensor_dtype_string(float16, 'float16').
tensor_dtype_string(uint8,   'uint8').
tensor_dtype_string(int8,    'int8').
tensor_dtype_string(int32,   'int32').

%% tensor_shape_list(+Dims, -ShapeList)
%%   The shape passed to numpy.reshape is a list (numpy accepts list).
%%   The dimensions list from the manifest is already the right format.
tensor_shape_list(Dims, Dims).

:- end_object.
