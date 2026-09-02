:- protocol(pcie_tracep).
    :- public(pcie_config/4).
    :- public(pcie_bandwidth/2).
    :- public(gpu_transfer_rate/3).
    :- public(pcie_bottleneck/1).
:- end_protocol.

:- object(pcie_trace,
    implements(pcie_tracep)).



%% pcie_config(Gen, Width, MaxGen, MaxWidth)
%% Query via: nvidia-smi --query-gpu=pcie.link.gen.current,...
pcie_config(Gen, Width, MaxGen, MaxWidth) :-
    % These would be populated by a shell call
    Gen = 1, Width = 8, MaxGen = 3, MaxWidth = 16.

%% Theoretical PCIe bandwidth in MB/s per direction
%% Gen1: 250 MB/s/lane, Gen2: 500, Gen3: ~1000, Gen4: ~2000, Gen5: ~4000
pcie_bandwidth(Gen, Width, BW_MBps) :-
    gen_rate(Gen, Rate),
    BW_MBps is Rate * Width.

gen_rate(1, 250).
gen_rate(2, 500).
gen_rate(3, 985).   % ~1 GB/s per lane after encoding overhead
gen_rate(4, 1969).
gen_rate(5, 3938).

%% Detect PCIe bottleneck
pcie_bottleneck(Reason) :-
    pcie_config(Gen, Width, MaxGen, MaxWidth),
    (Gen < MaxGen -> Reason = gen_downgrade(Gen, MaxGen)
    ; Width < MaxWidth -> Reason = width_downgrade(Width, MaxWidth)
    ; Reason = none).

:- end_object.
