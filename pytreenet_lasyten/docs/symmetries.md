# U1 and U1xU1

U1 stores one additive particle-number charge. U1xU1 stores `(N_up,N_down)` and
requires conservation of each spin population. The HK model and spin-conserving
bath coupling used here satisfy that requirement. Storage is bosonic with fermionic statistics
included via JW strings. Symmetric tensor operations and Lasyten adaptor can be found in util.lasyten_backend

Construction lives in `util.hk_u1_model`. `abelian_charges.fixed_charge` returns
the full charge tuple. `check_native_u1` checks every component and returns the
sum for compatibility with older callers. TTNO construction dispatches on the
reference tree's charge count. Excitations preserve the spin-resolved changes.

The U1xU1 initializer propagates QR's actual reduced/reordered parent space
before generating the parent tensor, avoiding the old node4 allocation error.
Run the compiled initializer grid when validating a new native backend version.

Bond caps are totals across sectors. Additional symmetry can reduce storage
without reducing wall time. Compare equal physical states/tolerances, current
bond dimensions, fidelity, norm and phase in performance studies.
