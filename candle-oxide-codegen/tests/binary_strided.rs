use candle_oxide_codegen::build_family;
use candle_oxide_codegen::desc::{binary_ops, Family};
use candle_oxide_codegen::testing::{assert_ptxas_accepts, entry_body};

/// A SINGLE-entry module. Every assertion below that uses a bare
/// `ptx.contains` or a bare count is sound only because the module holds
/// one entry; widening this filter would silently turn those assertions
/// module-scoped rather than entry-scoped. `testing::entry_body` exists so
/// an assertion can be scoped explicitly instead of relying on the module
/// holding one entry.
fn badd_only() -> Vec<candle_oxide_codegen::desc::OpDesc> {
    binary_ops()
        .into_iter()
        .filter(|op| op.name == "badd_f32")
        .collect()
}

#[test]
fn each_strided_path_decomposes_its_own_index() {
    let ptx = build_family(Family::Binary, &badd_only(), "sm_80").unwrap();
    // Every strided operand walks the dimensions once, dividing the running
    // index by `dims[dim_idx]` per step. Counting those divisions counts the
    // strided operands across the four layout paths:
    //
    //   both contiguous       0
    //   lhs contiguous only   1  (the rhs is decomposed)
    //   rhs contiguous only   1  (the lhs is decomposed)
    //   neither contiguous    2  (both are decomposed)
    //
    // The remainder never appears as `rem.u32`: LLVM strength-reduces
    // `idx % dim` into this same `div.u32` followed by `mul.lo.s32` and
    // `sub.s32`, so the division is the site that survives.
    //
    // Verified by mutation: forcing all four paths contiguous drops the count
    // to 0. The count is invariant under swapping the two single-strided
    // paths, so it does not pin which operand each path decomposes; that
    // pairing is covered by the GPU parity comparison.
    let divisions = ptx.matches("div.u32").count();
    assert_eq!(
        divisions, 4,
        "expected one index decomposition per strided operand across the four \
         layout paths, saw {divisions}:\n{ptx}"
    );
}

#[test]
fn the_null_strides_pointer_is_tested() {
    let ptx = build_family(Family::Binary, &badd_only(), "sm_80").unwrap();
    let body = entry_body(&ptx, "badd_f32");
    // `dims_and_strides == nullptr` selects the contiguous path without
    // dereferencing, so the very first predicate in the entry must be an
    // equality test of the strides pointer against the literal 0, before
    // anything is dereferenced. `is_contiguous`'s own decomposition loop
    // emits `setp.ne` comparisons of its own (a non-zero remainder check),
    // so accepting any `setp.ne.s64` in the whole body passes even with
    // this null check deleted; requiring the first predicate specifically
    // does not.
    let first_setp = body
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("setp."));
    // LLVM may also compare against the null pointer converted to the global
    // window (`mov.b64 %rdN, 0; cvta.to.global.u64 %rdM, %rdN`), which is the
    // address 0 there.
    let is_global_null = |reg: &str| {
        let lines: Vec<&str> = body.lines().map(str::trim).collect();
        lines.iter().any(|line| {
            let Some(rest) = line.strip_prefix("cvta.to.global.u64") else {
                return false;
            };
            let mut regs = rest.trim_end_matches(';').split(',').map(str::trim);
            regs.next() == Some(reg)
                && regs.next().is_some_and(|src| {
                    lines.contains(&format!("mov.b64 \t{src}, 0;").as_str())
                        || lines.contains(&format!("mov.b64 {src}, 0;").as_str())
                })
        })
    };
    match first_setp {
        Some(line) => {
            let rhs = line
                .trim_end_matches(';')
                .rsplit(',')
                .next()
                .unwrap()
                .trim();
            assert!(
                line.starts_with("setp.eq.b64") && (rhs == "0" || is_global_null(rhs)),
                "the first predicate in the entry must be `dims_and_strides == 0`, \
                 saw `{line}`:\n{body}"
            )
        }
        None => panic!("no predicate found before any load:\n{body}"),
    }
}

#[test]
fn the_strided_module_still_compiles_and_assembles() {
    let ops = binary_ops();
    let ptx = build_family(Family::Binary, &ops, "sm_80").unwrap();
    assert_ptxas_accepts(ptx.as_bytes(), "binary_strided_all", "sm_80");
}
