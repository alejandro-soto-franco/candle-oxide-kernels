use candle_oxide_codegen::desc::{all_ops, Arity, Family, OutType};

#[test]
fn the_table_covers_the_v1_surface() {
    let ops = all_ops();
    let names: Vec<&str> = ops.iter().map(|op| op.name.as_str()).collect();

    // 6 binary arithmetic ops x 2 dtypes.
    for stem in ["badd", "bsub", "bmul", "bdiv", "bmaximum", "bminimum"] {
        for suffix in ["f32", "f64"] {
            let want = format!("{stem}_{suffix}");
            assert!(names.contains(&want.as_str()), "missing {want}");
        }
    }
    // 6 comparisons x 2 dtypes, all writing u8.
    for stem in ["eq", "ne", "lt", "le", "gt", "ge"] {
        for suffix in ["f32", "f64"] {
            let want = format!("{stem}_{suffix}");
            let op = ops.iter().find(|op| op.name == want).unwrap_or_else(|| panic!("missing {want}"));
            assert_eq!(op.out, OutType::U8, "{want} writes uint8_t");
            assert_eq!(op.arity, Arity::Binary);
        }
    }
    // 11 unary ops x 2 dtypes.
    for stem in ["ucopy", "uneg", "usqr", "urecip", "uabs", "usqrt", "uexp", "ulog", "usin", "ucos", "utanh"] {
        for suffix in ["f32", "f64"] {
            let want = format!("{stem}_{suffix}");
            assert!(names.contains(&want.as_str()), "missing {want}");
        }
    }
    assert_eq!(ops.len(), 46, "12 binary + 12 comparison + 22 unary");
}

#[test]
fn every_name_is_unique() {
    let mut names: Vec<String> = all_ops().into_iter().map(|op| op.name).collect();
    names.sort();
    let before = names.len();
    names.dedup();
    assert_eq!(names.len(), before, "entry-point names collide inside a module");
}

#[test]
fn module_names_are_namespaced() {
    assert_eq!(Family::Binary.module_name(), "oxide::binary");
    assert_eq!(Family::Unary.module_name(), "oxide::unary");
}
