//! The op description table. One entry per generated kernel entry point.

/// Element type of an entry's operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    F64,
}

impl DType {
    pub fn suffix(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
        }
    }
}

/// Number of input buffers an entry reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arity {
    Unary,
    Binary,
}

/// Element type of an entry's output buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutType {
    /// Same as the operands, the `BINARY_OP` and `UNARY_OP` case.
    Same,
    /// `uint8_t`, the `BINARY_OP_OUT` comparison case.
    U8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpKind {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// What an entry computes from its loaded operands.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Body {
    Add,
    Sub,
    Mul,
    Div,
    Cmp(CmpKind),
    /// `x`, the identity.
    Copy,
    Neg,
    /// `x * x`.
    Sqr,
    /// `1.0 / x`, computed in double for both dtypes, matching `recipg`
    /// at `candle-kernels/src/cuda_utils.cuh:120`.
    Recip,
    /// A libdevice call named directly, one argument per operand. The name
    /// is the f32 form; `libdevice_name` picks the f64 form by dtype.
    Libdevice(&'static str),
}

/// Which PTX artifact an entry is packaged into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    Binary,
    Unary,
}

impl Family {
    pub fn artifact(self) -> &'static str {
        match self {
            Family::Binary => "binary.ptx",
            Family::Unary => "unary.ptx",
        }
    }
}

/// One generated entry point.
#[derive(Clone, Debug)]
pub struct OpDesc {
    pub name: String,
    pub dtype: DType,
    pub arity: Arity,
    pub out: OutType,
    pub body: Body,
}

impl OpDesc {
    pub fn family(&self) -> Family {
        match self.arity {
            Arity::Unary => Family::Unary,
            Arity::Binary => Family::Binary,
        }
    }

    /// The libdevice symbol this entry calls, at its own dtype.
    pub fn libdevice_name(&self) -> Option<&'static str> {
        let f32_name = match self.body {
            Body::Libdevice(name) => name,
            _ => return None,
        };
        Some(match (f32_name, self.dtype) {
            (name, DType::F32) => name,
            ("__nv_sqrtf", DType::F64) => "__nv_sqrt",
            ("__nv_expf", DType::F64) => "__nv_exp",
            ("__nv_logf", DType::F64) => "__nv_log",
            ("__nv_sinf", DType::F64) => "__nv_sin",
            ("__nv_cosf", DType::F64) => "__nv_cos",
            ("__nv_tanhf", DType::F64) => "__nv_tanh",
            ("__nv_fabsf", DType::F64) => "__nv_fabs",
            ("__nv_fmaxf", DType::F64) => "__nv_fmax",
            ("__nv_fminf", DType::F64) => "__nv_fmin",
            (other, DType::F64) => panic!("no f64 libdevice name recorded for {other}"),
        })
    }
}

fn binary(stem: &str, dtype: DType, body: Body) -> OpDesc {
    OpDesc {
        name: format!("{stem}_{}", dtype.suffix()),
        dtype,
        arity: Arity::Binary,
        out: OutType::Same,
        body,
    }
}

fn compare(stem: &str, dtype: DType, kind: CmpKind) -> OpDesc {
    OpDesc {
        name: format!("{stem}_{}", dtype.suffix()),
        dtype,
        arity: Arity::Binary,
        out: OutType::U8,
        body: Body::Cmp(kind),
    }
}

fn unary(stem: &str, dtype: DType, body: Body) -> OpDesc {
    OpDesc {
        name: format!("{stem}_{}", dtype.suffix()),
        dtype,
        arity: Arity::Unary,
        out: OutType::Same,
        body,
    }
}

/// Every binary entry, arithmetic then comparison.
pub fn binary_ops() -> Vec<OpDesc> {
    let mut ops = Vec::new();
    for dtype in [DType::F32, DType::F64] {
        ops.push(binary("badd", dtype, Body::Add));
        ops.push(binary("bsub", dtype, Body::Sub));
        ops.push(binary("bmul", dtype, Body::Mul));
        ops.push(binary("bdiv", dtype, Body::Div));
        ops.push(binary("bmaximum", dtype, Body::Libdevice("__nv_fmaxf")));
        ops.push(binary("bminimum", dtype, Body::Libdevice("__nv_fminf")));
        ops.push(compare("eq", dtype, CmpKind::Eq));
        ops.push(compare("ne", dtype, CmpKind::Ne));
        ops.push(compare("lt", dtype, CmpKind::Lt));
        ops.push(compare("le", dtype, CmpKind::Le));
        ops.push(compare("gt", dtype, CmpKind::Gt));
        ops.push(compare("ge", dtype, CmpKind::Ge));
    }
    ops
}

/// Every unary entry.
pub fn unary_ops() -> Vec<OpDesc> {
    let mut ops = Vec::new();
    for dtype in [DType::F32, DType::F64] {
        ops.push(unary("ucopy", dtype, Body::Copy));
        ops.push(unary("uneg", dtype, Body::Neg));
        ops.push(unary("usqr", dtype, Body::Sqr));
        ops.push(unary("urecip", dtype, Body::Recip));
        ops.push(unary("uabs", dtype, Body::Libdevice("__nv_fabsf")));
        ops.push(unary("usqrt", dtype, Body::Libdevice("__nv_sqrtf")));
        ops.push(unary("uexp", dtype, Body::Libdevice("__nv_expf")));
        ops.push(unary("ulog", dtype, Body::Libdevice("__nv_logf")));
        ops.push(unary("usin", dtype, Body::Libdevice("__nv_sinf")));
        ops.push(unary("ucos", dtype, Body::Libdevice("__nv_cosf")));
        ops.push(unary("utanh", dtype, Body::Libdevice("__nv_tanhf")));
    }
    ops
}

/// Every entry in both families.
pub fn all_ops() -> Vec<OpDesc> {
    let mut ops = binary_ops();
    ops.extend(unary_ops());
    ops
}
