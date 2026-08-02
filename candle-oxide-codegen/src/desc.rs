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

/// The parameter-list shape an entry declares, and with it the builder that
/// emits it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Arity {
    Unary,
    Binary,
    /// One of the fused chains, whose parameter list follows from the chain
    /// named in [`Body::Fused`] rather than from a buffer count: the three
    /// chains take five, six and seven words.
    Fused,
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

/// One of the three fused chains, each replacing a run of two Candle
/// elementwise kernels with a single pass.
///
/// The variant fixes the parameter list and the arithmetic together, since
/// for a fused entry the two are one choice. Every chain assumes contiguous
/// operands, which is what lets an operand index be an explicit expression
/// rather than a `num_dims`-deep walk.
///
/// The launch is shaped like the tensor: `ctaid.x` walks the last dimension
/// and the `y` and `z` dimensions carry the row, so every index term is a
/// special register and no chain divides. See `fused.rs`. `row` below is the
/// element offset of the block's row, and `col` the column within it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FusedChain {
    /// `out[row + col] = (x[row + col] * scale + 0.0f) + mask[b * ss + col]`,
    /// over `x` of shape `[B, H, S, S]` and `mask` of shape `[B, 1, S, S]`,
    /// with `ss = S*S`, `b = ctaid.z` and `row = (b * nctaid.y + ctaid.y) * ss`.
    ///
    /// The head count arrives as `nctaid.y`, so the kernel takes no `hss`.
    ///
    /// The `+ 0.0f` is `affine_f32`'s `add` operand, which Candle reaches
    /// through `Tensor::affine(scale, 0.0)`. It changes one value: a product
    /// of `-0.0` becomes `+0.0`. Omitting it disagrees with the reference
    /// whenever the mask element is `-0.0` as well.
    ScaleMask,
    /// `t = x[row + col] + bias[col]; out[row + col] = t * normcdf(t)`, with
    /// `row = ctaid.y * d`.
    BiasGeluErf,
    /// `out[row + col] = (x[row + col] + bias[col]) + residual[row + col]`,
    /// with `row = ctaid.y * d`. The parenthesisation is the order Candle
    /// evaluates, `Linear::forward` adding the bias before the caller adds
    /// the residual, and reassociating changes the result.
    BiasResidual,
}

impl FusedChain {
    /// The libdevice symbol this chain calls, if it calls one.
    ///
    /// `__nv_normcdff` is what Candle reaches through `normcdfg`
    /// (`cuda_utils.cuh:140`) and `gelu_erf_fwd` (`unary.cu:32-33`). Any
    /// other formulation loses bit-exactness against the kernel this chain
    /// replaces.
    pub fn libdevice_name(self) -> Option<&'static str> {
        match self {
            FusedChain::BiasGeluErf => Some("__nv_normcdff"),
            FusedChain::ScaleMask | FusedChain::BiasResidual => None,
        }
    }
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
    /// A fused chain, whose loads, index arithmetic and stores are one
    /// expression rather than a function of already-loaded operands.
    /// `fused.rs` emits it whole, so `emit_body` never sees this.
    Fused(FusedChain),
}

/// Which PTX artifact an entry is packaged into.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    Binary,
    Unary,
    Fused,
}

impl Family {
    pub fn artifact(self) -> &'static str {
        match self {
            Family::Binary => "binary.ptx",
            Family::Unary => "unary.ptx",
            Family::Fused => "fused.ptx",
        }
    }

    /// Whether this family compiles with FMA contraction on.
    ///
    /// The elementwise families match nvcc's default, so their parity
    /// comparisons are against like. The fused family switches it off for
    /// `fused_scale_mask_f32`: contraction turns `x * scale + mask` into one
    /// FMA and so into one rounding, while the two kernels it replaces round
    /// the multiply to memory and add the mask separately. The more accurate
    /// answer is the wrong one here. The other two chains carry no
    /// contraction site, so the setting costs them nothing.
    pub fn fma_contraction(self) -> bool {
        match self {
            Family::Binary | Family::Unary => true,
            Family::Fused => false,
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
            Arity::Fused => Family::Fused,
        }
    }

    /// The chain this entry computes, for a fused entry.
    pub fn chain(&self) -> Option<FusedChain> {
        match self.body {
            Body::Fused(chain) => Some(chain),
            _ => None,
        }
    }

    /// The libdevice symbol this entry calls, at its own dtype.
    pub fn libdevice_name(&self) -> Option<&'static str> {
        let f32_name = match self.body {
            Body::Libdevice(name) => name,
            // The fused chains are f32 only, so the chain names its symbol
            // directly and the dtype table below is bypassed.
            Body::Fused(chain) => return chain.libdevice_name(),
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

fn fused(stem: &str, chain: FusedChain) -> OpDesc {
    OpDesc {
        name: format!("{stem}_{}", DType::F32.suffix()),
        dtype: DType::F32,
        arity: Arity::Fused,
        out: OutType::Same,
        body: Body::Fused(chain),
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

/// Every fused chain entry.
///
/// f32 only. The chains exist for one model's f32 forward pass, and an f64
/// form would ship three entries nothing calls.
pub fn fused_ops() -> Vec<OpDesc> {
    vec![
        fused("fused_scale_mask", FusedChain::ScaleMask),
        fused("fused_bias_gelu_erf", FusedChain::BiasGeluErf),
        fused("fused_bias_residual", FusedChain::BiasResidual),
    ]
}

/// Every entry in every family.
pub fn all_ops() -> Vec<OpDesc> {
    let mut ops = binary_ops();
    ops.extend(unary_ops());
    ops.extend(fused_ops());
    ops
}
