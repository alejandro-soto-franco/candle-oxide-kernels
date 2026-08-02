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
    /// One of the fused family's entries, whose parameter list follows from
    /// the kind named in [`Body::Fused`] rather than from a buffer count:
    /// the six kinds take three, four, four, four, five and five words.
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

/// One entry of the fused family.
///
/// What the family's entries share is that each takes its index from the
/// launch geometry rather than computing it. The grid is shaped like the
/// tensor, so every index term is a special register or the induction
/// variable and no entry divides. Three of the five also fuse a run of two
/// Candle kernels into one pass; the other two are a single traversal that
/// Candle already performs, run without `get_strided_index`.
///
/// The variant fixes the parameter list and the arithmetic together, since
/// for a fused entry the two are one choice. Every entry assumes contiguous
/// operands, which is what lets an operand index be an explicit expression
/// rather than a `num_dims`-deep walk.
///
/// See `fused.rs` for the geometry. `row` below is the element offset of the
/// block's row and `col` the column within it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FusedKind {
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
    /// `out[row + col] = x[row + col] + bias[col]`, with `row = ctaid.y * d`.
    ///
    /// [`FusedKind::BiasGeluErf`] without the activation. It replaces the
    /// last-dimension broadcast add, which is every remaining `badd_f32` in
    /// the profile: three projection biases a layer and two embedding adds.
    /// Candle routes a broadcast down its strided path, since a broadcast
    /// carries a zero stride, so the kernel this displaces walks
    /// `num_dims` deep for an index this one reads from the grid.
    Bias,
    /// `out[((b * D2 + j) * D1 + i) * d + k] = src[((b * D1 + i) * D2 + j) * d + k]`,
    /// a rank-4 copy under a swap of dimensions 1 and 2, with `b = ctaid.z`,
    /// `i = ctaid.y`, `j = ctaid.x`, `D1 = nctaid.y` and `D2 = nctaid.x`.
    ///
    /// Both head transposes in the attention path are this shape, one each
    /// way. `ctaid.x` names the dimension being permuted, so it cannot also
    /// index the column: `k` runs from `tid.x` alone, stepping by `ntid.x`.
    ///
    /// This is the one kind whose read and write rows differ, which is what
    /// the permutation is.
    ///
    /// One row a block leaves `min(round_up_32(d), 256)` threads in it, 64
    /// at this model's head size, and stacking `rows` of them along `y` to
    /// fill the block back up has been tried and is slower. Measured against
    /// the committed entry on an RTX 5060 Laptop, medians of nine
    /// interleaved runs of 100 launches at the model's own 8.9M elements and
    /// at four times that: a block of `(64, 2, 1)` gives up 0.2 to 8.5 per
    /// cent, `(64, 4, 1)` 3.9 to 16.2, and `(64, 8, 1)` 8.7 to 27.6. The
    /// degradation is monotonic in the depth and holds at both transposes.
    /// Deepening the block multiplies the scatter the permutation already
    /// has on one side, since consecutive `tid.y` values address rows far
    /// apart in both operands.
    ///
    /// The entry needs no widening anyway: in isolation it runs at 316 to
    /// 329 GB/s where a straight `cuMemcpyDtoD` of the same bytes reaches
    /// 319.6, so it is already at this device's copy bandwidth. A lower
    /// figure seen inside a full forward pass comes from the surrounding
    /// context rather than from the launch geometry.
    Permute12,
    /// [`FusedKind::ScaleMask`] with Candle's softmax folded onto it, so the
    /// score tensor is never written between the two.
    ///
    /// Grid `(S, H, B)`, block `(32, 1, 1)`, with `s = ctaid.x`,
    /// `h = ctaid.y`, `b = ctaid.z`, `S = nctaid.x` and `H = nctaid.y`. The
    /// column count is `S`, so it comes from the grid too and this kind
    /// takes no word-sized parameter at all.
    ///
    /// ```text
    /// row  = ((b * H + h) * S + s) * S
    /// mrow = (b * S + s) * S
    /// t(col) = (scores[row + col] * scale + 0.0f) + mask[mrow + col]
    /// ```
    ///
    /// `t(col)` stands where `softmax_f32` reads `x[row * ncols + col]`, in
    /// the maximum pass and again in the exponential pass. Recomputing it is
    /// the point: keeping it is what costs the round trip through memory.
    ///
    /// The reduction is Candle's, term for term
    /// (`candle-kernels/src/reduce.cu:233-272`): a warp-wide butterfly over
    /// lanes 16, 8, 4, 2 and 1 for the maximum, the same again for the sum,
    /// and `1. / tmp` evaluated as a double reciprocal. It runs inside one
    /// warp, so it needs no shared memory and no barrier. This is the one
    /// kind whose result depends on the order of a floating-point reduction.
    ScaleMaskSoftmax,
}

impl FusedKind {
    /// Every libdevice symbol this entry calls, in the order it calls them.
    ///
    /// `__nv_normcdff` is what Candle reaches through `normcdfg`
    /// (`cuda_utils.cuh:140`) and `gelu_erf_fwd` (`unary.cu:32-33`);
    /// `__nv_fmaxf` and `__nv_expf` are what `maxg` and `expg` reach at f32
    /// (`cuda_utils.cuh:142,148`). Any other formulation loses
    /// bit-exactness against the kernel the entry replaces.
    pub fn libdevice_names(self) -> &'static [&'static str] {
        match self {
            FusedKind::BiasGeluErf => &["__nv_normcdff"],
            FusedKind::ScaleMaskSoftmax => &["__nv_fmaxf", "__nv_expf"],
            FusedKind::ScaleMask
            | FusedKind::BiasResidual
            | FusedKind::Bias
            | FusedKind::Permute12 => &[],
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
    /// A fused family entry, whose loads, index arithmetic and stores are
    /// one expression rather than a function of already-loaded operands.
    /// `fused.rs` emits it whole, so `emit_body` never sees this.
    Fused(FusedKind),
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

    /// The kind this entry computes, for a fused entry.
    pub fn kind(&self) -> Option<FusedKind> {
        match self.body {
            Body::Fused(kind) => Some(kind),
            _ => None,
        }
    }

    /// Every libdevice symbol this entry calls, at its own dtype.
    ///
    /// A fused entry may call more than one, so this is what decides whether
    /// a module links against libdevice. [`OpDesc::libdevice_name`] answers
    /// the narrower question `emit_body` asks, which is which single symbol
    /// a [`Body::Libdevice`] entry is.
    pub fn libdevice_symbols(&self) -> Vec<&'static str> {
        match self.body {
            Body::Fused(kind) => kind.libdevice_names().to_vec(),
            _ => self.libdevice_name().into_iter().collect(),
        }
    }

    /// The single libdevice symbol a [`Body::Libdevice`] entry calls, at its
    /// own dtype. `None` for every other body, the fused family included:
    /// a fused entry's symbols come from [`OpDesc::libdevice_symbols`].
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

fn fused(stem: &str, kind: FusedKind) -> OpDesc {
    OpDesc {
        name: format!("{stem}_{}", DType::F32.suffix()),
        dtype: DType::F32,
        arity: Arity::Fused,
        out: OutType::Same,
        body: Body::Fused(kind),
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

/// Every fused family entry.
///
/// f32 only. The entries exist for one model's f32 forward pass, and an f64
/// form would ship five entries nothing calls.
pub fn fused_ops() -> Vec<OpDesc> {
    vec![
        fused("fused_scale_mask", FusedKind::ScaleMask),
        fused("fused_bias_gelu_erf", FusedKind::BiasGeluErf),
        fused("fused_bias_residual", FusedKind::BiasResidual),
        fused("fused_bias", FusedKind::Bias),
        fused("fused_permute12", FusedKind::Permute12),
        fused("fused_scale_mask_softmax", FusedKind::ScaleMaskSoftmax),
    ]
}

/// Every entry in every family.
pub fn all_ops() -> Vec<OpDesc> {
    let mut ops = binary_ops();
    ops.extend(unary_ops());
    ops.extend(fused_ops());
    ops
}
