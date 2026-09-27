//! Row-exact 4-bit affine matvecs, ported from TensorFold's `rows.py` (MIT, see NOTICE).
//! Every input row is its own simdgroup running one row's loop, so a row's bits never
//! depend on how many rows share the call: serial decode and a verify window agree.
const std = @import("std");
const mlx = @import("mlx.zig");

pub const MAX_ROWS = 16;
/// Output rows a simdgroup.
const RPS = 4;

/// MLX's 4-bit `qmv_fast` loop (load_vector + qdot): a lane's 16 inputs are
/// pre-divided by 1, 16, 256, 4096 so masked nibbles need no shift, and their
/// sum adds each run of 4 in bf16 first, as MLX does on bfloat16_t. The tail
/// of a K that is not a multiple of 512 runs one more 16-input chunk on the
/// lanes below (K % 512) / 16. Reads nothing but its own row of x.
pub const HEADER =
    \\inline float rq_load16(const device bfloat* x, thread float* xt) {
    \\  float sum = 0.0f;
    \\  for (int i = 0; i < 16; i += 4) {
    \\    const bfloat a = x[i], b = x[i + 1], c = x[i + 2], d = x[i + 3];
    \\    sum += float(bfloat(float(bfloat(float(bfloat(float(a) + float(b))) + float(c))) + float(d)));
    \\    xt[i] = float(a); xt[i + 1] = float(b) / 16.0f; xt[i + 2] = float(c) / 256.0f; xt[i + 3] = float(d) / 4096.0f;
    \\  }
    \\  return sum;
    \\}
    \\inline float rq_qdot16(const device uint8_t* w, const thread float* xt, float scale, float bias, float sum) {
    \\  const device uint16_t* ws = (const device uint16_t*)w;
    \\  float accum = 0.0f;
    \\  for (int i = 0; i < 4; i++)
    \\    accum += (xt[4 * i] * (ws[i] & 0x000f) + xt[4 * i + 1] * (ws[i] & 0x00f0) +
    \\              xt[4 * i + 2] * (ws[i] & 0x0f00) + xt[4 * i + 3] * (ws[i] & 0xf000));
    \\  return scale * accum + sum * bias;
    \\}
    \\template <int K, int GS, int RPS>
    \\inline void rq_rowdot(const device uint8_t* w, const device bfloat* sc, const device bfloat* bi,
    \\                      const device bfloat* x, uint lane, thread float* acc) {
    \\  constexpr int KB = K / 2;
    \\  constexpr int KG = K / GS;
    \\  constexpr int FULL = K / 512 * 512;
    \\  w += lane * 8;
    \\  sc += lane / (GS / 16);
    \\  bi += lane / (GS / 16);
    \\  x += lane * 16;
    \\  for (int j = 0; j < RPS; j++) acc[j] = 0.0f;
    \\  for (int k0 = 0; k0 < FULL; k0 += 512) {
    \\    float xt[16];
    \\    const float sum = rq_load16(x, xt);
    \\    for (int j = 0; j < RPS; j++) acc[j] += rq_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    \\    w += 256; sc += 512 / GS; bi += 512 / GS; x += 512;
    \\  }
    \\  if (FULL < K && int(lane) < (K - FULL) / 16) {
    \\    float xt[16];
    \\    const float sum = rq_load16(x, xt);
    \\    for (int j = 0; j < RPS; j++) acc[j] += rq_qdot16(w + j * KB, xt, float(sc[j * KG]), float(bi[j * KG]), sum);
    \\  }
    \\  for (int j = 0; j < RPS; j++) acc[j] = simd_sum(acc[j]);
    \\}
;

/// One simdgroup per input row, each over BLK blocks of RPS output rows; the
/// row count is a launch dimension only.
const QMV_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int r = int(thread_position_in_threadgroup.x) / 32;
    \\const int row0 = int(thread_position_in_grid.y) * RPS;
    \\float acc[RPS];
    \\rq_rowdot<K, GS, RPS>((const device uint8_t*)w + size_t(row0) * (K / 2), scales + size_t(row0) * (K / GS),
    \\                      biases + size_t(row0) * (K / GS), x + size_t(r) * K, lane, acc);
    \\if (lane == 0)
    \\  for (int j = 0; j < RPS; j++) {
    \\    if (RELU2) {
    \\      const float h = metal::max(float(bfloat(acc[j])), 0.0f);
    \\      y[size_t(r) * N + row0 + j] = bfloat(h * h);
    \\    } else {
    \\      y[size_t(r) * N + row0 + j] = bfloat(acc[j]);
    \\    }
    \\  }
;

var qmv_kernel: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { rows: c_int, n: c_int, k: c_int, gs: u32, relu2: bool };
var cfg_cache: std.AutoHashMapUnmanaged(CfgKey, mlx.mlx_fast_metal_kernel_config) = .{};

pub fn newKernel(name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn qmvConfig(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    if (cfg_cache.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ key.rows, key.n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, .bfloat16));
    const blocks: c_int = if (key.rows <= 8) 2 else 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * key.rows, @divExact(key.n, RPS), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * key.rows, blocks, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "N", key.n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RPS", RPS));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "RELU2", key.relu2));
    try cfg_cache.put(std.heap.c_allocator, key, config);
    return config;
}

/// A 4-bit affine bf16 matrix [..., N, K/8] the kernels read: groups of 32, 64
/// or 128, K a multiple of 64, N a multiple of 8.
pub fn fits(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) bool {
    if (bits != 4 or bi.ctx == null) return false;
    if (group_size != 32 and group_size != 64 and group_size != 128) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    if (ws.len < 2) return false;
    return @rem(ws[ws.len - 1] * 8, 64) == 0 and @rem(ws[ws.len - 2], 8) == 0;
}

/// `x [..., K] @ w.T` for 1..MAX_ROWS rows, or null outside the kernel.
pub fn qmv(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    return qmvAct(x, w, sc, bi, bits, group_size, false, s);
}

/// `qmv` with mlx-lm's relu2 on the bf16 output (`relu2`): bf16(max(bf16(acc), 0)^2).
pub fn qmvAct(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, relu2: bool, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!fits(w, sc, bi, bits, group_size) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const ws = mlx.getShape(w);
    if (ws.len != 2) return null;
    const xs = mlx.getShape(x);
    if (xs.len == 0 or xs.len > 8) return null;
    const k = xs[xs.len - 1];
    if (k != ws[1] * 8) return null;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;
    const n = ws[0];

    if (qmv_kernel == null) qmv_kernel = try newKernel("msv_rowqmv", &.{ "x", "w", "scales", "biases" }, &.{"y"}, QMV_SOURCE);
    const inputs = [_]mlx.mlx_array{ x, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, qmv_kernel.?, in_vec, try qmvConfig(.{ .rows = rows, .n = n, .k = k, .gs = group_size, .relu2 = relu2 }), s));
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, y, &out_shape, xs.len, s));
    return r;
}

/// Routed experts, one (row, slot) pair a threadgroup column: pair p is row
/// p / TOPK's slot p % TOPK and reads expert IDS[p]; simdgroup g of
/// threadgroup (b, p) takes the expert's output rows RPS (SG b + g) ..
/// fc1 applies mlx-lm's relu2 on its bf16 output: bf16(max(bf16(sum), 0)^2).
const EXPERT_UP_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int p = int(threadgroup_position_in_grid.z);
    \\const size_t e = size_t(ids[p]);
    \\const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
    \\const size_t at = e * N + size_t(row0);
    \\float acc[RPS];
    \\rq_rowdot<K, GS, RPS>((const device uint8_t*)w + at * (K / 2), scales + at * (K / GS), biases + at * (K / GS),
    \\                      x + size_t(p / TOPK) * K, lane, acc);
    \\if (lane == 0)
    \\  for (int j = 0; j < RPS; j++) {
    \\    const float h = metal::max(float(bfloat(acc[j])), 0.0f);
    \\    y[size_t(p) * N + row0 + j] = bfloat(h * h);
    \\  }
;

/// fc2 over each pair's own activation row.
const EXPERT_DOWN_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int p = int(threadgroup_position_in_grid.z);
    \\const size_t e = size_t(ids[p]);
    \\const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
    \\const size_t at = e * N + size_t(row0);
    \\float acc[RPS];
    \\rq_rowdot<K, GS, RPS>((const device uint8_t*)w + at * (K / 2), scales + at * (K / GS), biases + at * (K / GS),
    \\                      x + size_t(p) * K, lane, acc);
    \\if (lane == 0)
    \\  for (int j = 0; j < RPS; j++) y[size_t(p) * N + row0 + j] = bfloat(acc[j]);
;

const EXPERT_SG = 2;
var up_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_kernel: ?mlx.mlx_fast_metal_kernel = null;
const ExpertKey = struct { up: bool, pairs: c_int, topk: c_int, n: c_int, k: c_int, gs: u32 };
var expert_cfgs: std.AutoHashMapUnmanaged(ExpertKey, mlx.mlx_fast_metal_kernel_config) = .{};

fn expertConfig(key: ExpertKey) !mlx.mlx_fast_metal_kernel_config {
    if (expert_cfgs.get(key)) |c| return c;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{ key.pairs, key.n };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * EXPERT_SG, @divExact(key.n, RPS * EXPERT_SG), key.pairs));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * EXPERT_SG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "N", key.n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RPS", RPS));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SG", EXPERT_SG));
    if (key.up) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPK", key.topk));
    try expert_cfgs.put(std.heap.c_allocator, key, config);
    return config;
}

fn expertLaunch(up: bool, x: mlx.mlx_array, ids: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, pairs: c_int, topk: c_int, gs: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const kslot = if (up) &up_kernel else &down_kernel;
    if (kslot.* == null) kslot.* = try newKernel(if (up) "msv_rowqmv_expert_up" else "msv_rowqmv_expert_down", &.{ "x", "ids", "w", "scales", "biases" }, &.{"y"}, if (up) EXPERT_UP_SOURCE else EXPERT_DOWN_SOURCE);
    const ws = mlx.getShape(w);
    const inputs = [_]mlx.mlx_array{ x, ids, w, sc, bi };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    const key = ExpertKey{ .up = up, .pairs = pairs, .topk = topk, .n = ws[1], .k = ws[2] * 8, .gs = gs };
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kslot.*.?, in_vec, try expertConfig(key), s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

/// mlx-lm's SwitchMLP (fc1, relu2, fc2) on x [R, D] bf16 for the uint32
/// expert ids [R * TOPK] (row-major): [R * TOPK, D] bf16, or null outside the
/// kernel. Each (row, slot) pair is computed on its own, the same way at any R.
pub fn experts(x: mlx.mlx_array, ids: mlx.mlx_array, topk: c_int, fc1: [3]mlx.mlx_array, fc2: [3]mlx.mlx_array, gs1: u32, gs2: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!fits(fc1[0], fc1[1], fc1[2], 4, gs1) or !fits(fc2[0], fc2[1], fc2[2], 4, gs2)) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const xs = mlx.getShape(x);
    const w1 = mlx.getShape(fc1[0]);
    const w2 = mlx.getShape(fc2[0]);
    if (xs.len != 2 or w1.len != 3 or w2.len != 3) return null;
    const rows = xs[0];
    const pairs = rows * topk;
    if (rows < 1 or rows > MAX_ROWS or mlx.mlx_array_size(ids) != @as(usize, @intCast(pairs))) return null;
    const block = RPS * EXPERT_SG;
    if (xs[1] != w1[2] * 8 or w2[2] != @divTrunc(w1[1], 8) or @rem(w1[1], block) != 0 or @rem(w2[1], block) != 0) return null;
    const act = try expertLaunch(true, x, ids, fc1[0], fc1[1], fc1[2], pairs, topk, gs1, s);
    defer _ = mlx.mlx_array_free(act);
    return try expertLaunch(false, act, ids, fc2[0], fc2[1], fc2[2], pairs, topk, gs2, s);
}

/// Dense bf16 router logits for R rows, w [E, D]: one threadgroup of SG
/// simdgroups per expert; simdgroup g sums its D / SG inputs (lane l takes 4
/// consecutive inputs at a time, 128 apart), then the simdgroup sums add in
/// order. The row count is a runtime value, so a row's bits never depend on it.
const ROUTER_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint g = simdgroup_index_in_threadgroup;
    \\const int e = int(threadgroup_position_in_grid.y);
    \\const int R = rows;
    \\constexpr int PART = D / SG;
    \\threadgroup float part[MAX_ROWS][SG];
    \\float acc[MAX_ROWS];
    \\for (int r = 0; r < MAX_ROWS; r++) acc[r] = 0.0f;
    \\const int begin = int(g) * PART;
    \\for (int c = begin + 4 * int(lane); c < begin + PART; c += 128) {
    \\  const float w0 = float(w[size_t(e) * D + c]), w1 = float(w[size_t(e) * D + c + 1]);
    \\  const float w2 = float(w[size_t(e) * D + c + 2]), w3 = float(w[size_t(e) * D + c + 3]);
    \\  for (int r = 0; r < MAX_ROWS; r++) {
    \\    if (r >= R) break;
    \\    const device bfloat* xr = x + r * D + c;
    \\    acc[r] = fma(float(xr[3]), w3, fma(float(xr[2]), w2, fma(float(xr[1]), w1, fma(float(xr[0]), w0, acc[r]))));
    \\  }
    \\}
    \\for (int r = 0; r < MAX_ROWS; r++) {
    \\  if (r >= R) break;
    \\  const float total = simd_sum(acc[r]);
    \\  if (lane == 0) part[r][g] = total;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (g == 0 && int(lane) < R) {
    \\  float total = 0.0f;
    \\  for (int k = 0; k < SG; k++) total += part[lane][k];
    \\  y[int(lane) * E + e] = bfloat(total);
    \\}
;

const ROUTER_SG = 8;
var router_kernel: ?mlx.mlx_fast_metal_kernel = null;
const RouterKey = struct { rows: c_int, e: c_int, d: c_int };
var router_cfgs: std.AutoHashMapUnmanaged(RouterKey, mlx.mlx_fast_metal_kernel_config) = .{};

/// `x [R, D] @ w.T` for a dense bf16 w [E, D] (a router), 1..MAX_ROWS rows,
/// or null outside the kernel.
pub fn router(x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    if (xs.len != 2 or ws.len != 2 or xs[1] != ws[1] or @rem(ws[1], 4 * ROUTER_SG) != 0) return null;
    if (xs[0] < 1 or xs[0] > MAX_ROWS) return null;
    const key = RouterKey{ .rows = xs[0], .e = ws[0], .d = ws[1] };
    if (router_kernel == null) router_kernel = try newKernel("msv_rowqmv_router", &.{ "x", "w", "rows" }, &.{"y"}, ROUTER_SOURCE);
    const config = router_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ key.rows, key.e }, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * ROUTER_SG, key.e, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * ROUTER_SG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "D", key.d));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "E", key.e));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SG", ROUTER_SG));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "MAX_ROWS", MAX_ROWS));
        try router_cfgs.put(std.heap.c_allocator, key, c);
        break :blk c;
    };
    const rows = mlx.mlx_array_new_int(key.rows);
    defer _ = mlx.mlx_array_free(rows);
    const inputs = [_]mlx.mlx_array{ x, w, rows };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, router_kernel.?, in_vec, config, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

// ── tests ──

const testing = std.testing;

fn randBf16(a: std.mem.Allocator, shape: []const c_int, scale: f32, rnd: std.Random, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try a.alloc(f32, n);
    defer a.free(buf);
    for (buf) |*v| v.* = scale * rnd.floatNorm(f32);
    const f = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

const Quant = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    fn deinit(q: Quant) void {
        _ = mlx.mlx_array_free(q.w);
        _ = mlx.mlx_array_free(q.sc);
        _ = mlx.mlx_array_free(q.bi);
    }
};

fn quantize(wf: mlx.mlx_array, gs: c_int, s: mlx.mlx_stream) !Quant {
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var q = Quant{ .w = mlx.mlx_array_new(), .sc = mlx.mlx_array_new(), .bi = mlx.mlx_array_new() };
    try mlx.check(mlx.mlx_vector_array_get(&q.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q.sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q.bi, triple, 2));
    return q;
}

fn sliceRows(x: mlx.mlx_array, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const k = mlx.getShape(x)[1];
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ lo, 0 }, 2, &[_]c_int{ hi, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

fn expectBitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

/// Max |got - truth| relative to max |truth|, truth = f32 x @ dequant(w).T.
fn relErr(got: mlx.mlx_array, x: mlx.mlx_array, q: Quant, gs: c_int, s: mlx.mlx_stream) !f32 {
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_dequantize(&wt, q.w, q.sc, q.bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var wtt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wtt);
    try mlx.check(mlx.mlx_transpose(&wtt, wt, s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wtt, s));
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var at = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(at);
    try mlx.check(mlx.mlx_abs(&at, truth, s));
    var md = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(md);
    try mlx.check(mlx.mlx_max(&md, ad, false, s));
    var mt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mt);
    try mlx.check(mlx.mlx_max(&mt, at, false, s));
    var a: f32 = 0;
    var b: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&a, md));
    try mlx.check(mlx.mlx_array_item_float32(&b, mt));
    return a / b;
}

test "rowqmv.qmv: every row of an R-row call equals its one-row call bit for bit, and matches f32 truth" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7f);
    const rnd = prng.random();
    // Nemotron-H's shapes: K = 2688 (not a multiple of 512), 4096, 1856, 3712.
    const Shape = struct { n: c_int, k: c_int, gs: c_int };
    for ([_]Shape{ .{ .n = 1024, .k = 2688, .gs = 64 }, .{ .n = 512, .k = 4096, .gs = 64 }, .{ .n = 2688, .k = 1856, .gs = 64 }, .{ .n = 256, .k = 3712, .gs = 32 }, .{ .n = 264, .k = 640, .gs = 128 } }) |sh| {
        const wf = try randBf16(a, &.{ sh.n, sh.k }, 0.02, rnd, s);
        defer _ = mlx.mlx_array_free(wf);
        const q = try quantize(wf, sh.gs, s);
        defer q.deinit();
        const x = try randBf16(a, &.{ MAX_ROWS, sh.k }, 1.0, rnd, s);
        defer _ = mlx.mlx_array_free(x);

        const all = (try qmv(x, q.w, q.sc, q.bi, 4, @intCast(sh.gs), s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(all);
        try testing.expect(try relErr(all, x, q, sh.gs, s) < 2e-2);
        var r: c_int = 0;
        while (r < MAX_ROWS) : (r += 1) {
            const one_x = try sliceRows(x, r, r + 1, s);
            defer _ = mlx.mlx_array_free(one_x);
            const one = (try qmv(one_x, q.w, q.sc, q.bi, 4, @intCast(sh.gs), s)).?;
            defer _ = mlx.mlx_array_free(one);
            const row = try sliceRows(all, r, r + 1, s);
            defer _ = mlx.mlx_array_free(row);
            try expectBitEqual(one, row, s);
        }
        // A 3-row window starting mid-batch gives the same rows too.
        const win_x = try sliceRows(x, 5, 8, s);
        defer _ = mlx.mlx_array_free(win_x);
        const win = (try qmv(win_x, q.w, q.sc, q.bi, 4, @intCast(sh.gs), s)).?;
        defer _ = mlx.mlx_array_free(win);
        const want = try sliceRows(all, 5, 8, s);
        defer _ = mlx.mlx_array_free(want);
        try expectBitEqual(win, want, s);
    }
}

test "rowqmv.qmvAct: relu2 is max(bf16(y), 0)^2 rounded to bf16, rows width-invariant" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const wf = try randBf16(a, &.{ 3712, 2688 }, 0.02, prng.random(), s);
    defer _ = mlx.mlx_array_free(wf);
    const q = try quantize(wf, 64, s);
    defer q.deinit();
    const x = try randBf16(a, &.{ 5, 2688 }, 1.0, prng.random(), s);
    defer _ = mlx.mlx_array_free(x);
    const plain = (try qmv(x, q.w, q.sc, q.bi, 4, 64, s)).?;
    defer _ = mlx.mlx_array_free(plain);
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var zb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zb);
    try mlx.check(mlx.mlx_astype(&zb, zero, .bfloat16, s));
    var mx_ = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx_);
    try mlx.check(mlx.mlx_maximum(&mx_, plain, zb, s));
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_square(&want, mx_, s));
    const got = (try qmvAct(x, q.w, q.sc, q.bi, 4, 64, true, s)).?;
    defer _ = mlx.mlx_array_free(got);
    try expectBitEqual(got, want, s);
    const x1 = try sliceRows(x, 3, 4, s);
    defer _ = mlx.mlx_array_free(x1);
    const one = (try qmvAct(x1, q.w, q.sc, q.bi, 4, 64, true, s)).?;
    defer _ = mlx.mlx_array_free(one);
    const row = try sliceRows(got, 3, 4, s);
    defer _ = mlx.mlx_array_free(row);
    try expectBitEqual(one, row, s);
}

test "rowqmv.qmv: declines what it cannot read" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const wf = try randBf16(a, &.{ 64, 512 }, 0.02, prng.random(), s);
    defer _ = mlx.mlx_array_free(wf);
    const q = try quantize(wf, 64, s);
    defer q.deinit();
    const x = try randBf16(a, &.{ MAX_ROWS + 1, 512 }, 1.0, prng.random(), s);
    defer _ = mlx.mlx_array_free(x);
    try testing.expect((try qmv(x, q.w, q.sc, q.bi, 4, 64, s)) == null);
    const x1 = try sliceRows(x, 0, 1, s);
    defer _ = mlx.mlx_array_free(x1);
    try testing.expect((try qmv(x1, q.w, q.sc, q.bi, 8, 64, s)) == null);
}

fn sumSq(v: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, v, s));
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_sum(&t, sq, false, s));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, t));
    return out;
}

fn takeExpert(bank: mlx.mlx_array, e: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(bank);
    var sl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sl);
    try mlx.check(mlx.mlx_slice(&sl, bank, &[_]c_int{ e, 0, 0 }, 3, &[_]c_int{ e + 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, sl, &[_]c_int{ sh[1], sh[2] }, 2, s));
    return out;
}

fn dequantF32(q: [3]mlx.mlx_array, gs: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&w, q[0], q[1], q[2], mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    return w;
}

test "rowqmv.experts: rows are width-invariant bit for bit and match the f32 SwitchMLP" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xE1);
    const rnd = prng.random();
    // Nemotron-H's geometry: D = 2688 and I = 1856 are not multiples of 512.
    const E: c_int = 8;
    const D: c_int = 2688;
    const I: c_int = 1856;
    const TOPK: c_int = 6;
    const f1 = try randBf16(a, &.{ E, I, D }, 0.03, rnd, s);
    defer _ = mlx.mlx_array_free(f1);
    const f2 = try randBf16(a, &.{ E, D, I }, 0.03, rnd, s);
    defer _ = mlx.mlx_array_free(f2);
    const q1 = try quantize(f1, 64, s);
    defer q1.deinit();
    const q2 = try quantize(f2, 64, s);
    defer q2.deinit();
    const fc1 = [3]mlx.mlx_array{ q1.w, q1.sc, q1.bi };
    const fc2 = [3]mlx.mlx_array{ q2.w, q2.sc, q2.bi };
    const x = try randBf16(a, &.{ MAX_ROWS, D }, 1.0, rnd, s);
    defer _ = mlx.mlx_array_free(x);
    var idv: [MAX_ROWS * TOPK]u32 = undefined;
    for (&idv) |*v| v.* = rnd.uintLessThan(u32, E);
    const ids = mlx.mlx_array_new_data(&idv, &[_]c_int{MAX_ROWS * TOPK}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ids);

    const all = (try experts(x, ids, TOPK, fc1, fc2, 64, 64, s)) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(all);
    try testing.expectEqualSlices(c_int, &.{ MAX_ROWS * TOPK, D }, mlx.getShape(all));

    // Width invariance: a 1-row call and a 4-row window give the same rows.
    for ([_][2]c_int{ .{ 0, 1 }, .{ 7, 8 }, .{ 3, 7 } }) |win| {
        const xw = try sliceRows(x, win[0], win[1], s);
        defer _ = mlx.mlx_array_free(xw);
        var idw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idw);
        try mlx.check(mlx.mlx_slice(&idw, ids, &[_]c_int{win[0] * TOPK}, 1, &[_]c_int{win[1] * TOPK}, 1, &[_]c_int{1}, 1, s));
        const got = (try experts(xw, idw, TOPK, fc1, fc2, 64, 64, s)).?;
        defer _ = mlx.mlx_array_free(got);
        const want = try sliceRows(all, win[0] * TOPK, win[1] * TOPK, s);
        defer _ = mlx.mlx_array_free(want);
        try expectBitEqual(got, want, s);
    }

    // Truth per pair: f32 x @ W1[e].T, relu^2, @ W2[e].T.
    var worst: f32 = 0;
    for (0..MAX_ROWS * TOPK) |p| {
        const e: c_int = @intCast(idv[p]);
        const r: c_int = @intCast(p / @as(usize, TOPK));
        var w1s: [3]mlx.mlx_array = undefined;
        var w2s: [3]mlx.mlx_array = undefined;
        for (0..3) |i| {
            w1s[i] = try takeExpert(fc1[i], e, s);
            w2s[i] = try takeExpert(fc2[i], e, s);
        }
        defer for (w1s ++ w2s) |t| {
            _ = mlx.mlx_array_free(t);
        };
        const d1 = try dequantF32(w1s, 64, s);
        defer _ = mlx.mlx_array_free(d1);
        const d2 = try dequantF32(w2s, 64, s);
        defer _ = mlx.mlx_array_free(d2);
        const xr = try sliceRows(x, r, r + 1, s);
        defer _ = mlx.mlx_array_free(xr);
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, xr, .float32, s));
        var t1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1);
        try mlx.check(mlx.mlx_transpose(&t1, d1, s));
        var h = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_matmul(&h, xf, t1, s));
        const zero = mlx.mlx_array_new_float(0);
        defer _ = mlx.mlx_array_free(zero);
        var hr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hr);
        try mlx.check(mlx.mlx_maximum(&hr, h, zero, s));
        var h2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h2);
        try mlx.check(mlx.mlx_square(&h2, hr, s));
        var t2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t2);
        try mlx.check(mlx.mlx_transpose(&t2, d2, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, h2, t2, s));
        const got = try sliceRows(all, @intCast(p), @intCast(p + 1), s);
        defer _ = mlx.mlx_array_free(got);
        var gf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gf);
        try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
        worst = @max(worst, @sqrt(try sumSq(d, s) / try sumSq(truth, s)));
    }
    try testing.expect(worst < 2e-2);
}

test "rowqmv.router: rows are width-invariant bit for bit and match f32 truth" {
    const s = mlx.gpuStream();
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA7);
    const rnd = prng.random();
    const w = try randBf16(a, &.{ 128, 2688 }, 0.05, rnd, s);
    defer _ = mlx.mlx_array_free(w);
    const x = try randBf16(a, &.{ MAX_ROWS, 2688 }, 1.0, rnd, s);
    defer _ = mlx.mlx_array_free(x);
    const all = (try router(x, w, s)) orelse return error.Declined;
    defer _ = mlx.mlx_array_free(all);
    for ([_][2]c_int{ .{ 0, 1 }, .{ 9, 10 }, .{ 2, 5 } }) |win| {
        const xw = try sliceRows(x, win[0], win[1], s);
        defer _ = mlx.mlx_array_free(xw);
        const got = (try router(xw, w, s)).?;
        defer _ = mlx.mlx_array_free(got);
        const want = try sliceRows(all, win[0], win[1], s);
        defer _ = mlx.mlx_array_free(want);
        try expectBitEqual(got, want, s);
    }
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var wf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wf);
    try mlx.check(mlx.mlx_astype(&wf, w, .float32, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wf, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wt, s));
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&gf, all, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
    try testing.expect(@sqrt(try sumSq(d, s) / try sumSq(truth, s)) < 1e-2);
}
