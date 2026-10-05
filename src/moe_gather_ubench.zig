//! Env-gated microbench for the two fused decode MoE kernels at Qwen3.8-Flash-Next's
//! shapes (E 512, top-10, hidden 2560, intermediate 640, 4-bit affine g64, bf16):
//! `gatherQmvGateUp` and `gatherQmvDownReduce`, against MLX's stock `gather_qmm`
//! chains, so a kernel variant is timed in isolation instead of behind a server boot.
//!   MOE_UBENCH=1 zig build test -Doptimize=ReleaseFast -Dtest-filter="moe gather kernels"

const std = @import("std");
const mlx = @import("mlx.zig");
const xfm = @import("transformer.zig");
const io_util = @import("io_util.zig");

const E: c_int = 512;
const TOPK: c_int = 10;
const H: c_int = 2560;
const I: c_int = 640;
const BITS: u32 = 4;
const GS: u32 = 64;
const WARM = 5;
const BATCHES = 20;
const CALLS_SMALL = 10;
const MAX_CALLS = 40;

const Bank = struct {
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,

    fn deinit(b: Bank) void {
        _ = mlx.mlx_array_free(b.w);
        _ = mlx.mlx_array_free(b.sc);
        _ = mlx.mlx_array_free(b.bi);
    }

    /// Bytes one call reads from this bank: TOPK rows of packed words + bf16 scale and bias per group.
    fn bytesPerCall(n: c_int, k: c_int) f64 {
        const words = @divExact(@as(f64, @floatFromInt(k)) * @as(f64, @floatFromInt(BITS)), 32.0);
        const groups = @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(GS));
        return @as(f64, @floatFromInt(TOPK)) * @as(f64, @floatFromInt(n)) * (words * 4.0 + groups * 4.0);
    }
};

fn randUniformBf16(rnd: std.Random, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

/// Any bit pattern is a valid 4-bit row, so the words come straight from the RNG.
fn affineBank(rnd: std.Random, n: c_int, k: c_int, s: mlx.mlx_stream) !Bank {
    const words: c_int = @intCast(@divExact(@as(u32, @intCast(k)) * BITS, 32));
    const groups: c_int = @divExact(k, @as(c_int, @intCast(GS)));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const w_shape = [_]c_int{ E, n, words };
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&w, &w_shape, 3, 4, key, s));
    const s_shape = [_]c_int{ E, n, groups };
    const sc = try randUniformBf16(rnd, &s_shape, 0.01, 0.06, s);
    const bi = try randUniformBf16(rnd, &s_shape, -0.5, 0.5, s);
    return .{ .w = w, .sc = sc, .bi = bi };
}

fn evalAll(outs: []const mlx.mlx_array) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (outs) |o| _ = mlx.mlx_vector_array_append_value(vec, o);
    try mlx.check(mlx.mlx_eval(vec));
}

fn allFinite(a: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var fin = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fin);
    try mlx.check(mlx.mlx_isfinite(&fin, a, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, fin, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var v = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, all));
    return v;
}

const Inputs = struct {
    s: mlx.mlx_stream,
    x: mlx.mlx_array, // [H] bf16
    x_down: mlx.mlx_array, // [TOPK, I] bf16
    gate: Bank,
    up: Bank,
    down: Bank,
    inds: [CALLS_SMALL]mlx.mlx_array, // uint32 [TOPK], distinct experts each, cycled by call index
    lhs_zero: mlx.mlx_array, // uint32 [TOPK] zeros: every expert reads the one token
    lhs_iota: mlx.mlx_array, // uint32 [TOPK] 0..9: every expert reads its own row
    scores: mlx.mlx_array, // [TOPK] bf16, sums to ~1
};

const Arm = enum { gateup, downred, pair, stock_gateup, stock_downred };

fn stockGateUp(in: *const Inputs, inds: mlx.mlx_array) !mlx.mlx_array {
    const s = in.s;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, in.x, &.{ 1, 1, H }, 3, s));
    var xg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xg);
    try mlx.check(mlx.mlx_broadcast_to(&xg, x3, &.{ TOPK, 1, H }, 3, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_gather_qmm(&g, xg, in.gate.w, in.gate.sc, in.gate.bi, in.lhs_zero, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_gather_qmm(&u, xg, in.up.w, in.up.sc, in.up.bi, in.lhs_zero, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var sg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sg);
    try mlx.check(mlx.mlx_sigmoid(&sg, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_multiply(&out, silu, u, s));
    return out;
}

fn stockDownReduce(in: *const Inputs, inds: mlx.mlx_array) !mlx.mlx_array {
    const s = in.s;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, in.x_down, &.{ TOPK, 1, I }, 3, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_gather_qmm(&d, x3, in.down.w, in.down.sc, in.down.bi, in.lhs_iota, inds, true, mlx.mlx_optional_int.some(@intCast(GS)), mlx.mlx_optional_int.some(@intCast(BITS)), "affine", false, s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, in.scores, &.{ TOPK, 1, 1 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 0, false, s));
    return out;
}

/// One call of `arm`. `pair` is the in-situ per-layer chain: gate+up, then down+reduce over
/// the activation it produced, so its two dispatches are dependent like a real layer's.
fn call(in: *const Inputs, arm: Arm, i: usize) !mlx.mlx_array {
    const inds = in.inds[i % CALLS_SMALL];
    switch (arm) {
        .gateup => return (try xfm.gatherQmvGateUp(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, 0)) orelse error.GateUpDeclined,
        .downred => return (try xfm.gatherQmvDownReduce(in.s, in.x_down, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownReduceDeclined,
        .pair => {
            const act = (try xfm.gatherQmvGateUp(in.s, in.x, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, inds, BITS, GS, .affine, 0)) orelse return error.GateUpDeclined;
            defer _ = mlx.mlx_array_free(act);
            return (try xfm.gatherQmvDownReduce(in.s, act, in.down.w, in.down.sc, in.down.bi, inds, in.scores, BITS, GS, .affine)) orelse error.DownReduceDeclined;
        },
        .stock_gateup => return stockGateUp(in, inds),
        .stock_downred => return stockDownReduce(in, inds),
    }
}

/// Returns the sorted wall times of BATCHES batches of `n` independent calls (distinct index
/// vectors per call, one eval per batch), after WARM warm-up batches.
fn timeBatches(in: *const Inputs, arm: Arm, n: usize, out: *[BATCHES]u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var outs: [MAX_CALLS]mlx.mlx_array = undefined;
    for (0..WARM + BATCHES) |b| {
        var sw = io_util.Stopwatch.init(io);
        for (outs[0..n], 0..) |*o, i| o.* = try call(in, arm, i);
        try evalAll(outs[0..n]);
        if (b >= WARM) out[b - WARM] = sw.read();
        for (outs[0..n]) |o| _ = mlx.mlx_array_free(o);
    }
    std.mem.sort(u64, out, {}, std.sort.asc(u64));
}

/// Prints the per-call time at CALLS_SMALL calls per batch and the MARGINAL per-call time (the
/// slope between CALLS_SMALL and MAX_CALLS calls per batch, which cancels the fixed eval/submit
/// cost each batch pays once), with the GB/s the marginal implies from the bank bytes a call reads.
fn bench(in: *const Inputs, arm: Arm, label: []const u8, bytes_per_call: f64) !void {
    var small: [BATCHES]u64 = undefined;
    try timeBatches(in, arm, CALLS_SMALL, &small);
    var large: [BATCHES]u64 = undefined;
    try timeBatches(in, arm, MAX_CALLS, &large);
    const span: f64 = @floatFromInt(MAX_CALLS - CALLS_SMALL);
    const med_us = @as(f64, @floatFromInt(small[BATCHES / 2])) / @as(f64, CALLS_SMALL) / 1000.0;
    const med_marg = (@as(f64, @floatFromInt(large[BATCHES / 2])) - @as(f64, @floatFromInt(small[BATCHES / 2]))) / span / 1000.0;
    const min_marg = (@as(f64, @floatFromInt(large[0])) - @as(f64, @floatFromInt(small[0]))) / span / 1000.0;
    std.debug.print("[moe-ubench] {s:<22} per-call@{d} {d:6.1} us | marginal median {d:6.1} us  min {d:6.1} us  ({d:5.1} MB/call -> {d:5.0} GB/s median, {d:5.0} GB/s min)\n", .{
        label, CALLS_SMALL, med_us, med_marg, min_marg, bytes_per_call / 1e6, bytes_per_call / (med_marg * 1e3), bytes_per_call / (min_marg * 1e3),
    });
}

// Bar: both fused kernels run at the Flash Next shapes and return finite outputs; timings are printed.
test "moe gather kernels: microbench (MOE_UBENCH=1)" {
    if (std.c.getenv("MOE_UBENCH") == null) return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var trash: [512]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    defer if (mlx.errorPending()) {
        _ = mlx.takeError(&trash);
    };
    xfm.gqmv_gateup_override = true;
    defer xfm.gqmv_gateup_override = null;
    xfm.downred_override = true;
    defer xfm.downred_override = null;

    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x30E0B3);
    const rnd = prng.random();

    var in: Inputs = undefined;
    in.s = s;
    in.x = try randUniformBf16(rnd, &.{H}, -1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(in.x);
    in.x_down = try randUniformBf16(rnd, &.{ TOPK, I }, -1.0, 1.0, s);
    defer _ = mlx.mlx_array_free(in.x_down);
    in.gate = try affineBank(rnd, I, H, s);
    defer in.gate.deinit();
    in.up = try affineBank(rnd, I, H, s);
    defer in.up.deinit();
    in.down = try affineBank(rnd, H, I, s);
    defer in.down.deinit();

    const ish = [_]c_int{TOPK};
    var made: usize = 0;
    defer for (in.inds[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&in.inds) |*a| {
        // Partial Fisher-Yates: TOPK distinct experts per vector, a fresh set per call.
        var pool: [E]u32 = undefined;
        for (&pool, 0..) |*p, k| p.* = @intCast(k);
        var pick: [TOPK]u32 = undefined;
        for (&pick, 0..) |*p, k| {
            const j = k + rnd.uintLessThan(usize, pool.len - k);
            std.mem.swap(u32, &pool[k], &pool[j]);
            p.* = pool[k];
        }
        a.* = mlx.mlx_array_new_data(&pick, &ish, 1, .uint32);
        made += 1;
    }
    const zeros = std.mem.zeroes([TOPK]u32);
    in.lhs_zero = mlx.mlx_array_new_data(&zeros, &ish, 1, .uint32);
    defer _ = mlx.mlx_array_free(in.lhs_zero);
    var iota: [TOPK]u32 = undefined;
    for (&iota, 0..) |*v, k| v.* = @intCast(k);
    in.lhs_iota = mlx.mlx_array_new_data(&iota, &ish, 1, .uint32);
    defer _ = mlx.mlx_array_free(in.lhs_iota);
    {
        var sc: [TOPK]f32 = undefined;
        var sum: f32 = 0;
        for (&sc) |*v| {
            v.* = 0.05 + rnd.float(f32);
            sum += v.*;
        }
        for (&sc) |*v| v.* /= sum;
        const sc32 = mlx.mlx_array_new_data(&sc, &ish, 1, .float32);
        defer _ = mlx.mlx_array_free(sc32);
        in.scores = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&in.scores, sc32, .bfloat16, s));
    }
    defer _ = mlx.mlx_array_free(in.scores);
    try evalAll(&.{ in.x, in.x_down, in.gate.w, in.gate.sc, in.gate.bi, in.up.w, in.up.sc, in.up.bi, in.down.w, in.down.sc, in.down.bi, in.scores });

    // Sanity bar: finite outputs from both fused kernels.
    {
        const a = try call(&in, .gateup, 0);
        defer _ = mlx.mlx_array_free(a);
        try std.testing.expect(try allFinite(a, s));
        const d = try call(&in, .downred, 0);
        defer _ = mlx.mlx_array_free(d);
        try std.testing.expect(try allFinite(d, s));
    }

    const gateup_bytes = 2.0 * Bank.bytesPerCall(I, H);
    const down_bytes = Bank.bytesPerCall(H, I);
    std.debug.print("\n[moe-ubench] E={d} topk={d} hidden={d} inter={d} {d}-bit g{d} bf16; {d} batches each of {d} and {d} independent calls, one eval per batch\n", .{ E, TOPK, H, I, BITS, GS, BATCHES, CALLS_SMALL, MAX_CALLS });
    try bench(&in, .gateup, "gatherQmvGateUp", gateup_bytes);
    try bench(&in, .downred, "gatherQmvDownReduce", down_bytes);
    try bench(&in, .pair, "gateup -> downred", gateup_bytes + down_bytes);
    try bench(&in, .stock_gateup, "stock gather_qmm g+u", gateup_bytes);
    try bench(&in, .stock_downred, "stock gather_qmm down", down_bytes);
    try std.testing.expect(!mlx.errorPending());
}
