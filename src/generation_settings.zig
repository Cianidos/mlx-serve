const std = @import("std");

pub const Surface = enum { chat, completions, messages, responses };
pub const Field = enum { temperature, top_p, top_k, repeat_penalty, presence_penalty, frequency_penalty, max_tokens, enable_thinking, reasoning_effort, reasoning_budget };
pub const Effort = enum { none, minimal, low, medium, high, xhigh, max };
pub const Value = union(enum) { number: f64, integer: i64, boolean: bool, effort: Effort };
pub const Rule = struct { value: Value, ignore_client: bool = false, source: Source = .global };
pub var cli_reasoning_budget: ?i32 = null;

pub const Profile = struct {
    rules: [std.enums.values(Field).len]?Rule = @splat(null),

    pub fn set(self: *Profile, field: Field, value: Value, ignore_client: bool) void {
        self.rules[@backingInt(field)] = .{ .value = value, .ignore_client = ignore_client };
    }
};
pub const Source = enum { client, model, global, cli, checkpoint, fallback };
pub const Resolved = struct {
    root: std.json.ObjectMap,
    values: [std.enums.values(Field).len]?std.json.Value = @splat(null),
    sources: [std.enums.values(Field).len]Source = @splat(.fallback),
    enforced: [std.enums.values(Field).len]bool = @splat(false),

    pub fn locked(self: Resolved, field: Field) bool {
        return self.enforced[@backingInt(field)];
    }

    pub fn json(self: Resolved, allocator: std.mem.Allocator) ![]u8 {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var object: std.json.ObjectMap = .empty;
        for (std.enums.values(Field)) |field| {
            const index = @backingInt(field);
            const value = self.values[index] orelse continue;
            var item: std.json.ObjectMap = .empty;
            try item.put(a, "source", .{ .string = @tagName(self.sources[index]) });
            try item.put(a, "ignore_client", .{ .bool = self.enforced[index] });
            try item.put(a, "value", value);
            try object.put(a, @tagName(field), .{ .object = item });
        }
        return std.json.Stringify.valueAlloc(allocator, std.json.Value{ .object = object }, .{});
    }
};

pub fn fieldKey(field: Field) []const u8 {
    return if (field == .reasoning_budget) "reasoning_budget_tokens" else @tagName(field);
}

fn nested(root: std.json.ObjectMap, object: []const u8, key: []const u8) ?std.json.Value {
    const value = root.get(object) orelse return null;
    return if (value == .object) value.object.get(key) else null;
}

fn clientValue(root: std.json.ObjectMap, field: Field, surface: Surface) ?std.json.Value {
    return switch (field) {
        .max_tokens => if (surface == .responses)
            root.get("max_output_tokens") orelse root.get("max_tokens") orelse root.get("max_completion_tokens")
        else
            root.get("max_tokens") orelse root.get("max_completion_tokens") orelse root.get("max_output_tokens"),
        .reasoning_budget => if (surface == .messages)
            nested(root, "thinking", "budget_tokens") orelse root.get("reasoning_budget_tokens") orelse root.get("reasoning_budget")
        else
            root.get("reasoning_budget_tokens") orelse root.get("reasoning_budget"),
        .reasoning_effort => if (surface == .messages)
            nested(root, "output_config", "effort") orelse root.get("reasoning_effort") orelse nested(root, "chat_template_kwargs", "reasoning_effort")
        else if (surface == .responses)
            nested(root, "reasoning", "effort") orelse root.get("reasoning_effort") orelse nested(root, "chat_template_kwargs", "reasoning_effort")
        else
            root.get("reasoning_effort") orelse nested(root, "chat_template_kwargs", "reasoning_effort"),
        .enable_thinking => blk: {
            if (surface == .messages) {
                if (nested(root, "thinking", "type")) |value| if (value == .string) {
                    break :blk .{ .bool = std.mem.eql(u8, value.string, "enabled") or std.mem.eql(u8, value.string, "adaptive") };
                };
            }
            if (root.get("enable_thinking") orelse nested(root, "chat_template_kwargs", "enable_thinking")) |value| break :blk value;
            if (surface == .responses and root.get("reasoning") != null) {
                const effort = nested(root, "reasoning", "effort");
                break :blk .{ .bool = effort == null or effort.? != .string or !std.mem.eql(u8, effort.?.string, "none") };
            }
            const effort = switch (surface) {
                .messages => nested(root, "output_config", "effort"),
                else => root.get("reasoning_effort") orelse nested(root, "chat_template_kwargs", "reasoning_effort"),
            };
            if (effort) |value| if (value == .string) break :blk .{ .bool = !std.mem.eql(u8, value.string, "none") };
            break :blk null;
        },
        .repeat_penalty => root.get("repeat_penalty") orelse root.get("repetition_penalty"),
        else => root.get(fieldKey(field)),
    };
}

fn jsonValue(value: Value) std.json.Value {
    return switch (value) {
        .number => |v| .{ .float = v },
        .integer => |v| .{ .integer = v },
        .boolean => |v| .{ .bool = v },
        .effort => |v| .{ .string = @tagName(v) },
    };
}

fn childCopy(a: std.mem.Allocator, root: std.json.ObjectMap, key: []const u8) !std.json.ObjectMap {
    const value = root.get(key) orelse return .empty;
    return if (value == .object) try value.object.clone(a) else .empty;
}

pub fn apply(allocator: std.mem.Allocator, root: std.json.ObjectMap, surface: Surface, global: Profile, model: Profile) !Resolved {
    var result = Resolved{ .root = try root.clone(allocator) };
    for (std.enums.values(Field)) |field| {
        const index = @backingInt(field);
        const rule = model.rules[index] orelse global.rules[index];
        const client = clientValue(root, field, surface);
        const from_model = model.rules[index] != null;
        if (field == .reasoning_effort and client == null and !(rule != null and rule.?.ignore_client)) {
            if (clientValue(root, .enable_thinking, surface)) |thinking| if (thinking == .bool and !thinking.bool) continue;
        }
        const chosen: std.json.Value = if (rule != null and (rule.?.ignore_client or client == null)) blk: {
            result.sources[index] = if (from_model) .model else rule.?.source;
            break :blk jsonValue(rule.?.value);
        } else if (client) |value| blk: {
            result.sources[index] = .client;
            break :blk value;
        } else continue;
        result.enforced[index] = if (rule) |r| r.ignore_client else false;
        result.values[index] = chosen;
        if (surface == .completions and (field == .enable_thinking or field == .reasoning_effort or field == .reasoning_budget)) {
            if (rule != null and rule.?.ignore_client) return error.UnsupportedCompletionThinkingPolicy;
            continue;
        }
        if (rule == null and surface != .chat and surface != .responses and field != .reasoning_budget and field != .max_tokens) continue;
        if (rule == null and field == .enable_thinking and surface == .responses and root.get("enable_thinking") == null) continue;
        try result.root.put(allocator, fieldKey(field), chosen);
        if (field == .max_tokens) {
            if (surface == .responses) try result.root.put(allocator, "max_output_tokens", chosen);
        }
        if (surface == .messages and (field == .enable_thinking or (field == .reasoning_budget and result.root.get("thinking") != null))) {
            var thinking = try childCopy(allocator, result.root, "thinking");
            if (field == .enable_thinking and chosen == .bool) {
                const old_type = thinking.get("type");
                if (!chosen.bool or old_type == null or result.locked(.enable_thinking)) try thinking.put(allocator, "type", .{ .string = if (chosen.bool) "enabled" else "disabled" });
            }
            if (field == .reasoning_budget) try thinking.put(allocator, "budget_tokens", chosen);
            try result.root.put(allocator, "thinking", .{ .object = thinking });
        }
        if (field == .reasoning_effort and (surface == .messages or surface == .responses)) {
            const key = if (surface == .messages) "output_config" else "reasoning";
            var object = try childCopy(allocator, result.root, key);
            try object.put(allocator, "effort", chosen);
            try result.root.put(allocator, key, .{ .object = object });
        }
    }
    if (result.locked(.reasoning_effort) and !result.locked(.enable_thinking)) {
        const effort = result.root.get("reasoning_effort").?;
        const on = effort == .string and !std.mem.eql(u8, effort.string, "none");
        try result.root.put(allocator, "enable_thinking", .{ .bool = on });
        result.values[@backingInt(Field.enable_thinking)] = .{ .bool = on };
        result.sources[@backingInt(Field.enable_thinking)] = result.sources[@backingInt(Field.reasoning_effort)];
        if (surface == .messages) {
            var thinking = try childCopy(allocator, result.root, "thinking");
            try thinking.put(allocator, "type", .{ .string = if (on) "enabled" else "disabled" });
            try result.root.put(allocator, "thinking", .{ .object = thinking });
        }
    }
    if (result.locked(.enable_thinking)) {
        const on = result.root.get("enable_thinking").?.bool;
        if (!on) {
            try result.root.put(allocator, "reasoning_effort", .{ .string = "none" });
            result.values[@backingInt(Field.reasoning_effort)] = .{ .string = "none" };
            result.sources[@backingInt(Field.reasoning_effort)] = result.sources[@backingInt(Field.enable_thinking)];
            if (surface == .messages or surface == .responses) {
                const key = if (surface == .messages) "output_config" else "reasoning";
                var object = try childCopy(allocator, result.root, key);
                try object.put(allocator, "effort", .{ .string = "none" });
                try result.root.put(allocator, key, .{ .object = object });
            }
        } else if (result.root.get("reasoning_effort")) |effort| {
            if (effort == .string and std.mem.eql(u8, effort.string, "none")) {
                _ = result.root.swapRemove("reasoning_effort");
                result.values[@backingInt(Field.reasoning_effort)] = null;
                if (surface == .messages or surface == .responses) {
                    const key = if (surface == .messages) "output_config" else "reasoning";
                    var object = try childCopy(allocator, result.root, key);
                    _ = object.swapRemove("effort");
                    try result.root.put(allocator, key, .{ .object = object });
                }
            }
        }
    }
    if (surface == .messages or surface == .responses) {
        if (result.root.get("enable_thinking")) |value| if (value == .bool and value.bool) {
            const key = if (surface == .messages) "thinking" else "reasoning";
            if (result.root.get(key) == null) {
                var object: std.json.ObjectMap = .empty;
                try object.put(allocator, if (surface == .messages) "type" else "effort", .{ .string = if (surface == .messages) "enabled" else "high" });
                try result.root.put(allocator, key, .{ .object = object });
            }
        };
    }
    const repeat_value = result.values[@backingInt(Field.repeat_penalty)];
    const neutral_repeat = if (repeat_value) |v| switch (v) {
        .float => |n| n == 1,
        .integer => |n| n == 1,
        else => false,
    } else false;
    if (!result.locked(.repeat_penalty) and result.root.get("frequency_penalty") != null and
        (result.sources[@backingInt(Field.frequency_penalty)] == .client or neutral_repeat) and
        result.sources[@backingInt(Field.repeat_penalty)] != .client)
    {
        _ = result.root.swapRemove("repeat_penalty");
        result.values[@backingInt(Field.repeat_penalty)] = null;
    }
    if (result.locked(.repeat_penalty)) {
        _ = result.root.swapRemove("frequency_penalty");
        result.values[@backingInt(Field.frequency_penalty)] = null;
    }
    if (result.locked(.frequency_penalty) and !result.locked(.repeat_penalty)) {
        _ = result.root.swapRemove("repeat_penalty");
        result.values[@backingInt(Field.repeat_penalty)] = null;
    }
    if (result.root.get("repeat_penalty") == null) {
        if (result.root.get("frequency_penalty")) |frequency| {
            const number: f64 = switch (frequency) {
                .float => |n| n,
                .integer => |n| @floatFromInt(n),
                else => 0,
            };
            try result.root.put(allocator, "repeat_penalty", .{ .float = 1 + @min(@max(number, 0), 2) });
        }
    }
    return result;
}

pub fn validateThinkingFallback(result: Resolved, thinking: bool) !void {
    if (result.locked(.enable_thinking)) {
        const value = result.values[@backingInt(Field.enable_thinking)] orelse return;
        if (value == .bool and value.bool != thinking) return error.UnsupportedThinkingPolicy;
    }
    if (result.locked(.reasoning_effort)) {
        const value = result.values[@backingInt(Field.reasoning_effort)] orelse return;
        if (value == .string and !std.mem.eql(u8, value.string, "none") and !thinking) return error.UnsupportedThinkingPolicy;
    }
}

pub fn validateEnginePolicy(result: Resolved, embedded: bool) !void {
    if (!embedded) return;
    for ([_]Field{ .repeat_penalty, .presence_penalty, .frequency_penalty }) |field| {
        const value = result.values[@backingInt(field)] orelse continue;
        const number: f64 = switch (value) {
            .float => |n| n,
            .integer => |n| @floatFromInt(n),
            else => continue,
        };
        const neutral: f64 = if (field == .repeat_penalty) 1 else 0;
        if (number != neutral and result.sources[@backingInt(field)] != .client) return error.UnsupportedEngineGenerationPolicy;
    }
}

pub fn parseProfile(value: std.json.Value) !Profile {
    if (value != .object) return error.InvalidGenerationSettings;
    var profile = Profile{};
    var it = value.object.iterator();
    while (it.next()) |item| {
        const field = std.meta.stringToEnum(Field, item.key_ptr.*) orelse return error.UnknownGenerationSetting;
        const object = item.value_ptr.*;
        if (object != .object) return error.InvalidGenerationSettings;
        for (object.object.keys()) |key| {
            if (!std.mem.eql(u8, key, "value") and !std.mem.eql(u8, key, "ignore_client")) return error.InvalidGenerationSettings;
        }
        const raw = object.object.get("value") orelse return error.InvalidGenerationSettings;
        const lock = object.object.get("ignore_client") orelse std.json.Value{ .bool = false };
        if (lock != .bool) return error.InvalidGenerationSettings;
        const v: Value = switch (field) {
            .enable_thinking => if (raw == .bool) .{ .boolean = raw.bool } else return error.InvalidGenerationSettings,
            .reasoning_effort => if (raw == .string) .{ .effort = std.meta.stringToEnum(Effort, raw.string) orelse return error.InvalidGenerationSettings } else return error.InvalidGenerationSettings,
            .top_k, .max_tokens, .reasoning_budget => blk: {
                const number: f64 = switch (raw) {
                    .integer => |n| @floatFromInt(n),
                    .float => |n| n,
                    else => return error.InvalidGenerationSettings,
                };
                const min: f64 = if (field == .reasoning_budget) -1 else 0;
                const max: f64 = if (field == .top_k) 1000 else std.math.maxInt(i32);
                if (!std.math.isFinite(number) or number < min or number > max or @trunc(number) != number) return error.InvalidGenerationSettings;
                break :blk .{ .integer = @intFromFloat(number) };
            },
            else => blk: {
                const number: f64 = switch (raw) {
                    .integer => |n| @floatFromInt(n),
                    .float => |n| n,
                    else => return error.InvalidGenerationSettings,
                };
                const max: f64 = if (field == .repeat_penalty) 10 else if (field == .top_p) 1 else 2;
                const min: f64 = if (field == .repeat_penalty) 0.01 else 0;
                if (!std.math.isFinite(number) or number < min or number > max) return error.InvalidGenerationSettings;
                break :blk .{ .number = number };
            },
        };
        profile.set(field, v, lock.bool);
    }
    return profile;
}

pub const FileCache = struct {
    mutex: std.Io.Mutex = .init,
    parsed: ?std.json.Parsed(std.json.Value) = null,
    stamp: ?Stamp = null,
    failure: ?anyerror = null,
    checked_ms: ?i64 = null,
    const Stamp = struct { size: u64, mtime: i96, inode: u64 };

    pub fn deinit(self: *FileCache) void {
        if (self.parsed) |*p| p.deinit();
        self.parsed = null;
    }

    pub fn get(self: *FileCache, io: std.Io, path: []const u8, model_path: ?[]const u8) !Profile {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const now = std.Io.Timestamp.now(io, .boot).toMilliseconds();
        if (self.checked_ms == null or now - self.checked_ms.? >= 1000) {
            self.checked_ms = now;
            const stamp: ?Stamp = if (std.Io.Dir.cwd().statFile(io, path, .{})) |s|
                .{ .size = s.size, .mtime = s.mtime.nanoseconds, .inode = @intCast(s.inode) }
            else |err| switch (err) {
                error.FileNotFound => null,
                else => {
                    self.checked_ms = null;
                    return err;
                },
            };
            if (!std.meta.eql(stamp, self.stamp)) {
                self.stamp = stamp;
                self.failure = null;
                self.deinit();
                if (stamp != null) {
                    const a = std.heap.page_allocator;
                    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1024 * 1024)) catch |err| {
                        self.failure = err;
                        return err;
                    };
                    defer a.free(bytes);
                    self.parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch |err| {
                        self.failure = err;
                        return err;
                    };
                }
            }
        }
        if (self.failure) |err| return err;
        const parsed = self.parsed orelse return .{};
        if (parsed.value != .object) return error.InvalidGenerationSettings;
        if (model_path) |path_key| {
            var it = parsed.value.object.iterator();
            while (it.next()) |item| {
                if (!std.mem.eql(u8, std.mem.trimEnd(u8, path_key, "/"), std.mem.trimEnd(u8, item.key_ptr.*, "/"))) continue;
                if (item.value_ptr.* != .object) return error.InvalidGenerationSettings;
                const obj = item.value_ptr.object;
                var profile = if (obj.get("generation_defaults")) |value| try parseProfile(value) else Profile{};
                if (obj.get("chat_template_kwargs")) |kw| if (kw == .object) {
                    if (profile.rules[@backingInt(Field.enable_thinking)] == null) {
                        if (kw.object.get("enable_thinking")) |v| if (v == .bool) profile.set(.enable_thinking, .{ .boolean = v.bool }, false);
                    }
                    if (profile.rules[@backingInt(Field.reasoning_effort)] == null) {
                        if (kw.object.get("reasoning_effort")) |v| if (v == .string) {
                            if (std.meta.stringToEnum(Effort, v.string)) |effort| profile.set(.reasoning_effort, .{ .effort = effort }, false);
                        };
                    }
                };
                return profile;
            }
            return .{};
        }
        return parseProfile(parsed.value);
    }
};

pub fn profileJson(a: std.mem.Allocator, profile: Profile) ![]u8 {
    var object: std.json.ObjectMap = .empty;
    defer object.deinit(a);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    for (std.enums.values(Field)) |field| if (profile.rules[@backingInt(field)]) |rule| {
        var item: std.json.ObjectMap = .empty;
        try item.put(arena.allocator(), "value", jsonValue(rule.value));
        try item.put(arena.allocator(), "ignore_client", .{ .bool = rule.ignore_client });
        try object.put(a, @tagName(field), .{ .object = item });
    };
    return std.json.Stringify.valueAlloc(a, std.json.Value{ .object = object }, .{});
}

test "generation settings: resolved diagnostics retain values sources and locks" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global = Profile{};
    global.set(.reasoning_budget, .{ .integer = 1024 }, true);
    const result = try apply(a, .empty, .chat, global, .{});
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, a, try result.json(a), .{});
    const budget = parsed.object.get("reasoning_budget").?.object;
    try t.expectEqual(@as(i64, 1024), budget.get("value").?.integer);
    try t.expectEqualStrings("global", budget.get("source").?.string);
    try t.expect(budget.get("ignore_client").?.bool);
}

test "generation settings: file lookup failures never fall open on the next request" {
    const t = std.testing;
    const path = try t.allocator.alloc(u8, std.fs.max_path_bytes + 1);
    defer t.allocator.free(path);
    @memset(path, 'x');
    var cache = FileCache{};
    defer cache.deinit();
    try t.expectError(error.NameTooLong, cache.get(t.io, path, null));
    try t.expectError(error.NameTooLong, cache.get(t.io, path, null));
}

test "generation settings: effective values reflect dominant thinking and penalty controls" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global = Profile{};
    global.set(.enable_thinking, .{ .boolean = false }, true);
    global.set(.reasoning_effort, .{ .effort = .high }, true);
    const off = try apply(a, .empty, .chat, global, .{});
    try validateThinkingFallback(off, false);
    try t.expectEqualStrings("none", off.values[@backingInt(Field.reasoning_effort)].?.string);
    global = .{};
    global.set(.reasoning_effort, .{ .effort = .low }, true);
    const on = try apply(a, .empty, .chat, global, .{});
    try t.expect(on.values[@backingInt(Field.enable_thinking)].?.bool);
    global = .{};
    global.set(.repeat_penalty, .{ .number = 1.1 }, false);
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"frequency_penalty\":2}", .{});
    const penalty = try apply(a, request.object, .chat, global, .{});
    try t.expect(penalty.values[@backingInt(Field.repeat_penalty)] == null);
}

test "generation settings: structured-output fallback cannot silently disable locked thinking" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global = Profile{};
    global.set(.enable_thinking, .{ .boolean = true }, true);
    const result = try apply(a, .empty, .chat, global, .{});
    try t.expectError(error.UnsupportedThinkingPolicy, validateThinkingFallback(result, false));
    try validateThinkingFallback(result, true);
}

test "generation settings: unsupported configured penalties never pretend to work on embedded engines" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var global = Profile{};
    global.set(.repeat_penalty, .{ .number = 1.2 }, true);
    const got = try apply(a, .empty, .chat, global, .{});
    try t.expectError(error.UnsupportedEngineGenerationPolicy, validateEnginePolicy(got, true));
    try validateEnginePolicy(got, false);
    global.set(.repeat_penalty, .{ .number = 1 }, true);
    try validateEnginePolicy(try apply(a, .empty, .chat, global, .{}), true);
}

test "generation settings: explicit client thinking beats default effort across chat APIs" {
    const t = std.testing;
    for ([_]Surface{ .chat, .messages, .responses }) |surface| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const text = switch (surface) {
            .chat => "{\"enable_thinking\":false}",
            .messages => "{\"thinking\":{\"type\":\"disabled\"}}",
            .responses => "{\"reasoning\":{\"effort\":\"none\"}}",
            else => unreachable,
        };
        const request = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        var model = Profile{};
        model.set(.reasoning_effort, .{ .effort = .high }, false);
        const result = try apply(a, request.object, surface, .{}, model);
        try t.expectEqual(Source.client, result.sources[@backingInt(Field.enable_thinking)]);
        const effort = result.root.get("reasoning_effort");
        try t.expect(effort == null or (effort.? == .string and std.mem.eql(u8, effort.?.string, "none")));
    }
}

test "generation settings: defaults fill omissions without rewriting client native thinking" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"thinking\":{\"type\":\"adaptive\",\"display\":\"omitted\"},\"output_config\":{\"effort\":\"xhigh\"}}", .{});
    var model = Profile{};
    model.set(.enable_thinking, .{ .boolean = false }, false);
    model.set(.reasoning_effort, .{ .effort = .low }, false);
    model.set(.reasoning_budget, .{ .integer = 1024 }, false);
    const got = try apply(a, request.object, .messages, .{}, model);
    const thinking = got.root.get("thinking").?.object;
    try t.expectEqualStrings("adaptive", thinking.get("type").?.string);
    try t.expectEqualStrings("omitted", thinking.get("display").?.string);
    try t.expectEqual(@as(i64, 1024), thinking.get("budget_tokens").?.integer);
    try t.expectEqualStrings("xhigh", got.root.get("output_config").?.object.get("effort").?.string);
    try t.expectEqual(Source.client, got.sources[@backingInt(Field.reasoning_effort)]);
    try t.expectEqual(Source.model, got.sources[@backingInt(Field.reasoning_budget)]);
}

test "generation settings: missing rules preserve native request fields and explicit off stays off" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"thinking\":{\"type\":\"disabled\"},\"output_config\":{\"effort\":\"xhigh\"}}", .{});
    const got = try apply(a, request.object, .messages, .{}, .{});
    try t.expectEqualStrings("disabled", got.root.get("thinking").?.object.get("type").?.string);
    var model = Profile{};
    model.set(.enable_thinking, .{ .boolean = true }, false);
    model.set(.reasoning_effort, .{ .effort = .high }, false);
    const off = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"thinking\":{\"type\":\"disabled\"}}", .{});
    const resolved = try apply(a, off.object, .messages, .{}, model);
    try t.expectEqualStrings("disabled", resolved.root.get("thinking").?.object.get("type").?.string);
    try t.expect(resolved.root.get("output_config") == null);
}

test "generation settings: frequency defaults reach the sampler on every text API" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var global = Profile{};
    global.set(.frequency_penalty, .{ .number = 0.5 }, false);
    for (std.enums.values(Surface)) |surface| {
        const result = try apply(arena.allocator(), .empty, surface, global, .{});
        try t.expectEqual(@as(f64, 1.5), result.root.get("repeat_penalty").?.float);
    }
}

test "generation settings: client frequency penalty overrides inherited repetition alias" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"frequency_penalty\":2}", .{});
    var global = Profile{};
    global.set(.repeat_penalty, .{ .number = 1.1 }, false);
    const result = try apply(a, request.object, .chat, global, .{});
    try t.expectEqual(@as(f64, 3), result.root.get("repeat_penalty").?.float);
    global.set(.repeat_penalty, .{ .number = 1.1 }, true);
    const locked = try apply(a, request.object, .chat, global, .{});
    try t.expectEqual(@as(f64, 1.1), locked.root.get("repeat_penalty").?.float);
    try t.expect(locked.root.get("frequency_penalty") == null);
}

test "generation settings: Responses honors an explicit thinking switch with native effort" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"enable_thinking\":false,\"reasoning\":{\"effort\":\"xhigh\"}}", .{});
    var global = Profile{};
    global.set(.enable_thinking, .{ .boolean = false }, true);
    const off = try apply(a, request.object, .responses, global, .{});
    try t.expectEqualStrings("none", off.root.get("reasoning").?.object.get("effort").?.string);
    global.set(.enable_thinking, .{ .boolean = true }, true);
    const on = try apply(a, request.object, .responses, global, .{});
    try t.expectEqualStrings("xhigh", on.root.get("reasoning").?.object.get("effort").?.string);
}

test "generation settings: frequency defaults replace neutral inherited repetition" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var global = Profile{};
    global.set(.repeat_penalty, .{ .number = 1 }, false);
    global.set(.frequency_penalty, .{ .number = 0.5 }, false);
    const got = try apply(arena.allocator(), .empty, .chat, global, .{});
    try t.expectEqual(@as(f64, 1.5), got.root.get("repeat_penalty").?.float);
}

test "generation settings: locked thinking-on cannot be disabled by client effort none" {
    const t = std.testing;
    for ([_]Surface{ .chat, .messages, .responses }) |surface| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"reasoning_effort\":\"none\",\"thinking\":{\"type\":\"disabled\"},\"output_config\":{\"effort\":\"none\"},\"reasoning\":{\"effort\":\"none\"}}", .{});
        var global = Profile{};
        global.set(.enable_thinking, .{ .boolean = true }, true);
        const got = try apply(a, request.object, surface, global, .{});
        try t.expect(got.root.get("enable_thinking").?.bool);
        if (surface == .messages) try t.expectEqualStrings("enabled", got.root.get("thinking").?.object.get("type").?.string);
        if (surface == .responses) {
            const effort = got.root.get("reasoning").?.object.get("effort");
            try t.expect(effort == null or !std.mem.eql(u8, effort.?.string, "none"));
        }
    }
}

test "generation settings: locked effort enables thinking while a locked off switch dominates" {
    const t = std.testing;
    for ([_]Surface{ .chat, .messages, .responses }) |surface| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const request = try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"enable_thinking\":false,\"thinking\":{\"type\":\"disabled\"},\"reasoning\":{\"effort\":\"none\"}}", .{});
        var global = Profile{};
        global.set(.reasoning_effort, .{ .effort = .low }, true);
        const on = try apply(a, request.object, surface, global, .{});
        if (surface == .messages) try t.expectEqualStrings("enabled", on.root.get("thinking").?.object.get("type").?.string);
        if (surface == .responses) try t.expectEqualStrings("low", on.root.get("reasoning").?.object.get("effort").?.string);
        global.set(.enable_thinking, .{ .boolean = false }, true);
        const off = try apply(a, request.object, surface, global, .{});
        try t.expectEqualStrings("none", off.root.get("reasoning_effort").?.string);
    }
}

test "generation settings: configured thinking budget accepts JSON integer wire values" {
    const t = std.testing;
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, "{\"reasoning_budget\":{\"value\":1024.0,\"ignore_client\":true}}", .{});
    defer parsed.deinit();
    try t.expectEqual(@as(i64, 1024), (try parseProfile(parsed.value)).rules[@backingInt(Field.reasoning_budget)].?.value.integer);
}

test "generation settings: invalid configured locks never fall open" {
    const t = std.testing;
    const good = try std.json.parseFromSlice(std.json.Value, t.allocator, "{\"top_k\":{\"value\":0,\"ignore_client\":true}}", .{});
    defer good.deinit();
    try t.expect((try parseProfile(good.value)).rules[@backingInt(Field.top_k)].?.ignore_client);
    for ([_][]const u8{ "{\"top_k\":{\"value\":-1}}", "{\"enable_thinking\":{\"value\":false,\"ignore_client\":1}}", "{\"reasoning_budget\":{\"value\":-2}}", "{\"reasoning_effort\":{\"value\":\"banana\"}}", "{\"top_k\":{\"value\":0,\"ignore_clent\":true}}" }) |text| {
        const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, text, .{});
        defer parsed.deinit();
        try t.expectError(error.InvalidGenerationSettings, parseProfile(parsed.value));
    }
}

test "generation settings: every field has client/model/global precedence and independent locks" {
    const t = std.testing;
    for (std.enums.values(Field)) |field| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const choices: [3]Value = switch (field) {
            .enable_thinking => .{ .{ .boolean = true }, .{ .boolean = false }, .{ .boolean = true } },
            .reasoning_effort => .{ .{ .effort = .low }, .{ .effort = .medium }, .{ .effort = .high } },
            else => .{ .{ .integer = 1 }, .{ .integer = 2 }, .{ .integer = 3 } },
        };
        var global = Profile{};
        var model = Profile{};
        global.set(field, choices[0], true);
        model.set(field, choices[1], false);
        var root: std.json.ObjectMap = .empty;
        const client: std.json.Value = switch (choices[2]) {
            .boolean => |v| .{ .bool = v },
            .effort => |v| .{ .string = @tagName(v) },
            .integer => |v| .{ .integer = v },
            else => unreachable,
        };
        const key = if (field == .reasoning_budget) "reasoning_budget_tokens" else @tagName(field);
        try root.put(a, key, client);
        const unlocked = try apply(a, root, .chat, global, model);
        try t.expectEqual(Source.client, unlocked.sources[@backingInt(field)]);
        try t.expect(!unlocked.locked(field));
        model.set(field, choices[1], true);
        const locked = try apply(a, root, .chat, global, model);
        try t.expectEqual(Source.model, locked.sources[@backingInt(field)]);
        try t.expect(locked.locked(field));
        const inherited = try apply(a, root, .chat, global, .{});
        try t.expectEqual(Source.global, inherited.sources[@backingInt(field)]);
        try t.expect(inherited.locked(field));
    }
}

test "generation settings: Claude adaptive effort cannot bypass locked thinking or budget" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const request = try std.json.parseFromSliceLeaky(std.json.Value, a,
        \\{"max_tokens":32000,"thinking":{"type":"adaptive","display":"omitted"},"output_config":{"effort":"xhigh"}}
    , .{});
    var global = Profile{};
    global.set(.reasoning_budget, .{ .integer = 1024 }, true);
    const budget = try apply(a, request.object, .messages, global, .{});
    try t.expectEqual(@as(i64, 1024), budget.root.get("thinking").?.object.get("budget_tokens").?.integer);
    global.set(.enable_thinking, .{ .boolean = false }, true);
    const off = try apply(a, request.object, .messages, global, .{});
    try t.expectEqualStrings("disabled", off.root.get("thinking").?.object.get("type").?.string);
    try t.expectEqualStrings("none", off.root.get("output_config").?.object.get("effort").?.string);
}

test "generation settings: file snapshots refresh after edits and invalid policy remains an error" {
    const t = std.testing;
    const io = t.io;
    var directory = t.tmpDir(.{});
    defer directory.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try directory.dir.realPath(io, &path_buf)];
    const path = try std.fs.path.join(t.allocator, &.{ dir, "generation-settings.json" });
    defer t.allocator.free(path);
    var cache = FileCache{};
    defer cache.deinit();
    try directory.dir.writeFile(io, .{ .sub_path = "generation-settings.json", .data = "{\"reasoning_budget\":{\"value\":1024,\"ignore_client\":true}}" });
    const old = try cache.get(io, path, null);
    try t.expectEqual(@as(i64, 1024), old.rules[@backingInt(Field.reasoning_budget)].?.value.integer);
    try directory.dir.writeFile(io, .{ .sub_path = "generation-settings.json", .data = "{\"reasoning_budget\":{\"value\":512,\"ignore_client\":false}}" });
    cache.checked_ms = null;
    const new = try cache.get(io, path, null);
    try t.expectEqual(@as(i64, 512), new.rules[@backingInt(Field.reasoning_budget)].?.value.integer);
    try t.expectEqual(@as(i64, 1024), old.rules[@backingInt(Field.reasoning_budget)].?.value.integer);
    try directory.dir.writeFile(io, .{ .sub_path = "generation-settings.json", .data = "{broken" });
    cache.checked_ms = null;
    try t.expectError(error.SyntaxError, cache.get(io, path, null));
    try t.expectError(error.SyntaxError, cache.get(io, path, null));
}

test "generation settings: per-model generation rules outrank legacy kwargs without overwriting other fields" {
    const t = std.testing;
    const io = t.io;
    var directory = t.tmpDir(.{});
    defer directory.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try directory.dir.realPath(io, &path_buf)];
    const path = try std.fs.path.join(t.allocator, &.{ dir, "model-settings.json" });
    defer t.allocator.free(path);
    try directory.dir.writeFile(io, .{ .sub_path = "model-settings.json", .data = "{\"/model/\":{\"ctx_size\":8192,\"chat_template_kwargs\":{\"enable_thinking\":true,\"reasoning_effort\":\"high\"},\"generation_defaults\":{\"reasoning_effort\":{\"value\":\"low\",\"ignore_client\":true}}}}" });
    var cache = FileCache{};
    defer cache.deinit();
    const profile = try cache.get(io, path, "/model");
    try t.expectEqual(Effort.low, profile.rules[@backingInt(Field.reasoning_effort)].?.value.effort);
    try t.expect(profile.rules[@backingInt(Field.enable_thinking)].?.value.boolean);
    try t.expect(profile.rules[@backingInt(Field.temperature)] == null);
}

test "generation settings: explicit neutral values and API aliases override unlocked defaults" {
    const t = std.testing;
    for (std.enums.values(Surface)) |surface| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        const request = try std.json.parseFromSliceLeaky(std.json.Value, a,
            \\{"top_k":0,"repeat_penalty":1,"presence_penalty":0,"frequency_penalty":0,"max_completion_tokens":12,"max_output_tokens":12,"reasoning_budget":0}
        , .{});
        var global = Profile{};
        global.set(.top_k, .{ .integer = 40 }, false);
        global.set(.max_tokens, .{ .integer = 100 }, false);
        global.set(.reasoning_budget, .{ .integer = 1024 }, false);
        const got = try apply(a, request.object, surface, global, .{});
        try t.expectEqual(@as(i64, 0), got.root.get("top_k").?.integer);
        const max_key = if (surface == .responses) "max_output_tokens" else "max_tokens";
        try t.expectEqual(@as(i64, 12), got.root.get(max_key).?.integer);
    }
}
