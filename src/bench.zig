const std = @import("std");
const tensormod = @import("tensor.zig");
const Tensor = tensormod.Tensor;
const ops = @import("ops.zig");
const matmul = ops.matmul;
const T = ops.Tile;

pub fn calculateGflops(allocator: std.mem.Allocator, M: usize, N: usize, K: usize, iterations: usize) !f64 {
    const shape_a = [_]usize{ M, K };
    const shape_b = [_]usize{ K, N };

    var a = try Tensor(f32).init(allocator, &shape_a);
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &shape_b);
    defer b.deinit();

    // Initialize with random data
    var prng = std.rand.DefaultPrng.init(0);
    var random = prng.random();
    for (a.data) |*val| val.* = random.float(f32);
    for (b.data) |*val| val.* = random.float(f32);

    // Warmup run
    {
        var warmup = try matmul(f32, a, b, allocator);
        defer warmup.deinit();
    }

    var gflops_array = try allocator.alloc(f64, iterations);
    defer allocator.free(gflops_array);

    for (0..iterations) |i| {
        var timer = try std.time.Timer.start();
        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();
        const elapsed_ns = timer.read();

        const opers = 2 * M * N * K; // multiply-add is 2 operations
        const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
        gflops_array[i] = @as(f64, @floatFromInt(opers)) / seconds / 1e9;
    }

    // Calculate average GFLOPS
    var total_gflops: f64 = 0;
    for (gflops_array) |gflops| {
        total_gflops += gflops;
    }
    return total_gflops / @as(f64, @floatFromInt(iterations));
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Define test sizes
    const sizes = [_]struct { m: usize, n: usize, k: usize }{
        .{ .m = 256, .n = 256, .k = 256 },
        .{ .m = 512, .n = 512, .k = 512 },
        .{ .m = 1024, .n = 1024, .k = 1024 },
        .{ .m = 1024, .n = 2048, .k = 1024 },
        .{ .m = 2048, .n = 2048, .k = 2048 },
        .{ .m = 2048, .n = 4096, .k = 2048 },
        .{ .m = 4096, .n = 4096, .k = 4096 },
        .{ .m = 8192, .n = 2048, .k = 8192 },
        .{ .m = 1152, .n = 4304, .k = 1152 },
    };

    const iterations = 5;

    try std.io.getStdOut().writer().print("\nRunning MatMul Benchmark\n", .{});
    try std.io.getStdOut().writer().print("T = {d} \n", .{T});
    try std.io.getStdOut().writer().print("Number of threads = {d}\n", .{try std.Thread.getCpuCount()});

    for (sizes) |size| {
        const avg_gflops = try calculateGflops(allocator, size.m, size.n, size.k, iterations);
        try std.io.getStdOut().writer().print("Matrix size: {d}x{d}x{d}, GFLOPS: {d:.2}\n", .{ size.m, size.n, size.k, avg_gflops });
    }
}
