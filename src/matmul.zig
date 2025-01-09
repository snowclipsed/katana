const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const ops = @import("ops.zig");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const testing = std.testing;

// Tuning parameters
const T: usize = 64; // Tile size
const V: usize = 32; // Vector size

const WorkItem = struct {
    i: usize,
    j: usize,
};

const ThreadContext = struct {
    a: []const f32,
    b: []const f32,
    c: []f32,
    M: usize,
    N: usize,
    K: usize,
    work_queue: *ArrayList(WorkItem),
    mutex: std.Thread.Mutex,
};

fn tiledMultiplyKernel(
    A: []const f32,
    B: []const f32,
    local_C: *[T][T]f32,
    N: usize,
    K: usize,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    var A_local: [T][T]f32 = undefined;
    var B_local: [T][T]f32 = undefined;

    // Load A and B into local buffers
    for (0..T) |i| {
        for (0..T) |k| {
            if (i_start + i < i_end and k_start + k < k_end) {
                A_local[i][k] = A[(i_start + i) * K + (k_start + k)];
            } else {
                A_local[i][k] = 0;
            }
        }
    }

    for (0..T) |k| {
        for (0..T) |j| {
            if (k_start + k < k_end and j_start + j < j_end) {
                B_local[k][j] = B[(k_start + k) * N + (j_start + j)];
            } else {
                B_local[k][j] = 0;
            }
        }
    }

    // Compute tile with vectorization
    var i: usize = 0;
    while (i < T) : (i += 1) {
        var j: usize = 0;
        while (j < T) : (j += V) {
            var vec_sum: @Vector(V, f32) = @splat(0);
            var k: usize = 0;
            while (k < T) : (k += 1) {
                const a_val = A_local[i][k];
                const a_vec = @as(@Vector(V, f32), @splat(a_val));
                const b_vec = blk: {
                    var temp: @Vector(V, f32) = undefined;
                    for (0..V) |idx| {
                        temp[idx] = B_local[k][j + idx];
                    }
                    break :blk temp;
                };
                vec_sum += a_vec * b_vec;
            }

            // Store results in local buffer
            for (0..V) |idx| {
                local_C[i][j + idx] += vec_sum[idx];
            }
        }
    }
}

fn workerThread(context: *ThreadContext) void {
    while (true) {
        // Get next work item
        context.mutex.lock();
        const work_item = if (context.work_queue.popOrNull()) |item| item else {
            context.mutex.unlock();
            break;
        };
        context.mutex.unlock();

        // Process tile
        const i_start = work_item.i * T;
        const j_start = work_item.j * T;
        const i_end = @min(i_start + T, context.M);
        const j_end = @min(j_start + T, context.N);

        var local_C: [T][T]f32 = [_][T]f32{[_]f32{0} ** T} ** T;

        var k: usize = 0;
        while (k < context.K) : (k += T) {
            const k_end = @min(k + T, context.K);
            tiledMultiplyKernel(
                context.a,
                context.b,
                &local_C,
                context.N,
                context.K,
                i_start,
                j_start,
                k,
                i_end,
                j_end,
                k_end,
            );
        }

        // Accumulate results to global C
        for (i_start..i_end) |i| {
            for (j_start..j_end) |j| {
                context.c[i * context.N + j] += local_C[i - i_start][j - j_start];
            }
        }
    }
}

pub fn matmul(comptime DataType: type, a: Tensor(DataType), b: Tensor(DataType), allocator: Allocator) !Tensor(DataType) {
    if (a.shape.len != 2 or b.shape.len != 2) {
        return error.InvalidDimensions;
    }
    if (a.shape[1] != b.shape[0]) {
        return error.ShapeMismatch;
    }

    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(DataType).init(allocator, &result_shape);
    errdefer result.deinit();

    // Calculate number of tiles and create work items
    const tiles_M = (M + T - 1) / T;
    const tiles_N = (N + T - 1) / T;

    // Initialize work queue
    var work_queue = ArrayList(WorkItem).init(allocator);
    defer work_queue.deinit();

    // Populate work queue
    for (0..tiles_M) |i| {
        for (0..tiles_N) |j| {
            try work_queue.append(.{ .i = i, .j = j });
        }
    }

    // Shuffle work queue for better load balancing
    var rng = std.rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    rng.random().shuffle(WorkItem, work_queue.items);

    // Initialize thread pool
    const thread_count = try std.Thread.getCpuCount();
    var thread_pool = try ArrayList(std.Thread).initCapacity(allocator, thread_count);
    defer thread_pool.deinit();

    // Create thread context
    var context = ThreadContext{
        .a = a.getSlice(),
        .b = b.getSlice(),
        .c = result.getSlice(),
        .M = M,
        .N = N,
        .K = K,
        .work_queue = &work_queue,
        .mutex = std.Thread.Mutex{},
    };

    // Spawn worker threads
    for (0..thread_count) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, workerThread, .{&context}));
    }

    // Wait for all threads to complete
    for (thread_pool.items) |thread| {
        thread.join();
    }

    return result;
}

// test "optimized matmul benchmark" {
//     var arena = std.heap.ArenaAllocator.init(testing.allocator);
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     const sizes = [_]struct { m: usize, n: usize, k: usize }{
//         .{ .m = 256, .n = 256, .k = 256 },
//         .{ .m = 512, .n = 512, .k = 512 },
//         .{ .m = 1024, .n = 1024, .k = 1024 },
//         .{ .m = 1024, .n = 2048, .k = 1024 },
//         .{ .m = 2048, .n = 2048, .k = 2048 },
//     };
//     const iterations = 5;

//     std.debug.print("\nT = {d} \nV = {d}\n", .{ T, V });
//     std.debug.print("Number of threads = {d}\n", .{try std.Thread.getCpuCount()});

//     for (sizes) |size| {
//         const shape_a = [_]usize{ size.m, size.k };
//         const shape_b = [_]usize{ size.k, size.n };

//         var a = try Tensor(f32).init(allocator, &shape_a);
//         defer a.deinit();
//         var b = try Tensor(f32).init(allocator, &shape_b);
//         defer b.deinit();

//         // Initialize with random data
//         var prng = std.rand.DefaultPrng.init(0);
//         var random = prng.random();
//         for (a.data) |*val| val.* = random.float(f32);
//         for (b.data) |*val| val.* = random.float(f32);

//         // Warmup run
//         {
//             var warmup = try matmul(f32, a, b, allocator);
//             defer warmup.deinit();
//         }

//         var gflops_array = try allocator.alloc(f64, iterations);
//         defer allocator.free(gflops_array);

//         for (0..iterations) |i| {
//             var timer = try std.time.Timer.start();
//             var result = try matmul(f32, a, b, allocator);
//             defer result.deinit();
//             const elapsed_ns = timer.read();

//             const opers = 2.0 * @as(f64, @floatFromInt(size.m * size.n * size.k));
//             const seconds = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
//             gflops_array[i] = opers / seconds / 1e9;
//         }

//         // Calculate average GFLOPS
//         var total_gflops: f64 = 0;
//         for (gflops_array) |gflops| {
//             total_gflops += gflops;
//         }
//         const avg_gflops = total_gflops / @as(f64, @floatFromInt(iterations));

//         std.debug.print("Matrix size: {d}x{d}x{d}, GFLOPS: {d:.2}\n", .{ size.m, size.n, size.k, avg_gflops });
//     }
// }

test "matmul basic functionality" {
    const allocator = testing.allocator;

    // Test case 1: Square matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;
        a.data[3] = 4.0;

        b.data[0] = 5.0;
        b.data[1] = 6.0;
        b.data[2] = 7.0;
        b.data[3] = 8.0;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Compare with known result
        try testing.expectApproxEqAbs(result.data[0], 19.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[1], 22.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[2], 43.0, 1e-6);
        try testing.expectApproxEqAbs(result.data[3], 50.0, 1e-6);
    }
}

test "matmul edge cases" {
    const allocator = testing.allocator;

    // Test case 1: 1x1 matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer b.deinit();

        a.data[0] = 3.0;
        b.data[0] = 4.0;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectApproxEqAbs(result.data[0], 12.0, 1e-6);
    }

    // Test case 2: Tall matrix × Wide matrix
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 3, 1 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
        defer b.deinit();

        a.data[0] = 1.0;
        a.data[1] = 2.0;
        a.data[2] = 3.0;

        b.data[0] = 4.0;
        b.data[1] = 5.0;
        b.data[2] = 6.0;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], @as(usize, 3));
        try testing.expectEqual(result.shape[1], @as(usize, 3));
    }

    // Test case 3: Zero matrices
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        @memset(a.data, 0);
        @memset(b.data, 0);

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        for (result.data) |val| {
            try testing.expectEqual(val, 0);
        }
    }
}

test "matmul error cases" {
    const allocator = testing.allocator;

    // Test case 1: Mismatched dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.ShapeMismatch, matmul(f32, a, b, allocator));
    }

    // Test case 2: Invalid dimensions
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{2});
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        try testing.expectError(error.InvalidDimensions, matmul(f32, a, b, allocator));
    }
}

test "matmul correctness against reference" {
    const allocator = testing.allocator;
    const test_sizes = [_][3]usize{
        .{ 32, 32, 32 }, // Small square
        .{ 47, 35, 23 }, // Odd sizes
        .{ 128, 64, 96 }, // Rectangular
        .{ 1, 64, 128 }, // Single row × wide
        .{ 128, 1, 64 }, // Tall × single column
        .{ T - 1, T + 1, T }, // Around tile size
        .{ T, T, T }, // Exactly tile size
        .{ T + 1, T - 1, T }, // Around tile size
    };

    for (test_sizes) |size| {
        const M = size[0];
        const N = size[1];
        const K = size[2];

        // Create random input tensors
        var a = try ops.randomTensor(f32, allocator, &[_]usize{ M, K }, 42);
        defer a.deinit();
        var b = try ops.randomTensor(f32, allocator, &[_]usize{ K, N }, 43);
        defer b.deinit();

        // Compute using tiled matmul
        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Compute using reference matmul
        var expected = try ops.matmul(f32, &a, b);
        defer expected.deinit();

        // Compare results
        const eps: f32 = 1e-4; // Allow for some floating-point error
        for (result.data, expected.data) |val, exp| {
            try testing.expectApproxEqAbs(val, exp, eps);
        }

        std.debug.print("Test passed for size: M={}, N={}, K={}\n", .{ M, N, K });
    }
}

test "matmul numerical stability" {
    const allocator = testing.allocator;

    // Test case 1: Moderately large numbers
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // Using smaller values to avoid overflow
        const large: f32 = 1e3;
        @memset(a.data, large);
        @memset(b.data, large);

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            // Verify no infinity and reasonable magnitude
            try testing.expect(!std.math.isInf(val));
            try testing.expect(!std.math.isNan(val));
            // For 2x2 matrices filled with 1e3, each element should be 2 * (1e3 * 1e3) = 2e6
            try testing.expectApproxEqAbs(val, 2e6, 1e-6);
        }
    }

    // Test case 2: Small but non-zero numbers
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        const small: f32 = 1e-3;
        @memset(a.data, small);
        @memset(b.data, small);

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            try testing.expect(!std.math.isNan(val));
            try testing.expect(val > 0); // Should be positive
            // For 2x2 matrices filled with 1e-3, each element should be 2 * (1e-3 * 1e-3) = 2e-6
            try testing.expectApproxEqAbs(val, 2e-6, 1e-9);
        }
    }

    // Test case 3: Mixed magnitudes
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer b.deinit();

        // First row large, second row small
        a.data[0] = 1e3;
        a.data[1] = 1e3;
        a.data[2] = 1e-3;
        a.data[3] = 1e-3;

        b.data[0] = 1e-3;
        b.data[1] = 1e3;
        b.data[2] = 1e-3;
        b.data[3] = 1e3;

        var result = try matmul(f32, a, b, allocator);
        defer result.deinit();

        // Check results
        for (result.data) |val| {
            try testing.expect(!std.math.isInf(val));
            try testing.expect(!std.math.isNan(val));
        }
    }
}
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
    try std.io.getStdOut().writer().print("T = {d} \nV = {d} \n", .{ T, V });
    try std.io.getStdOut().writer().print("Number of threads = {d}\n", .{try std.Thread.getCpuCount()});

    for (sizes) |size| {
        const avg_gflops = try calculateGflops(allocator, size.m, size.n, size.k, iterations);
        try std.io.getStdOut().writer().print("Matrix size: {d}x{d}x{d}, GFLOPS: {d:.2}\n", .{ size.m, size.n, size.k, avg_gflops });
    }
}
