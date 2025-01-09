const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating
const Tensor = @import("tensor.zig").Tensor;
const Slice = @import("tensor.zig").Slice;
const ops = @import("ops.zig");
const StabilityError = @import("tensor.zig").StabilityError;
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;
const expectApproxEqAbs = testing.expectApproxEqAbs;

test "tensor basic operations" {
    const allocator = testing.allocator;

    // Test initialization
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();

    try testing.expectEqual(@as(usize, 2), tensor1.shape[0]);
    try testing.expectEqual(@as(usize, 3), tensor1.shape[1]);

    // Test fill - assuming fill remains in core tensor functionality
    tensor1.fill(2.0);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 2.0), value);
    }

    // Test scalar operations using the ops module
    ops.scalarAdd(f32, &tensor1, 1.0);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 3.0), value);
    }

    ops.scalarMultiply(f32, &tensor1, 2.0);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 6.0), value);
    }

    // Additional tests for other ops
    var tensor2 = try tensor1.copy();
    defer tensor2.deinit();

    try ops.add(f32, &tensor1, tensor2);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 12.0), value); // 6.0 + 6.0 = 12.0
    }

    try ops.subtract(f32, &tensor1, tensor2);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 6.0), value); // 12.0 - 6.0 = 6.0
    }
}

test "tensor element-wise operations" {
    const allocator = testing.allocator;

    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer tensor1.deinit();
    tensor1.fill(2.0);

    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer tensor2.deinit();
    tensor2.fill(3.0);

    try ops.add(f32, &tensor1, tensor2);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 5.0), value);
    }

    try ops.subtract(f32, &tensor1, tensor2);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 2.0), value);
    }

    try ops.multiply(f32, &tensor1, tensor2);
    for (tensor1.data) |value| {
        try testing.expectEqual(@as(f32, 6.0), value);
    }
}

test "tensor reshape" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    try tensor.reshape(&[_]usize{ 3, 2 });
    try testing.expectEqual(@as(usize, 3), tensor.shape[0]);
    try testing.expectEqual(@as(usize, 2), tensor.shape[1]);

    // Test incompatible reshape
    try testing.expectError(error.IncompatibleShape, tensor.reshape(&[_]usize{ 2, 4 }));
}

test "tensor transpose" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with test data
    for (tensor.data, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(i));
    }

    try ops.transpose(f32, &tensor);

    try testing.expectEqual(@as(usize, 3), tensor.shape[0]);
    try testing.expectEqual(@as(usize, 2), tensor.shape[1]);

    // Verify transpose
    try testing.expectEqual(@as(f32, 0.0), tensor.data[0]);
    try testing.expectEqual(@as(f32, 3.0), tensor.data[1]);
    try testing.expectEqual(@as(f32, 1.0), tensor.data[2]);
    try testing.expectEqual(@as(f32, 4.0), tensor.data[3]);
    try testing.expectEqual(@as(f32, 2.0), tensor.data[4]);
    try testing.expectEqual(@as(f32, 5.0), tensor.data[5]);
}

test "tensor matrix multiplication" {
    const allocator = testing.allocator;

    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();
    tensor1.fill(2.0);

    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer tensor2.deinit();
    tensor2.fill(3.0);

    var result = try ops.matmul(f32, &tensor1, tensor2);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);

    // Each element should be 2 * 3 * 3 = 18 (dot product of row and column)
    for (result.data) |value| {
        try testing.expectEqual(@as(f32, 18.0), value);
    }
}

test "tensor copy" {
    const allocator = testing.allocator;

    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer tensor1.deinit();
    tensor1.fill(2.0);

    var tensor2 = try tensor1.copy();
    defer tensor2.deinit();

    // Verify copy
    try testing.expectEqualSlices(usize, tensor1.shape, tensor2.shape);
    try testing.expectEqualSlices(f32, tensor1.data, tensor2.data);

    // Verify independence
    tensor1.fill(3.0);
    try testing.expectEqual(@as(f32, 2.0), tensor2.data[0]);
}

test "complex reshape operations" {
    const allocator = testing.allocator;

    // Test 1: Complex shape transformations
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
        defer tensor.deinit();

        // Fill with sequential values
        for (tensor.data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i));
        }

        // Test multiple reshapes in sequence
        try tensor.reshape(&[_]usize{ 4, 6 });
        try tensor.reshape(&[_]usize{ 3, 8 });
        try tensor.reshape(&[_]usize{ 24, 1 });
        try tensor.reshape(&[_]usize{ 1, 24 });

        // Verify data remains in correct order
        for (tensor.data, 0..) |value, i| {
            try expectEqual(@as(f32, @floatFromInt(i)), value);
        }
    }

    // Test 2: Edge case - reshape to same total size but different dimension count
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 4, 3 });
        defer tensor.deinit();

        for (tensor.data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i));
        }

        // Reshape to more dimensions
        try tensor.reshape(&[_]usize{ 2, 2, 3 });
        try expectEqual(@as(usize, 3), tensor.shape.len);
        try expectEqual(@as(usize, 12), tensor.data.len);

        // Reshape back to fewer dimensions
        try tensor.reshape(&[_]usize{12});
        try expectEqual(@as(usize, 1), tensor.shape.len);

        // Verify data preservation
        for (tensor.data, 0..) |value, i| {
            try expectEqual(@as(f32, @floatFromInt(i)), value);
        }
    }

    // Test 3: Zero dimension handling
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{0});
        defer tensor.deinit();

        try tensor.reshape(&[_]usize{ 0, 0 });
        try expectEqual(@as(usize, 0), tensor.data.len);

        // Should fail when trying to reshape to non-zero size
        try expectError(error.IncompatibleShape, tensor.reshape(&[_]usize{1}));
    }

    // Test 4: Maximum size reshaping
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 1024, 1024 });
        defer tensor.deinit();

        // Reshape to many small dimensions
        const many_dims = [_]usize{ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }; // 2^20
        try tensor.reshape(&many_dims);
        try expectEqual(@as(usize, 20), tensor.shape.len);
        try expectEqual(@as(usize, 1024 * 1024), tensor.data.len);

        // Reshape back to 2D
        try tensor.reshape(&[_]usize{ 1024, 1024 });
        try expectEqual(@as(usize, 2), tensor.shape.len);
    }
}

test "tensor unsqueeze" {
    const allocator = testing.allocator;
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with test data
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Test unsqueeze in middle
    try tensor.unsqueeze(1);
    try testing.expectEqual(@as(usize, 3), tensor.shape.len);
    try testing.expectEqual(@as(usize, 2), tensor.shape[0]);
    try testing.expectEqual(@as(usize, 1), tensor.shape[1]);
    try testing.expectEqual(@as(usize, 3), tensor.shape[2]);

    // Test unsqueeze at end with negative index
    try tensor.unsqueeze(-1);
    try testing.expectEqual(@as(usize, 4), tensor.shape.len);
    try testing.expectEqual(@as(usize, 2), tensor.shape[0]);
    try testing.expectEqual(@as(usize, 1), tensor.shape[1]);
    try testing.expectEqual(@as(usize, 3), tensor.shape[2]);
    try testing.expectEqual(@as(usize, 1), tensor.shape[3]);

    // Verify data unchanged
    try testing.expectEqual(@as(f32, 0), tensor.data[0]);
    try testing.expectEqual(@as(f32, 1), tensor.data[1]);
}

test "tensor unsqueeze edge cases" {
    const allocator = testing.allocator;
    var tensor = try Tensor(f32).init(allocator, &[_]usize{5});
    defer tensor.deinit();

    // Test unsqueeze on 1D tensor
    try tensor.unsqueeze(0);
    try testing.expectEqual(@as(usize, 2), tensor.shape.len);
    try testing.expectEqual(@as(usize, 1), tensor.shape[0]);
    try testing.expectEqual(@as(usize, 5), tensor.shape[1]);

    // Test error cases
    try testing.expectError(error.InvalidDimension, tensor.unsqueeze(3)); // Too large dimension
    try testing.expectError(error.InvalidDimension, tensor.unsqueeze(-4)); // Too negative dimension
}

// Tests
test "getSliceRange - basic 2D slicing" {
    const allocator = testing.allocator;

    // Create a 3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Test slicing first two columns
    var slices = [_]Slice{
        Slice.full(),
        Slice.from(0, 2),
    };

    var result = try tensor.getSliceRange(&slices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectApproxEqAbs(@as(f32, 0), result.data[0], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1), result.data[1], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 4), result.data[2], 0.001);
    try testing.expectApproxEqAbs(@as(f32, 5), result.data[3], 0.001);
}

test "getSliceRange - 3D slicing with full dimensions" {
    const allocator = testing.allocator;

    // Create a 2x3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Test slicing with full range in first dimension
    var slices = [_]Slice{
        Slice.full(),
        Slice.from(1, 3),
        Slice.from(0, 2),
    };

    var result = try tensor.getSliceRange(&slices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 2), result.shape[2]);
}

test "getSliceRange - error cases" {
    const allocator = testing.allocator;

    // Create a 2x3 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Test too many slices
    var too_many_slices = [_]Slice{
        Slice.full(),
        Slice.full(),
        Slice.full(),
    };
    try testing.expectError(error.TooManySlices, tensor.getSliceRange(&too_many_slices));

    // Test out of bounds
    var out_of_bounds = [_]Slice{
        Slice.from(0, 3),
        Slice.from(0, 4),
    };
    try testing.expectError(error.SliceOutOfBounds, tensor.getSliceRange(&out_of_bounds));

    // Test invalid slice (start > end)
    var invalid_slice = [_]Slice{
        Slice.from(2, 1),
        Slice.full(),
    };
    try testing.expectError(error.InvalidSlice, tensor.getSliceRange(&invalid_slice));
}

test "getSliceRange - partial slicing" {
    const allocator = testing.allocator;

    // Create a 3x4x5 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 4, 5 });
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Test partial slicing (only specify first two dimensions)
    var slices = [_]Slice{
        Slice.from(1, 3),
        Slice.from(0, 2),
    };

    var result = try tensor.getSliceRange(&slices);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.shape[0]);
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 5), result.shape[2]);
}

test "complex transpose operations" {
    const allocator = testing.allocator;

    // Test 1: Transpose with non-uniform values
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 4 });
        defer tensor.deinit();

        // Fill with non-uniform pattern
        const pattern = [_]f32{
            1.0, -2.0,  3.0,  -4.0,
            5.0, -6.0,  7.0,  -8.0,
            9.0, -10.0, 11.0, -12.0,
        };
        @memcpy(tensor.data, &pattern);

        try ops.transpose(f32, &tensor);

        // Expected result after transpose
        const expected = [_]f32{
            1.0,  5.0,  9.0,
            -2.0, -6.0, -10.0,
            3.0,  7.0,  11.0,
            -4.0, -8.0, -12.0,
        };

        try expectEqual(@as(usize, 4), tensor.shape[0]);
        try expectEqual(@as(usize, 3), tensor.shape[1]);
        try testing.expectEqualSlices(f32, &expected, tensor.data);
    }

    // Test 2: Multiple transpose operations
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 5 });
        defer tensor.deinit();

        for (tensor.data, 0..) |*value, i| {
            value.* = @as(f32, @floatFromInt(i));
        }

        // Perform multiple transposes
        try ops.transpose(f32, &tensor); // 5x2
        try ops.transpose(f32, &tensor); // 2x5
        try ops.transpose(f32, &tensor); // 5x2
        try ops.transpose(f32, &tensor); // 2x5

        // Should be back to original
        const expected = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        try testing.expectEqualSlices(f32, &expected, tensor.data);
    }

    // Test 3: Transpose with identical dimensions
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 3 });
        defer tensor.deinit();

        const pattern = [_]f32{
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        };
        @memcpy(tensor.data, &pattern);

        try ops.transpose(f32, &tensor);

        const expected = [_]f32{
            1, 4, 7,
            2, 5, 8,
            3, 6, 9,
        };
        try testing.expectEqualSlices(f32, &expected, tensor.data);
    }

    // Test 4: Transpose of single-row and single-column matrices
    {
        // Single row
        var row_tensor = try Tensor(f32).init(allocator, &[_]usize{ 1, 5 });
        defer row_tensor.deinit();

        const row_data = [_]f32{ 1, 2, 3, 4, 5 };
        @memcpy(row_tensor.data, &row_data);

        try ops.transpose(f32, &row_tensor);
        try expectEqual(@as(usize, 5), row_tensor.shape[0]);
        try expectEqual(@as(usize, 1), row_tensor.shape[1]);

        // Single column
        var col_tensor = try Tensor(f32).init(allocator, &[_]usize{ 5, 1 });
        defer col_tensor.deinit();

        const col_data = [_]f32{ 1, 2, 3, 4, 5 };
        @memcpy(col_tensor.data, &col_data);

        try ops.transpose(f32, &col_tensor);
        try expectEqual(@as(usize, 1), col_tensor.shape[0]);
        try expectEqual(@as(usize, 5), col_tensor.shape[1]);
    }

    // Test 5: Error cases for transpose
    {
        // 1D tensor
        var tensor_1d = try Tensor(f32).init(allocator, &[_]usize{5});
        defer tensor_1d.deinit();
        try expectError(error.UnsupportedDimension, ops.transpose(f32, &tensor_1d));

        // 3D tensor
        var tensor_3d = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 2 });
        defer tensor_3d.deinit();
        try expectError(error.UnsupportedDimension, ops.transpose(f32, &tensor_3d));
    }
}

test "Tensor accumulate basic functionality" {
    const allocator = testing.allocator;

    // Test 1: Basic 1D accumulation
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{6});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{6});
        defer t1.deinit();
        defer t2.deinit();

        // Fill with simple sequence
        for (t1.data, 0..) |*val, i| {
            val.* = @floatFromInt(i + 1); // [1,2,3,4,5,6]
        }
        for (t2.data) |*val| {
            val.* = 1; // [1,1,1,1,1,1]
        }

        try ops.accumulate(f32, &t1, t2);

        // Expected: [2,5,9,14,20,27]
        try testing.expectApproxEqAbs(t1.data[0], 2.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[1], 5.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[5], 27.0, 0.001);
    }

    // Test 2: 2D tensor accumulation
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer t1.deinit();
        defer t2.deinit();

        // Initialize t1 with increasing values
        for (t1.data, 0..) |*val, i| {
            val.* = @floatFromInt(i + 1); // [1,2,3,4,5,6]
        }
        t2.fill(2.0); // Fill t2 with constant value

        try ops.accumulate(f32, &t1, t2);

        // Expected running sum: [3,7,12,18,25,33]
        try testing.expectApproxEqAbs(t1.data[0], 3.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[2], 12.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[5], 33.0, 0.001);
    }

    // Test 3: Different data types
    {
        var t1 = try Tensor(i32).init(allocator, &[_]usize{4});
        var t2 = try Tensor(i32).init(allocator, &[_]usize{4});
        defer t1.deinit();
        defer t2.deinit();

        t1.fill(1);
        t2.fill(1);

        try ops.accumulate(i32, &t1, t2);

        // Expected: [2,4,6,8]
        try testing.expectEqual(t1.data[0], 2);
        try testing.expectEqual(t1.data[1], 4);
        try testing.expectEqual(t1.data[3], 8);
    }

    // Test 4: Large tensor stress test
    {
        const size = 1000;
        var t1 = try Tensor(f32).init(allocator, &[_]usize{size});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{size});
        defer t1.deinit();
        defer t2.deinit();

        t1.fill(1.0);
        t2.fill(1.0);

        try ops.accumulate(f32, &t1, t2);

        // Check first, middle and last elements
        try testing.expectApproxEqAbs(t1.data[0], 2.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[size - 1], @as(f32, @floatFromInt(size * 2)), 0.001);
    }

    // Test 5: Error cases
    {
        // Shape mismatch
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer t1.deinit();
        defer t2.deinit();

        try testing.expectError(error.ShapeMismatch, ops.accumulate(f32, &t1, t2));
    }

    // Test 6: Zero and negative numbers
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{4});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{4});
        defer t1.deinit();
        defer t2.deinit();

        t1.data[0] = -1.0;
        t1.data[1] = 0.0;
        t1.data[2] = 1.0;
        t1.data[3] = -2.0;

        t2.data[0] = 1.0;
        t2.data[1] = -1.0;
        t2.data[2] = 0.0;
        t2.data[3] = 2.0;

        try ops.accumulate(f32, &t1, t2);

        // Expected: [0,-1,0,0]
        try testing.expectApproxEqAbs(t1.data[0], 0.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[1], -1.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[2], 0.0, 0.001);
        try testing.expectApproxEqAbs(t1.data[3], 0.0, 0.001);
    }

    // Test 7: Small tensors edge case
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{1});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{1});
        defer t1.deinit();
        defer t2.deinit();

        t1.fill(2.0);
        t2.fill(3.0);

        try ops.accumulate(f32, &t1, t2);

        try testing.expectApproxEqAbs(t1.data[0], 5.0, 0.001);
    }
}

test "Tensor concatenation edge cases" {
    const allocator = testing.allocator;

    // Test 1: Basic zero dimension handling
    {
        const M: usize = 0;
        const N: usize = 2;
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ M, N });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 0, 2 });
        defer t1.deinit();
        defer t2.deinit();

        // Concatenate along first dimension
        var result1 = try ops.concat(f32, t1, t2, 0);
        defer result1.deinit();

        try testing.expectEqual(result1.shape[0], 0);
        try testing.expectEqual(result1.shape[1], 2);

        // Concatenate along second dimension
        var result2 = try ops.concat(f32, t1, t2, 1);
        defer result2.deinit();

        try testing.expectEqual(result2.shape[0], 0);
        try testing.expectEqual(result2.shape[1], 4);
    }

    // Test 2: Mixed zero and non-zero dimensions
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 0, 3 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 0, 3 });
        defer t1.deinit();
        defer t2.deinit();

        // Concatenate along first dimension
        var result = try ops.concat(f32, t1, t2, 0);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], 4);
        try testing.expectEqual(result.shape[1], 0);
        try testing.expectEqual(result.shape[2], 3);
    }

    // Test 3: Regular tensor concatenation
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer t1.deinit();
        defer t2.deinit();

        // Fill tensors with test data
        for (t1.data, 0..) |*val, i| {
            val.* = @floatFromInt(i + 1);
        }
        for (t2.data, 0..) |*val, i| {
            val.* = @floatFromInt(i + 7);
        }

        var result = try ops.concat(f32, t1, t2, 0);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], 4);
        try testing.expectEqual(result.shape[1], 3);

        // Check some values
        try testing.expectApproxEqAbs(result.data[0], 1.0, 0.001);
        try testing.expectApproxEqAbs(result.data[5], 6.0, 0.001);
        try testing.expectApproxEqAbs(result.data[6], 7.0, 0.001);
        try testing.expectApproxEqAbs(result.data[11], 12.0, 0.001);
    }

    // Test 4: High dimensional tensor
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 2 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 2 });
        defer t1.deinit();
        defer t2.deinit();

        t1.fill(1.0);
        t2.fill(2.0);

        var result = try ops.concat(f32, t1, t2, 1);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], 2);
        try testing.expectEqual(result.shape[1], 4);
        try testing.expectEqual(result.shape[2], 2);

        // Check first element from first tensor
        try testing.expectApproxEqAbs(result.data[0], 1.0, 0.001);
        // Check first element from second tensor
        try testing.expectApproxEqAbs(result.data[4], 2.0, 0.001);
    }

    // Test 5: Large dimensions
    {
        const size: usize = 1000;
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ size, 2 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ size, 2 });
        defer t1.deinit();
        defer t2.deinit();

        t1.fill(1.0);
        t2.fill(2.0);

        var result = try ops.concat(f32, t1, t2, 0);
        defer result.deinit();

        try testing.expectEqual(result.shape[0], size * 2);
        try testing.expectEqual(result.shape[1], 2);

        // Check values at boundaries
        try testing.expectApproxEqAbs(result.data[0], 1.0, 0.001);
        try testing.expectApproxEqAbs(result.data[size * 2], 2.0, 0.001);
    }

    // Test 6: Last dimension concatenation
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 3 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 3 });
        defer t1.deinit();
        defer t2.deinit();

        // Fill tensors with different values
        t1.fill(1.0);
        t2.fill(2.0);

        // Concatenate along last dimension
        var result = try ops.concat(f32, t1, t2, 2);
        defer result.deinit();

        // Check dimensions
        try testing.expectEqual(result.shape[0], 2);
        try testing.expectEqual(result.shape[1], 2);
        try testing.expectEqual(result.shape[2], 6);

        // Check values from first tensor
        try testing.expectApproxEqAbs(result.data[0], 1.0, 0.001);
        try testing.expectApproxEqAbs(result.data[2], 1.0, 0.001);
        // Check values from second tensor
        try testing.expectApproxEqAbs(result.data[3], 2.0, 0.001);
        try testing.expectApproxEqAbs(result.data[5], 2.0, 0.001);
    }

    // Test 7: Error cases
    {
        // Test shape mismatch
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 4 });
        defer t1.deinit();
        defer t2.deinit();

        try testing.expectError(error.IncompatibleShapes, ops.concat(f32, t1, t2, 0));

        // Test invalid dimension
        var t3 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        var t4 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer t3.deinit();
        defer t4.deinit();

        try testing.expectError(error.InvalidDimension, ops.concat(f32, t3, t4, 2));
    }
}

test "stack basic functionality" {
    const allocator = testing.allocator;

    // Create two 2x3 tensors
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor2.deinit();

    // Fill tensors with test data
    for (tensor1.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (tensor2.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 10);
    }

    // Stack along dimension 0
    var tensors = [_]Tensor(f32){ tensor1, tensor2 };
    var result = try ops.stack(f32, &tensors, 0);
    defer result.deinit();

    // Check result shape
    try testing.expectEqual(@as(usize, 3), result.shape.len);
    try testing.expectEqual(@as(usize, 2), result.shape[0]); // Number of stacked tensors
    try testing.expectEqual(@as(usize, 2), result.shape[1]);
    try testing.expectEqual(@as(usize, 3), result.shape[2]);

    // Check values
    try testing.expectEqual(@as(f32, 0), result.data[0]);
    try testing.expectEqual(@as(f32, 10), result.data[6]);
}

test "stack error cases" {
    const allocator = testing.allocator;

    // Create tensors with different shapes
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 4 });
    defer tensor2.deinit();

    // Test shape mismatch
    var tensors_mismatched = [_]Tensor(f32){ tensor1, tensor2 };
    try testing.expectError(error.ShapeMismatch, ops.stack(f32, &tensors_mismatched, 0));

    // Test invalid dimension
    var tensors_valid = [_]Tensor(f32){ tensor1, tensor1 };
    try testing.expectError(error.InvalidDimension, ops.stack(f32, &tensors_valid, 3));

    // Test empty tensor list
    var empty_tensors = [_]Tensor(f32){};
    try testing.expectError(error.EmptyTensorList, ops.stack(f32, &empty_tensors, 0));
}

test "stack along different dimensions" {
    const allocator = testing.allocator;

    // Create two 2x3 tensors
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor2.deinit();

    // Fill with test data
    for (tensor1.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (tensor2.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 10);
    }

    var tensors = [_]Tensor(f32){ tensor1, tensor2 };

    // Stack along different dimensions and verify shapes
    {
        var result_dim0 = try ops.stack(f32, &tensors, 0);
        defer result_dim0.deinit();
        try testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 3 }, result_dim0.shape);
    }
    {
        var result_dim1 = try ops.stack(f32, &tensors, 1);
        defer result_dim1.deinit();
        try testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 3 }, result_dim1.shape);
    }
    {
        var result_dim2 = try ops.stack(f32, &tensors, 2);
        defer result_dim2.deinit();
        try testing.expectEqualSlices(usize, &[_]usize{ 2, 3, 2 }, result_dim2.shape);
    }
}

test "normalizeDim basic functionality" {
    // Test positive dimensions
    try expectEqual(@as(usize, 0), try ops.normalizeDim(0, 3));
    try expectEqual(@as(usize, 2), try ops.normalizeDim(2, 3));

    // Test negative dimensions
    try expectEqual(@as(usize, 2), try ops.normalizeDim(-1, 3)); // -1 in 3 dims = index 2
    try expectEqual(@as(usize, 1), try ops.normalizeDim(-2, 3)); // -2 in 3 dims = index 1
    try expectEqual(@as(usize, 0), try ops.normalizeDim(-3, 3)); // -3 in 3 dims = index 0

    // Test error cases
    try expectError(error.InvalidDimension, ops.normalizeDim(3, 3)); // Too large positive
    try expectError(error.InvalidDimension, ops.normalizeDim(-4, 3)); // Too large negative
}

test "flatten basic functionality" {
    const allocator = testing.allocator;

    // Create a 2x3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Test flattening last two dimensions
    try tensor.flatten(1, 2);

    // Check new shape: should be [2, 12]
    try expectEqual(@as(usize, 2), tensor.shape.len);
    try expectEqual(@as(usize, 2), tensor.shape[0]);
    try expectEqual(@as(usize, 12), tensor.shape[1]);

    // Verify data remains in correct order
    try expectEqual(@as(f32, 0), tensor.data[0]);
    try expectEqual(@as(f32, 5), tensor.data[5]);
    try expectEqual(@as(f32, 12), tensor.data[12]);
}

test "flatten with negative dimensions" {
    const allocator = testing.allocator;

    // Create a 2x3x4x5 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4, 5 });
    defer tensor.deinit();

    // Test with negative dimensions
    try tensor.flatten(-3, -2); // Should flatten dimensions 1 and 2

    // Check new shape: should be [2, 12, 5]
    try expectEqual(@as(usize, 3), tensor.shape.len);
    try expectEqual(@as(usize, 2), tensor.shape[0]);
    try expectEqual(@as(usize, 12), tensor.shape[1]);
    try expectEqual(@as(usize, 5), tensor.shape[2]);
}

test "flatten error cases" {
    const allocator = testing.allocator;

    // Create a 2x3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    // Test invalid dimension range
    try expectError(error.InvalidDimRange, tensor.flatten(2, 1));

    // Test out of bounds dimensions
    try expectError(error.InvalidDimension, tensor.flatten(3, 4));
    try expectError(error.InvalidDimension, tensor.flatten(-4, -1));
}

test "flatten entire tensor" {
    const allocator = testing.allocator;

    // Create a 2x3x4 tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    // Fill with sequential values
    for (tensor.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    // Flatten entire tensor
    try tensor.flatten(0, 2);

    // Check new shape: should be [24]
    try expectEqual(@as(usize, 1), tensor.shape.len);
    try expectEqual(@as(usize, 24), tensor.shape[0]);

    // Verify data order
    try expectEqual(@as(f32, 0), tensor.data[0]);
    try expectEqual(@as(f32, 23), tensor.data[23]);
}

test "stackAndFlatten 2D tensors with different dimensions" {
    const allocator = testing.allocator;

    // Create two 2x3 tensors
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor2.deinit();

    // Fill with test data
    for (tensor1.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (tensor2.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 10);
    }

    // Test stacking at different dimensions
    {
        // Stack at beginning: [2,3] -> [2,2,3] -> [2,6]
        var result = try ops.stackAndFlatten(f32, tensor1, tensor2, 0);
        defer result.deinit();
        try expectEqual(@as(usize, 2), result.shape.len);
        try expectEqual(@as(usize, 2), result.shape[0]);
        try expectEqual(@as(usize, 6), result.shape[1]);
    }

    {
        // Stack at end: [2,3] -> [2,3,2] -> [2,6]
        var result = try ops.stackAndFlatten(f32, tensor1, tensor2, -1);
        defer result.deinit();
        try expectEqual(@as(usize, 2), result.shape.len);
        try expectEqual(@as(usize, 2), result.shape[0]);
        try expectEqual(@as(usize, 6), result.shape[1]);
    }

    {
        // Stack in middle: [2,3] -> [2,2,3] -> [2,6]
        var result = try ops.stackAndFlatten(f32, tensor1, tensor2, 1);
        defer result.deinit();
        try expectEqual(@as(usize, 2), result.shape.len);
        try expectEqual(@as(usize, 2), result.shape[0]);
        try expectEqual(@as(usize, 6), result.shape[1]);
    }
}

test "stackAndFlatten 3D tensors" {
    const allocator = testing.allocator;

    // Create two 2x3x4 tensors
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor2.deinit();

    // Fill with test data
    for (tensor1.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (tensor2.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 100);
    }

    // Test stacking along last dimension
    var result = try ops.stackAndFlatten(f32, tensor1, tensor2, -1);
    defer result.deinit();

    // Check shape: should be [2, 3, 8]
    try expectEqual(@as(usize, 3), result.shape.len);
    try expectEqual(@as(usize, 2), result.shape[0]);
    try expectEqual(@as(usize, 3), result.shape[1]);
    try expectEqual(@as(usize, 8), result.shape[2]);

    // Verify data ordering
    try expectEqual(@as(f32, 0), result.data[0]); // First element from tensor1
    try expectEqual(@as(f32, 100), result.data[1]); // First element from tensor2
}

test "stackAndFlatten error cases" {
    const allocator = testing.allocator;

    // Create tensors with different shapes
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 2, 4 });
    defer tensor2.deinit();

    // Test shape mismatch
    try expectError(error.ShapeMismatch, ops.stackAndFlatten(f32, tensor1, tensor2, -1));

    // Test invalid dimension
    var tensor3 = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor3.deinit();
    try expectError(error.InvalidDimension, ops.stackAndFlatten(f32, tensor1, tensor3, 3));
}

test "stackAndFlatten with singleton dimensions" {
    const allocator = testing.allocator;

    // Create two 1x3 tensors
    var tensor1 = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
    defer tensor1.deinit();
    var tensor2 = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
    defer tensor2.deinit();

    // Fill with test data
    for (tensor1.data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (tensor2.data, 0..) |*val, i| {
        val.* = @floatFromInt(i + 10);
    }

    // Test stacking along different dimensions
    {
        var result = try ops.stackAndFlatten(f32, tensor1, tensor2, -1);
        defer result.deinit();

        try expectEqual(@as(usize, 2), result.shape.len);
        try expectEqual(@as(usize, 1), result.shape[0]);
        try expectEqual(@as(usize, 6), result.shape[1]);

        // Verify data ordering
        try expectEqual(@as(f32, 0), result.data[0]); // First element from tensor1
        try expectEqual(@as(f32, 10), result.data[1]); // First element from tensor2
    }
}

test "Tensor outer product" {
    const allocator = testing.allocator;

    // Test 1: Basic outer product
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{3});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{2});
        defer t1.deinit();
        defer t2.deinit();

        // Set values: t1 = [1, 2, 3], t2 = [4, 5]
        t1.data[0] = 1.0;
        t1.data[1] = 2.0;
        t1.data[2] = 3.0;
        t2.data[0] = 4.0;
        t2.data[1] = 5.0;

        var result = try ops.outer(f32, t1, t2);
        defer result.deinit();

        // Check dimensions
        try testing.expectEqual(result.shape[0], 3);
        try testing.expectEqual(result.shape[1], 2);

        // Expected result:
        // [1 * 4  1 * 5]   [4   5]
        // [2 * 4  2 * 5] = [8  10]
        // [3 * 4  3 * 5]   [12 15]
        try testing.expectApproxEqAbs(result.data[0], 4.0, 0.001); // 1 * 4
        try testing.expectApproxEqAbs(result.data[1], 5.0, 0.001); // 1 * 5
        try testing.expectApproxEqAbs(result.data[2], 8.0, 0.001); // 2 * 4
        try testing.expectApproxEqAbs(result.data[3], 10.0, 0.001); // 2 * 5
        try testing.expectApproxEqAbs(result.data[4], 12.0, 0.001); // 3 * 4
        try testing.expectApproxEqAbs(result.data[5], 15.0, 0.001); // 3 * 5
    }

    // Test 2: Outer product with zeros
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{2});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{2});
        defer t1.deinit();
        defer t2.deinit();

        t1.data[0] = 0.0;
        t1.data[1] = 1.0;
        t2.data[0] = 1.0;
        t2.data[1] = 0.0;

        var result = try ops.outer(f32, t1, t2);
        defer result.deinit();

        try testing.expectApproxEqAbs(result.data[0], 0.0, 0.001);
        try testing.expectApproxEqAbs(result.data[1], 0.0, 0.001);
        try testing.expectApproxEqAbs(result.data[2], 1.0, 0.001);
        try testing.expectApproxEqAbs(result.data[3], 0.0, 0.001);
    }

    // Test 3: Error case - non-1D tensors
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        var t2 = try Tensor(f32).init(allocator, &[_]usize{2});
        defer t1.deinit();
        defer t2.deinit();

        try testing.expectError(error.InvalidDimensions, ops.outer(f32, t1, t2));
    }

    // Test 4: Large vectors
    {
        const size = 1000;
        var t1 = try Tensor(f32).init(allocator, &[_]usize{size});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{size});
        defer t1.deinit();
        defer t2.deinit();

        // Fill with simple values
        t1.fill(2.0);
        t2.fill(3.0);

        var result = try ops.outer(f32, t1, t2);
        defer result.deinit();

        // Check dimensions
        try testing.expectEqual(result.shape[0], size);
        try testing.expectEqual(result.shape[1], size);

        // Check some values (all should be 6.0 = 2.0 * 3.0)
        try testing.expectApproxEqAbs(result.data[0], 6.0, 0.001);
        try testing.expectApproxEqAbs(result.data[size - 1], 6.0, 0.001);
        try testing.expectApproxEqAbs(result.data[size * size - 1], 6.0, 0.001);
    }

    // Test 5: Outer product with negative numbers
    {
        var t1 = try Tensor(f32).init(allocator, &[_]usize{2});
        var t2 = try Tensor(f32).init(allocator, &[_]usize{2});
        defer t1.deinit();
        defer t2.deinit();

        t1.data[0] = -2.0;
        t1.data[1] = 3.0;
        t2.data[0] = 1.0;
        t2.data[1] = -4.0;

        var result = try ops.outer(f32, t1, t2);
        defer result.deinit();

        try testing.expectApproxEqAbs(result.data[0], -2.0, 0.001); // -2 * 1
        try testing.expectApproxEqAbs(result.data[1], 8.0, 0.001); // -2 * -4
        try testing.expectApproxEqAbs(result.data[2], 3.0, 0.001); // 3 * 1
        try testing.expectApproxEqAbs(result.data[3], -12.0, 0.001); // 3 * -4
    }
}

test "Tensor stability checks" {
    const allocator = testing.allocator;

    // Test 1: Basic stability check with normal values
    {
        var t = try Tensor(f32).init(allocator, &[_]usize{4});
        defer t.deinit();

        t.data[0] = 1.0;
        t.data[1] = 2.0;
        t.data[2] = -3.0;
        t.data[3] = 0.0;

        try testing.expect(try ops.isStable(f32, t));
        try testing.expect(!try ops.hasNaN(f32, t));
        try testing.expect(!try ops.hasInf(f32, t));

        // Check should pass
        try ops.checkStability(f32, t);
    }

    // Test 2: NaN detection
    {
        var t = try Tensor(f32).init(allocator, &[_]usize{3});
        defer t.deinit();

        t.data[0] = 1.0;
        t.data[1] = std.math.nan(f32);
        t.data[2] = 3.0;

        try testing.expect(!try ops.isStable(f32, t));
        try testing.expect(try ops.hasNaN(f32, t));
        try testing.expect(!try ops.hasInf(f32, t));

        // Get detailed info
        const info = try ops.getStabilityInfo(f32, t);
        try testing.expect(info.has_nan);
        try testing.expect(!info.has_pos_inf);
        try testing.expect(!info.has_neg_inf);
        try testing.expectEqual(info.nan_count, 1);
        try testing.expectEqual(info.first_nan_index.?, 1);

        // Check should fail with HasNaN
        try testing.expectError(StabilityError.HasNaN, ops.checkStability(f32, t));
    }

    // Test 3: Infinity detection
    {
        var t = try Tensor(f32).init(allocator, &[_]usize{4});
        defer t.deinit();

        t.data[0] = 1.0;
        t.data[1] = std.math.inf(f32);
        t.data[2] = -std.math.inf(f32);
        t.data[3] = 2.0;

        try testing.expect(!try ops.isStable(f32, t));
        try testing.expect(!try ops.hasNaN(f32, t));
        try testing.expect(try ops.hasInf(f32, t));

        const info = try ops.getStabilityInfo(f32, t);
        try testing.expect(!info.has_nan);
        try testing.expect(info.has_pos_inf);
        try testing.expect(info.has_neg_inf);
        try testing.expectEqual(info.pos_inf_count, 1);
        try testing.expectEqual(info.neg_inf_count, 1);
        try testing.expectEqual(info.first_pos_inf_index.?, 1);
        try testing.expectEqual(info.first_neg_inf_index.?, 2);

        // Check should fail with HasPositiveInfinity
        try testing.expectError(StabilityError.HasPositiveInfinity, ops.checkStability(f32, t));
    }

    // Test 4: Replace unstable values
    {
        var t = try Tensor(f32).init(allocator, &[_]usize{5});
        defer t.deinit();

        t.data[0] = 1.0;
        t.data[1] = std.math.nan(f32);
        t.data[2] = std.math.inf(f32);
        t.data[3] = -std.math.inf(f32);
        t.data[4] = 2.0;

        try ops.replaceUnstable(f32, &t, 0.0);

        try testing.expect(try ops.isStable(f32, t));
        try testing.expectApproxEqAbs(t.data[0], 1.0, 0.001);
        try testing.expectApproxEqAbs(t.data[1], 0.0, 0.001);
        try testing.expectApproxEqAbs(t.data[2], 0.0, 0.001);
        try testing.expectApproxEqAbs(t.data[3], 0.0, 0.001);
        try testing.expectApproxEqAbs(t.data[4], 2.0, 0.001);
    }

    // Test 5: Integer type stability (should always be stable)
    {
        var t = try Tensor(i32).init(allocator, &[_]usize{3});
        defer t.deinit();

        t.data[0] = -1;
        t.data[1] = 0;
        t.data[2] = 1;

        try testing.expect(try ops.isStable(i32, t));
        try testing.expect(!try ops.hasNaN(i32, t));
        try testing.expect(!try ops.hasInf(i32, t));

        // Check should pass
        try ops.checkStability(i32, t);
    }

    // Test 6: Mixed unstable values
    {
        var t = try Tensor(f32).init(allocator, &[_]usize{6});
        defer t.deinit();

        t.data[0] = std.math.nan(f32);
        t.data[1] = 1.0;
        t.data[2] = std.math.inf(f32);
        t.data[3] = 2.0;
        t.data[4] = -std.math.inf(f32);
        t.data[5] = std.math.nan(f32);

        const info = try ops.getStabilityInfo(f32, t);
        try testing.expect(info.has_nan);
        try testing.expect(info.has_pos_inf);
        try testing.expect(info.has_neg_inf);
        try testing.expectEqual(info.nan_count, 2);
        try testing.expectEqual(info.pos_inf_count, 1);
        try testing.expectEqual(info.neg_inf_count, 1);
        try testing.expectEqual(info.first_nan_index.?, 0);
        try testing.expectEqual(info.first_pos_inf_index.?, 2);
        try testing.expectEqual(info.first_neg_inf_index.?, 4);
    }
}

test "Tensor Slice Tests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test 1: Simple 2x2x2 tensor
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 2 });
        defer tensor.deinit();

        // Fill with sequential values
        for (tensor.data, 0..) |*val, i| {
            val.* = @floatFromInt(i);
        }

        // Get slice of first layer (should be 2x2)
        var slice = try tensor.getDimensionSlice(0, 0);
        defer slice.deinit();

        try expectEqual(@as(usize, 2), slice.shape.len);
        try expectEqual(@as(usize, 2), slice.shape[0]);
        try expectEqual(@as(usize, 2), slice.shape[1]);
        try expectEqual(@as(f32, 0), slice.data[0]);
        try expectEqual(@as(f32, 1), slice.data[1]);
        try expectEqual(@as(f32, 2), slice.data[2]);
        try expectEqual(@as(f32, 3), slice.data[3]);
    }

    // Test 2: 3x2x2 tensor, slice middle dimension
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 2, 2 });
        defer tensor.deinit();

        for (tensor.data, 0..) |*val, i| {
            val.* = @floatFromInt(i);
        }

        var slice = try tensor.getDimensionSlice(1, 0);
        defer slice.deinit();

        try expectEqual(@as(usize, 2), slice.shape.len);
        try expectEqual(@as(usize, 3), slice.shape[0]);
        try expectEqual(@as(usize, 2), slice.shape[1]);

        // Check first row
        try expectEqual(@as(f32, 0), slice.data[0]);
        try expectEqual(@as(f32, 1), slice.data[1]);

        // Check second row
        try expectEqual(@as(f32, 4), slice.data[2]);
        try expectEqual(@as(f32, 5), slice.data[3]);

        // Check third row
        try expectEqual(@as(f32, 8), slice.data[4]);
        try expectEqual(@as(f32, 9), slice.data[5]);
    }

    // Test 3: Error cases
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 2, 2 });
        defer tensor.deinit();

        try expectError(error.InvalidDimension, tensor.getDimensionSlice(3, 0));
        try expectError(error.IndexOutOfBounds, tensor.getDimensionSlice(0, 2));
    }

    // Test 4: 1D tensor slice (should result in scalar)
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{3});
        defer tensor.deinit();

        for (tensor.data, 0..) |*val, i| {
            val.* = @floatFromInt(i);
        }

        var slice = try tensor.getDimensionSlice(0, 1);
        defer slice.deinit();

        try expectEqual(@as(usize, 0), slice.shape.len);
        try expectEqual(@as(usize, 1), slice.data.len);
        try expectEqual(@as(f32, 1), slice.data[0]);
    }

    // Test 5: Non-uniform dimensions
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
        defer tensor.deinit();

        for (tensor.data, 0..) |*val, i| {
            val.* = @floatFromInt(i);
        }

        // Slice along first dimension
        {
            var slice = try tensor.getDimensionSlice(0, 0);
            defer slice.deinit();
            try expectEqual(@as(usize, 2), slice.shape.len);
            try expectEqual(@as(usize, 3), slice.shape[0]);
            try expectEqual(@as(usize, 4), slice.shape[1]);
        }

        // Slice along middle dimension
        {
            var slice = try tensor.getDimensionSlice(1, 1);
            defer slice.deinit();
            try expectEqual(@as(usize, 2), slice.shape.len);
            try expectEqual(@as(usize, 2), slice.shape[0]);
            try expectEqual(@as(usize, 4), slice.shape[1]);
        }
    }
}

test "layerNorm basic functionality" {
    const allocator = testing.allocator;

    // Create test input tensor
    var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer input.deinit();

    // Fill with test data that will have mean 0 and variance 1 after normalization
    input.data[0] = -1.0;
    input.data[1] = 0.0;
    input.data[2] = 1.0;
    input.data[3] = -1.0;
    input.data[4] = 0.0;
    input.data[5] = 1.0;

    // Create weight and bias tensors
    var weight = try Tensor(f32).init(allocator, &[_]usize{3});
    defer weight.deinit();
    @memset(weight.data, 1.0); // Scale factor of 1

    var bias = try Tensor(f32).init(allocator, &[_]usize{3});
    defer bias.deinit();
    @memset(bias.data, 0.0); // No bias

    // Apply layer normalization
    var result = try ops.layerNorm(f32, input, weight, bias, 1e-5);
    defer result.deinit();

    // Verify output shape
    try testing.expectEqual(input.shape.len, result.shape.len);
    try testing.expectEqual(input.shape[0], result.shape[0]);
    try testing.expectEqual(input.shape[1], result.shape[1]);

    // Verify first row is normalized (mean ≈ 0, variance ≈ 1)
    const eps = 1e-5;
    var mean: f32 = 0;
    for (0..3) |i| {
        mean += result.data[i];
    }
    mean /= 3;
    try testing.expect(@abs(mean) < eps);

    var variance: f32 = 0;
    for (0..3) |i| {
        const diff = result.data[i] - mean;
        variance += diff * diff;
    }
    variance /= 3;
    try testing.expect(@abs(variance - 1.0) < 0.01); // Allow for some numerical error
}

test "layerNorm stability checks" {
    const allocator = testing.allocator;

    // Test case 1: Input contains NaN
    {
        var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer input.deinit();
        input.data[0] = std.math.nan(f32);

        var weight = try Tensor(f32).init(allocator, &[_]usize{3});
        defer weight.deinit();
        @memset(weight.data, 1.0);

        var bias = try Tensor(f32).init(allocator, &[_]usize{3});
        defer bias.deinit();
        @memset(bias.data, 0.0);

        try testing.expectError(error.HasNaN, ops.layerNorm(f32, input, weight, bias, 1e-5));
    }

    // Test case 2: Zero variance
    {
        var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer input.deinit();
        @memset(input.data, 1.0); // All same values -> zero variance

        var weight = try Tensor(f32).init(allocator, &[_]usize{3});
        defer weight.deinit();
        @memset(weight.data, 1.0);

        var bias = try Tensor(f32).init(allocator, &[_]usize{3});
        defer bias.deinit();
        @memset(bias.data, 0.0);

        var result = try ops.layerNorm(f32, input, weight, bias, 1e-5);
        defer result.deinit();

        // Should still work due to epsilon
        try testing.expect(!std.math.isNan(result.data[0]));
    }

    // Test case 3: Negative epsilon
    {
        var input = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer input.deinit();
        @memset(input.data, 1.0);

        var weight = try Tensor(f32).init(allocator, &[_]usize{3});
        defer weight.deinit();
        @memset(weight.data, 1.0);

        var bias = try Tensor(f32).init(allocator, &[_]usize{3});
        defer bias.deinit();
        @memset(bias.data, 0.0);

        try testing.expectError(error.InvalidEpsilon, ops.layerNorm(f32, input, weight, bias, -1e-5));
    }
}

test "broadcast_add basic" {
    const allocator = testing.allocator;

    // Test case 1: [2,3] + [3]
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{3});
        defer b.deinit();

        // Initialize values
        for (0..6) |i| {
            a.data[i] = @floatFromInt(i);
        }
        for (0..3) |i| {
            b.data[i] = 1.0;
        }

        try ops.broadcast_add(f32, &a, b);

        // Check results
        try testing.expectApproxEqAbs(a.data[0], 1.0, 1e-6);
        try testing.expectApproxEqAbs(a.data[1], 2.0, 1e-6);
        try testing.expectApproxEqAbs(a.data[2], 3.0, 1e-6);
        try testing.expectApproxEqAbs(a.data[3], 4.0, 1e-6);
        try testing.expectApproxEqAbs(a.data[4], 5.0, 1e-6);
        try testing.expectApproxEqAbs(a.data[5], 6.0, 1e-6);
    }

    // Test case 2: [2,3] + [1]
    {
        var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
        defer a.deinit();
        var b = try Tensor(f32).init(allocator, &[_]usize{1});
        defer b.deinit();

        // Initialize values
        for (0..6) |i| {
            a.data[i] = @floatFromInt(i);
        }
        b.data[0] = 1.0;

        try ops.broadcast_add(f32, &a, b);

        // Each element should be increased by 1
        for (0..6) |i| {
            try testing.expectApproxEqAbs(a.data[i], @as(f32, @floatFromInt(i)) + 1.0, 1e-6);
        }
    }
}

test "broadcast_multiply - same shape tensors" {
    const allocator = testing.allocator;
    // Initialize tensors with same shape [2, 2]
    var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();

    // Set values
    a.data[0] = 1.0;
    a.data[1] = 2.0;
    a.data[2] = 3.0;
    a.data[3] = 4.0;
    b.data[0] = 2.0;
    b.data[1] = 3.0;
    b.data[2] = 4.0;
    b.data[3] = 5.0;

    // Perform broadcast multiplication
    try ops.broadcast_multiply(f32, &a, b);

    // Check results
    try expectEqual(a.data[0], 2.0);
    try expectEqual(a.data[1], 6.0);
    try expectEqual(a.data[2], 12.0);
    try expectEqual(a.data[3], 20.0);
}

test "broadcast_multiply - broadcasting scalar to matrix" {
    // Initialize matrix and scalar tensors
    const allocator = testing.allocator;
    var matrix = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer matrix.deinit();
    var scalar = try Tensor(f32).init(allocator, &[_]usize{1});
    defer scalar.deinit();

    // Set values
    matrix.data[0] = 1.0;
    matrix.data[1] = 2.0;
    matrix.data[2] = 3.0;
    matrix.data[3] = 4.0;
    matrix.data[4] = 5.0;
    matrix.data[5] = 6.0;
    scalar.data[0] = 2.0;

    // Perform broadcast multiplication
    try ops.broadcast_multiply(f32, &matrix, scalar);

    // Check results
    try expectEqual(matrix.data[0], 2.0);
    try expectEqual(matrix.data[1], 4.0);
    try expectEqual(matrix.data[2], 6.0);
    try expectEqual(matrix.data[3], 8.0);
    try expectEqual(matrix.data[4], 10.0);
    try expectEqual(matrix.data[5], 12.0);
}

test "broadcast_multiply - broadcasting vector to matrix" {
    const allocator = testing.allocator;
    // Initialize matrix and vector tensors
    var matrix = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer matrix.deinit();
    var vector = try Tensor(f32).init(allocator, &[_]usize{2});
    defer vector.deinit();

    // Set values
    matrix.data[0] = 1.0;
    matrix.data[1] = 2.0;
    matrix.data[2] = 3.0;
    matrix.data[3] = 4.0;
    matrix.data[4] = 5.0;
    matrix.data[5] = 6.0;
    vector.data[0] = 2.0;
    vector.data[1] = 3.0;

    // Perform broadcast multiplication
    try ops.broadcast_multiply(f32, &matrix, vector);

    // Check results
    try expectEqual(matrix.data[0], 2.0);
    try expectEqual(matrix.data[1], 6.0);
    try expectEqual(matrix.data[2], 6.0);
    try expectEqual(matrix.data[3], 12.0);
    try expectEqual(matrix.data[4], 10.0);
    try expectEqual(matrix.data[5], 18.0);
}

test "broadcast_multiply - integer tensors" {
    const allocator = testing.allocator;
    var a = try Tensor(i32).init(allocator, &[_]usize{4});
    defer a.deinit();
    var b = try Tensor(i32).init(allocator, &[_]usize{2});
    defer b.deinit();

    a.data[0] = 1;
    a.data[1] = 2;
    a.data[2] = 3;
    a.data[3] = 4;
    b.data[0] = 2;
    b.data[1] = 3;

    try ops.broadcast_multiply(i32, &a, b);

    try expectEqual(a.data[0], 2);
    try expectEqual(a.data[1], 6);
    try expectEqual(a.data[2], 6);
    try expectEqual(a.data[3], 12);
}

test "broadcast_subtract - same shape tensors" {
    const allocator = testing.allocator;
    var a = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer a.deinit();
    var b = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer b.deinit();

    a.data[0] = 5.0;
    a.data[1] = 7.0;
    a.data[2] = 9.0;
    a.data[3] = 11.0;
    b.data[0] = 1.0;
    b.data[1] = 2.0;
    b.data[2] = 3.0;
    b.data[3] = 4.0;

    try ops.broadcast_subtract(f32, &a, b);

    try expectEqual(a.data[0], 4.0);
    try expectEqual(a.data[1], 5.0);
    try expectEqual(a.data[2], 6.0);
    try expectEqual(a.data[3], 7.0);
}

test "broadcast_subtract - broadcasting scalar to matrix" {
    const allocator = testing.allocator;
    var matrix = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer matrix.deinit();
    var scalar = try Tensor(f32).init(allocator, &[_]usize{1});
    defer scalar.deinit();

    matrix.data[0] = 5.0;
    matrix.data[1] = 6.0;
    matrix.data[2] = 7.0;
    matrix.data[3] = 8.0;
    matrix.data[4] = 9.0;
    matrix.data[5] = 10.0;
    scalar.data[0] = 2.0;

    try ops.broadcast_subtract(f32, &matrix, scalar);

    try expectEqual(matrix.data[0], 3.0);
    try expectEqual(matrix.data[1], 4.0);
    try expectEqual(matrix.data[2], 5.0);
    try expectEqual(matrix.data[3], 6.0);
    try expectEqual(matrix.data[4], 7.0);
    try expectEqual(matrix.data[5], 8.0);
}

test "broadcast_subtract - broadcasting vector to matrix" {
    const allocator = testing.allocator;
    var matrix = try Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer matrix.deinit();
    var vector = try Tensor(f32).init(allocator, &[_]usize{2});
    defer vector.deinit();

    matrix.data[0] = 5.0;
    matrix.data[1] = 7.0;
    matrix.data[2] = 9.0;
    matrix.data[3] = 11.0;
    matrix.data[4] = 13.0;
    matrix.data[5] = 15.0;
    vector.data[0] = 1.0;
    vector.data[1] = 2.0;

    try ops.broadcast_subtract(f32, &matrix, vector);

    try expectEqual(matrix.data[0], 4.0);
    try expectEqual(matrix.data[1], 5.0);
    try expectEqual(matrix.data[2], 8.0);
    try expectEqual(matrix.data[3], 9.0);
    try expectEqual(matrix.data[4], 12.0);
    try expectEqual(matrix.data[5], 13.0);
}

test "broadcast_subtract - integer tensors" {
    const allocator = testing.allocator;
    var a = try Tensor(i32).init(allocator, &[_]usize{4});
    defer a.deinit();
    var b = try Tensor(i32).init(allocator, &[_]usize{2});
    defer b.deinit();

    a.data[0] = 10;
    a.data[1] = 12;
    a.data[2] = 14;
    a.data[3] = 16;
    b.data[0] = 2;
    b.data[1] = 3;

    try ops.broadcast_subtract(i32, &a, b);

    try expectEqual(a.data[0], 8);
    try expectEqual(a.data[1], 9);
    try expectEqual(a.data[2], 12);
    try expectEqual(a.data[3], 13);
}

test "getChunk basic functionality" {
    const allocator = testing.allocator;

    // Test case 1: Simple 2D tensor chunking along dimension 1
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 6 });
        defer tensor.deinit();

        // Fill with sequential values
        for (0..12) |i| {
            tensor.data[i] = @floatFromInt(i);
        }

        // Get middle chunk (should be values 2,3 and 8,9)
        var chunk = try ops.getChunk(f32, tensor, 1, 1, 3);
        defer chunk.deinit();

        // Verify shape
        try testing.expectEqual(chunk.shape.len, @as(usize, 2));
        try testing.expectEqual(chunk.shape[0], @as(usize, 2));
        try testing.expectEqual(chunk.shape[1], @as(usize, 2));

        // Verify values
        try testing.expectEqual(chunk.data[0], 2);
        try testing.expectEqual(chunk.data[1], 3);
        try testing.expectEqual(chunk.data[2], 8);
        try testing.expectEqual(chunk.data[3], 9);
    }

    // Test case 2: 3D tensor chunking along middle dimension
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 4, 3 });
        defer tensor.deinit();

        // Fill with sequential values
        for (0..24) |i| {
            tensor.data[i] = @floatFromInt(i);
        }

        // Get first chunk
        var chunk = try ops.getChunk(f32, tensor, 1, 0, 2);
        defer chunk.deinit();

        // Verify shape
        try testing.expectEqual(chunk.shape.len, @as(usize, 3));
        try testing.expectEqual(chunk.shape[0], @as(usize, 2));
        try testing.expectEqual(chunk.shape[1], @as(usize, 2));
        try testing.expectEqual(chunk.shape[2], @as(usize, 3));
    }
}

test "getChunk error cases" {
    const allocator = testing.allocator;

    // Setup test tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 6 });
    defer tensor.deinit();

    // Test case 1: Invalid dimension
    try testing.expectError(error.InvalidDimension, ops.getChunk(f32, tensor, 2, 0, 2));

    // Test case 2: Invalid number of chunks
    try testing.expectError(error.InvalidNumChunks, ops.getChunk(f32, tensor, 1, 0, 7));
    try testing.expectError(error.InvalidNumChunks, ops.getChunk(f32, tensor, 1, 0, 0));

    // Test case 3: Invalid chunk index
    try testing.expectError(error.InvalidChunkIndex, ops.getChunk(f32, tensor, 1, 3, 3));

    // Test case 4: Uneven chunk size
    try testing.expectError(error.UnevenChunkSize, ops.getChunk(f32, tensor, 1, 0, 4));
}

test "getChunk all chunks" {
    const allocator = testing.allocator;

    // Create a test tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 6 });
    defer tensor.deinit();

    // Fill with sequential values
    for (0..12) |i| {
        tensor.data[i] = @floatFromInt(i);
    }

    // Get all chunks and verify they combine to form the original tensor
    const num_chunks = 3;
    var chunks: [num_chunks]Tensor(f32) = undefined;
    defer {
        for (0..num_chunks) |i| {
            if (chunks[i].data.len > 0) {
                chunks[i].deinit();
            }
        }
    }

    // Get all chunks
    for (0..num_chunks) |i| {
        chunks[i] = try ops.getChunk(f32, tensor, 1, i, num_chunks);
    }

    // Verify dimensions of each chunk
    for (chunks) |chunk| {
        try testing.expectEqual(chunk.shape[0], @as(usize, 2));
        try testing.expectEqual(chunk.shape[1], @as(usize, 2));
    }

    // Verify values in each chunk
    // First chunk should have values 0,1,6,7
    try testing.expectEqual(chunks[0].data[0], 0);
    try testing.expectEqual(chunks[0].data[1], 1);
    try testing.expectEqual(chunks[0].data[2], 6);
    try testing.expectEqual(chunks[0].data[3], 7);

    // Second chunk should have values 2,3,8,9
    try testing.expectEqual(chunks[1].data[0], 2);
    try testing.expectEqual(chunks[1].data[1], 3);
    try testing.expectEqual(chunks[1].data[2], 8);
    try testing.expectEqual(chunks[1].data[3], 9);

    // Third chunk should have values 4,5,10,11
    try testing.expectEqual(chunks[2].data[0], 4);
    try testing.expectEqual(chunks[2].data[1], 5);
    try testing.expectEqual(chunks[2].data[2], 10);
    try testing.expectEqual(chunks[2].data[3], 11);
}

test "transpose - 2x2 matrix" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer tensor.deinit();

    // Initialize with sequential values
    tensor.data[0] = 1.0;
    tensor.data[1] = 2.0;
    tensor.data[2] = 3.0;
    tensor.data[3] = 4.0;

    try ops.transposeAxes(f32, &tensor, 0, 1);

    try testing.expectEqual(tensor.shape[0], 2);
    try testing.expectEqual(tensor.shape[1], 2);
    try testing.expectApproxEqAbs(tensor.data[0], 1.0, 0.0001);
    try testing.expectApproxEqAbs(tensor.data[1], 3.0, 0.0001);
    try testing.expectApproxEqAbs(tensor.data[2], 2.0, 0.0001);
    try testing.expectApproxEqAbs(tensor.data[3], 4.0, 0.0001);
}

test "transpose - 3x3 matrix" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 3, 3 });
    defer tensor.deinit();

    // Initialize with sequential values
    for (0..9) |i| {
        tensor.data[i] = @floatFromInt(i + 1);
    }

    try ops.transposeAxes(f32, &tensor, 0, 1);

    try testing.expectEqual(tensor.shape[0], 3);
    try testing.expectEqual(tensor.shape[1], 3);

    // Expected values after transpose:
    // [1 4 7]
    // [2 5 8]
    // [3 6 9]
    const expected = [_]f32{ 1, 4, 7, 2, 5, 8, 3, 6, 9 };
    for (expected, 0..) |val, i| {
        try testing.expectApproxEqAbs(tensor.data[i], val, 0.0001);
    }
}

test "transpose - 3D tensor" {
    const allocator = testing.allocator;
    const dim1 = 2;
    const dim2 = 3;
    const dim3 = 2;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ dim1, dim2, dim3 });
    defer tensor.deinit();

    // Fill with sequential values
    for (0..tensor.data.len) |i| {
        tensor.data[i] = @floatFromInt(i + 1);
    }

    // Test transposing first two dimensions
    try ops.transposeAxes(f32, &tensor, 0, 1);

    // Verify shape
    try testing.expectEqual(tensor.shape[0], dim2);
    try testing.expectEqual(tensor.shape[1], dim1);
    try testing.expectEqual(tensor.shape[2], dim3);

    // Test transposing with last dimension
    try ops.transposeAxes(f32, &tensor, 1, 2);

    // Verify new shape
    try testing.expectEqual(tensor.shape[0], dim2);
    try testing.expectEqual(tensor.shape[1], dim3);
    try testing.expectEqual(tensor.shape[2], dim1);
}

test "transpose - error cases" {
    const allocator = testing.allocator;

    // Test invalid dimension indices
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
        defer tensor.deinit();

        try testing.expectError(error.InvalidDimension, ops.transposeAxes(f32, &tensor, 0, 2));
        try testing.expectError(error.InvalidDimension, ops.transposeAxes(f32, &tensor, 2, 0));
    }
}

test "transpose - edge cases" {
    const allocator = testing.allocator;

    // Test 1x1 matrix
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 1, 1 });
        defer tensor.deinit();
        tensor.data[0] = 42.0;

        try ops.transposeAxes(f32, &tensor, 0, 1);
        try testing.expectEqual(tensor.shape[0], 1);
        try testing.expectEqual(tensor.shape[1], 1);
        try testing.expectApproxEqAbs(tensor.data[0], 42.0, 0.0001);
    }

    // Test 1xN matrix
    {
        var tensor = try Tensor(f32).init(allocator, &[_]usize{ 1, 3 });
        defer tensor.deinit();
        tensor.data[0] = 1.0;
        tensor.data[1] = 2.0;
        tensor.data[2] = 3.0;

        try ops.transposeAxes(f32, &tensor, 0, 1);
        try testing.expectEqual(tensor.shape[0], 3);
        try testing.expectEqual(tensor.shape[1], 1);
        try testing.expectApproxEqAbs(tensor.data[0], 1.0, 0.0001);
        try testing.expectApproxEqAbs(tensor.data[1], 2.0, 0.0001);
        try testing.expectApproxEqAbs(tensor.data[2], 3.0, 0.0001);
    }
}

test "transpose - different data types" {
    const allocator = testing.allocator;

    // Test with integers
    {
        var tensor = try Tensor(i32).init(allocator, &[_]usize{ 2, 2 });
        defer tensor.deinit();
        tensor.data[0] = 1;
        tensor.data[1] = 2;
        tensor.data[2] = 3;
        tensor.data[3] = 4;

        try ops.transposeAxes(i32, &tensor, 0, 1);
        try testing.expectEqual(tensor.data[0], 1);
        try testing.expectEqual(tensor.data[1], 3);
        try testing.expectEqual(tensor.data[2], 2);
        try testing.expectEqual(tensor.data[3], 4);
    }

    // Test with f64
    {
        var tensor = try Tensor(f64).init(allocator, &[_]usize{ 2, 2 });
        defer tensor.deinit();
        tensor.data[0] = 1.5;
        tensor.data[1] = 2.5;
        tensor.data[2] = 3.5;
        tensor.data[3] = 4.5;

        try ops.transposeAxes(f64, &tensor, 0, 1);
        try testing.expectApproxEqAbs(tensor.data[0], 1.5, 0.0001);
        try testing.expectApproxEqAbs(tensor.data[1], 3.5, 0.0001);
        try testing.expectApproxEqAbs(tensor.data[2], 2.5, 0.0001);
        try testing.expectApproxEqAbs(tensor.data[3], 4.5, 0.0001);
    }
}

test "transpose - stress test with large dimensions" {
    const allocator = testing.allocator;

    // Test with a large 3D tensor
    {
        const dim1 = 64;
        const dim2 = 32;
        const dim3 = 16;

        var tensor = try Tensor(f32).init(allocator, &[_]usize{ dim1, dim2, dim3 });
        defer tensor.deinit();

        // Fill with a pattern we can verify
        for (0..tensor.data.len) |i| {
            tensor.data[i] = @floatFromInt(i % 100); // Use modulo to keep numbers manageable
        }

        // Transpose different combinations of axes
        try ops.transposeAxes(f32, &tensor, 0, 1);
        try ops.transposeAxes(f32, &tensor, 1, 2);
        try ops.transposeAxes(f32, &tensor, 0, 2);

        // Verify final dimensions
        try testing.expectEqual(tensor.shape.len, 3);
    }

    // Test with a very large 2D matrix
    {
        const rows = 1000;
        const cols = 1000;

        var tensor = try Tensor(f32).init(allocator, &[_]usize{ rows, cols });
        defer tensor.deinit();

        // Fill with a verifiable pattern
        for (0..tensor.data.len) |i| {
            tensor.data[i] = @floatFromInt(i % 1000);
        }

        // Transpose multiple times
        for (0..5) |_| {
            try ops.transposeAxes(f32, &tensor, 0, 1);
        }

        // After even number of transposes, should be back to original shape
        try testing.expectEqual(tensor.shape[0], rows);
        try testing.expectEqual(tensor.shape[1], cols);
    }
}

test "transpose - consecutive operations" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3, 4 });
    defer tensor.deinit();

    // Fill with sequential values
    for (0..tensor.data.len) |i| {
        tensor.data[i] = @floatFromInt(i);
    }

    // Perform series of transpose operations
    try ops.transposeAxes(f32, &tensor, 0, 1);
    try ops.transposeAxes(f32, &tensor, 1, 2);
    try ops.transposeAxes(f32, &tensor, 0, 2);

    // Verify dimensions after multiple operations
    try testing.expectEqual(tensor.shape.len, 3);
}

test "transpose - special values" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 2 });
    defer tensor.deinit();

    // Test with special floating point values
    // Initial matrix:
    // [inf    -inf]
    // [nan     0.0]
    tensor.data[0] = std.math.inf(f32);
    tensor.data[1] = -std.math.inf(f32);
    tensor.data[2] = std.math.nan(f32);
    tensor.data[3] = 0.0;

    try ops.transposeAxes(f32, &tensor, 0, 1);

    // After transpose, should be:
    // [inf     nan]
    // [-inf    0.0]
    try testing.expect(std.math.isInf(tensor.data[0]));
    try testing.expect(std.math.isNan(tensor.data[1]));
    try testing.expect(std.math.isNegativeInf(tensor.data[2]));
    try testing.expectApproxEqAbs(tensor.data[3], 0.0, 0.0001);
}

test "ops - zeros" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test 1D tensor
    {
        const shape = [_]usize{3};
        var tensor = try ops.zeros(f32, allocator, &shape);
        defer tensor.deinit();

        try expectEqual(tensor.shape.len, 1);
        try expectEqual(tensor.shape[0], 3);
        try expectEqual(tensor.data.len, 3);

        // Verify all elements are zero
        for (tensor.data) |val| {
            try expectEqual(val, 0);
        }
    }

    // Test 2D tensor
    {
        const shape = [_]usize{ 2, 3 };
        var tensor = try ops.zeros(f32, allocator, &shape);
        defer tensor.deinit();

        try expectEqual(tensor.shape.len, 2);
        try expectEqual(tensor.shape[0], 2);
        try expectEqual(tensor.shape[1], 3);
        try expectEqual(tensor.data.len, 6);

        // Verify all elements are zero
        for (tensor.data) |val| {
            try expectEqual(val, 0);
        }
    }
}

const math = std.math;

test "ops.gelu basic functionality f32" {
    const allocator = testing.allocator;

    // Test 1D tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{5});
    defer tensor.deinit();

    // Set test values
    tensor.data[0] = -2.0;
    tensor.data[1] = -1.0;
    tensor.data[2] = 0.0;
    tensor.data[3] = 1.0;
    tensor.data[4] = 2.0;

    try ops.gelu(f32, &tensor);

    try testing.expectApproxEqAbs(tensor.data[0], -0.045402, 0.001);
    try testing.expectApproxEqAbs(tensor.data[1], -0.158808, 0.001);
    try testing.expectApproxEqAbs(tensor.data[2], 0.0, 0.001);
    try testing.expectApproxEqAbs(tensor.data[3], 0.841192, 0.001);
    try testing.expectApproxEqAbs(tensor.data[4], 1.954598, 0.001);
}

test "ops.gelu f64 precision" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f64).init(allocator, &[_]usize{3});
    defer tensor.deinit();

    tensor.data[0] = -0.5;
    tensor.data[1] = 0.5;
    tensor.data[2] = 1.5;

    try ops.gelu(f64, &tensor);

    try testing.expectApproxEqAbs(tensor.data[0], -0.154286, 0.000001);
    try testing.expectApproxEqAbs(tensor.data[1], 0.345714, 0.000001);
    try testing.expectApproxEqAbs(tensor.data[2], 1.399572, 0.000001);
}

test "ops.gelu 2D tensor shape preservation" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with some values
    for (tensor.data, 0..) |*x, i| {
        x.* = @floatFromInt(i);
    }

    try ops.gelu(f32, &tensor);

    // Just verify the shape is preserved
    try testing.expect(tensor.shape.len == 2);
    try testing.expect(tensor.shape[0] == 2);
    try testing.expect(tensor.shape[1] == 3);
}

test "ops.gelu empty tensor" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{0});
    defer tensor.deinit();

    try ops.gelu(f32, &tensor);
}

test "ops.gelu large values" {
    const allocator = testing.allocator;

    var tensor = try Tensor(f32).init(allocator, &[_]usize{3});
    defer tensor.deinit();

    tensor.data[0] = -10.0;
    tensor.data[1] = 10.0;
    tensor.data[2] = 100.0;

    try ops.gelu(f32, &tensor);

    // For very negative values, GELU approaches 0
    try testing.expectApproxEqAbs(tensor.data[0], 0.0, 0.001);
    // For large positive values, GELU approaches the input value
    try testing.expectApproxEqAbs(tensor.data[1], 10.0, 0.1);
    try testing.expectApproxEqAbs(tensor.data[2], 100.0, 0.1);
}

test "argmax basic functionality" {
    // const std = @import("std");
    // const testing = std.testing;
    const allocator = testing.allocator;

    // Create a test tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{5});
    defer tensor.deinit();

    // Fill with test data
    tensor.data[0] = 1.0;
    tensor.data[1] = 3.0;
    tensor.data[2] = 0.5;
    tensor.data[3] = 2.0;
    tensor.data[4] = 1.5;

    const result = try ops.argmax(f32, tensor);
    try testing.expectEqual(result, 1); // Index 1 has the highest value (3.0)
}

test "argmax with 2D tensor" {
    // const std = @import("std");
    // const testing = std.testing;
    const allocator = testing.allocator;

    // Create a 2D tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer tensor.deinit();

    // Fill with test data [[-1.0, 2.0, 1.0], [0.5, 3.0, -0.5]]
    tensor.data[0] = -1.0;
    tensor.data[1] = 2.0;
    tensor.data[2] = 1.0;
    tensor.data[3] = 0.5;
    tensor.data[4] = 3.0;
    tensor.data[5] = -0.5;

    const result = try ops.argmax(f32, tensor);
    try testing.expectEqual(result, 1); // Index 1 in the last dimension has the highest value
}

test "argmax with empty tensor" {
    // const std = @import("std");
    // const testing = std.testing;
    const allocator = testing.allocator;

    // Create an empty tensor
    var tensor = try Tensor(f32).init(allocator, &[_]usize{0});
    defer tensor.deinit();

    try testing.expectError(error.EmptyTensor, ops.argmax(f32, tensor));
}
