const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating
const helper = @import("helper.zig");
pub usingnamespace @import("ops.zig");

pub fn Tensor(comptime DataType: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        shape: []usize,
        data: []align(32) DataType,
        // id: usize = undefined,

        // var id_mutex = std.Thread.Mutex{};
        // var next_id: usize = 0;

        // Original functions...
        /// Initializes a tensor with the given shape and allocates memory for its data.
        ///
        /// This function calculates the total size of the tensor by multiplying the dimensions
        /// provided in the `shape` array. It then allocates memory for the shape and data of the tensor
        /// using the provided allocator. The data memory is zero-initialized.
        ///
        /// Parameters:
        /// - `allocator`: The allocator to use for memory allocation.
        /// - `shape`: An array of dimensions specifying the shape of the tensor.
        ///
        /// Returns:
        /// - `Self`: A new instance of the tensor with allocated and zero-initialized data.
        ///
        /// Errors:
        /// - Returns an error if memory allocation for the shape or data fails.
        pub fn init(allocator: Allocator, shape: []const usize) !Self {
            var size: u128 = 1;
            for (shape) |dim| {
                size = size * dim;
                // Check if we would overflow usize
                if (size > std.math.maxInt(usize)) {
                    std.debug.print("size entered = {any}", .{size});
                    return error.TensorTooLarge;
                }
            }
            const shape_copy = try allocator.alloc(usize, shape.len);
            @memcpy(shape_copy, shape);

            // Now we know size fits in usize
            const final_size: usize = @intCast(size);
            const data = try allocator.alignedAlloc(DataType, 32, final_size);
            if (DataType == bool) {
                @memset(data, false);
            } else {
                @memset(data, 0);
            }

            const self = Tensor(DataType){
                .allocator = allocator,
                .shape = shape_copy,
                .data = data,
            };

            // Assign unique ID
            // id_mutex.lock();
            // defer id_mutex.unlock();
            // self.id = next_id;
            // next_id += 1;

            // std.debug.print("Created tensor {d} at {x} with shape {any}\n", .{ self.id, @returnAddress(), shape });

            return self;
        }

        /// Deinitializes the tensor by freeing its allocated memory for shape and data.
        ///
        /// This function should be called when the tensor is no longer needed to
        /// release the memory allocated for its shape and data arrays.
        ///
        /// Parameters:
        /// - `self`: A pointer to the tensor instance to be deinitialized.
        ///
        /// Note:
        /// Ensure that the allocator used for allocating the shape and data is the same
        /// allocator used for freeing them to avoid undefined behavior.
        pub fn deinit(self: *Self) void {
            // std.debug.print("Destroying tensor {d} at {x} with shape {any}\n", .{ self.id, @returnAddress(), self.shape });
            self.allocator.free(self.shape);
            self.allocator.free(self.data);
        }

        /// Returns a slice of the tensor's data.
        ///
        /// This function provides access to the underlying data of the tensor as a slice.
        ///
        /// Returns:
        /// - `[]DataType`: A slice containing the tensor's data.
        pub fn getSlice(self: Self) []DataType {
            return self.data;
        }

        /// Returns the dimensions of the tensor as a struct with fields `m` and `n`.
        ///
        /// The dimensions are derived from the `shape` array of the tensor.
        ///
        /// Returns:
        /// - A struct with two fields:
        ///   - `m`: The size of the first dimension.
        ///   - `n`: The size of the second dimension.
        pub fn getDimensions(self: Self) struct { m: usize, n: usize } {
            return .{
                .m = self.shape[0],
                .n = self.shape[1],
            };
        }

        /// Returns the total number of elements in the tensor.
        /// This function calculates the total number of elements in the tensor by multiplying
        /// all the dimensions specified in the `shape` array.
        fn calculateSize(shape: []const usize) usize {
            var size: usize = 1;
            for (shape) |dim| {
                size *= dim;
            }
            return size;
        }

        /// Reshapes the tensor to the specified new shape.
        ///
        /// This function changes the shape of the tensor to the given `new_shape`.
        /// It first calculates the total size of the new shape and verifies that it
        /// matches the current size of the tensor. If the sizes do not match, it
        /// returns an `IncompatibleShape` error. If the sizes match, it updates the
        /// tensor's shape to the new shape.
        ///
        /// # Parameters
        /// - `self`: A pointer to the tensor object.
        /// - `new_shape`: A slice of `usize` representing the new shape.
        ///
        /// # Returns
        /// - `!void`: Returns an error if the new shape is incompatible with the current size.
        ///
        /// # Errors
        /// - `IncompatibleShape`: Returned if the total size of the new shape does not match the current size of the tensor.
        ///
        /// # Example
        /// ```zig
        /// var tensor = Tensor.init(allocator, shape, data);
        /// try tensor.reshape(new_shape);
        /// ```
        // Shape operations
        pub fn reshape(self: *Self, new_shape: []const usize) !void {
            var new_size: usize = 1;
            for (new_shape) |dim| {
                new_size *= dim;
            }

            // Verify that the new shape is compatible
            const current_size = @as(usize, @intCast(self.data.len));
            if (new_size != current_size) {
                return error.IncompatibleShape;
            }

            // Update shape
            const new_shape_copy = try self.allocator.alloc(usize, new_shape.len);
            @memcpy(new_shape_copy, new_shape);

            self.allocator.free(self.shape);
            self.shape = new_shape_copy;
        }

        // Inside the Tensor struct definition:

        /// Adds a dimension of size 1 at the specified position.
        /// Negative dimensions are supported - they count from the end.
        /// For example: -1 means insert at the end, -2 means insert at second to last position
        pub fn unsqueeze(self: *Self, dim: isize) !void {
            const positive_dim = if (dim >= 0)
                @as(usize, @intCast(dim))
            else blk: {
                const n_dims: isize = @intCast(self.shape.len);
                const adjusted_dim = n_dims + 1 + dim;
                if (adjusted_dim < 0) return error.InvalidDimension;
                break :blk @as(usize, @intCast(adjusted_dim));
            };

            // Verify dimension is valid
            if (positive_dim > self.shape.len) return error.InvalidDimension;

            // Create new shape with extra dimension
            var new_shape = try self.allocator.alloc(usize, self.shape.len + 1);
            errdefer self.allocator.free(new_shape);

            // Copy shape values with new dimension of size 1
            @memcpy(new_shape[0..positive_dim], self.shape[0..positive_dim]);
            new_shape[positive_dim] = 1;
            if (positive_dim < self.shape.len) {
                @memcpy(new_shape[positive_dim + 1 ..], self.shape[positive_dim..]);
            }

            // Update tensor shape
            self.allocator.free(self.shape);
            self.shape = new_shape;
        }

        /// Flattens a specified range of dimensions in the tensor in-place.
        ///
        /// This function modifies the tensor's shape by flattening dimensions from
        /// start_dim to end_dim (inclusive) into a single dimension. Dimensions
        /// before start_dim and after end_dim remain unchanged.
        ///
        /// Parameters:
        /// - start_dim: Starting dimension to flatten (can be negative to count from end)
        /// - end_dim: Ending dimension to flatten (can be negative to count from end)
        ///
        /// Returns: void
        ///
        /// Errors:
        /// - InvalidDimension: If start_dim or end_dim are out of bounds
        /// - InvalidDimRange: If start_dim > end_dim
        pub fn flatten(self: *Self, start_dim: isize, end_dim: isize) !void {
            const positive_start = try helper.normalizeDim(start_dim, self.shape.len);
            const positive_end = try helper.normalizeDim(end_dim, self.shape.len);

            if (positive_start > positive_end) {
                return error.InvalidDimRange;
            }

            // Calculate the size of the flattened dimension
            var flat_size: usize = 1;
            for (positive_start..positive_end + 1) |i| {
                flat_size *= self.shape[i];
            }

            // Create new shape
            const new_shape_len = self.shape.len - (positive_end - positive_start);
            var new_shape = try self.allocator.alloc(usize, new_shape_len);
            errdefer self.allocator.free(new_shape);

            // Copy dimensions before flattened dimensions
            @memcpy(new_shape[0..positive_start], self.shape[0..positive_start]);

            // Add flattened dimension
            new_shape[positive_start] = flat_size;

            // Copy dimensions after flattened dimensions
            if (positive_end + 1 < self.shape.len) {
                @memcpy(
                    new_shape[positive_start + 1 ..],
                    self.shape[positive_end + 1 ..],
                );
            }

            // Free old shape and update with new shape
            self.allocator.free(self.shape);
            self.shape = new_shape;
        }
        pub fn getDimensionSlice(self: Self, dim: usize, index: usize) !Self {
            // Verify dimension is valid
            if (dim >= self.shape.len) {
                return error.InvalidDimension;
            }

            // Verify index is within bounds
            if (index >= self.shape[dim]) {
                return error.IndexOutOfBounds;
            }

            // Special case: 1D tensor becomes scalar (0D tensor)
            if (self.shape.len == 1) {
                var result = try Self.init(self.allocator, &[_]usize{});
                result.data[0] = self.data[index];
                return result;
            }

            // Create new shape by removing the specified dimension
            var new_shape = try self.allocator.alloc(usize, self.shape.len - 1);
            errdefer self.allocator.free(new_shape);

            // Copy shape excluding the specified dimension
            var new_idx: usize = 0;
            for (self.shape, 0..) |size, i| {
                if (i != dim) {
                    new_shape[new_idx] = size;
                    new_idx += 1;
                }
            }

            // Create new tensor with reduced dimensions
            var result = try Self.init(self.allocator, new_shape);
            errdefer result.deinit();
            self.allocator.free(new_shape);

            // Calculate strides for source tensor
            var src_strides = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(src_strides);

            src_strides[self.shape.len - 1] = 1;
            var i: usize = self.shape.len - 1;
            while (i > 0) {
                i -= 1;
                src_strides[i] = src_strides[i + 1] * self.shape[i + 1];
            }

            // For dimensions > 1, calculate destination strides
            if (result.shape.len > 0) {
                var dst_strides = try self.allocator.alloc(usize, result.shape.len);
                defer self.allocator.free(dst_strides);

                dst_strides[result.shape.len - 1] = 1;
                i = result.shape.len - 1;
                while (i > 0) {
                    i -= 1;
                    dst_strides[i] = dst_strides[i + 1] * result.shape[i + 1];
                }

                // Create coordinate arrays
                var src_coords = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(src_coords);
                @memset(src_coords, 0);

                // Set the fixed dimension to the specified index
                src_coords[dim] = index;

                var dst_coords = try self.allocator.alloc(usize, result.shape.len);
                defer self.allocator.free(dst_coords);
                @memset(dst_coords, 0);

                // Copy data
                const total_elements = calculateSize(result.shape);
                var result_idx: usize = 0;

                while (result_idx < total_elements) : (result_idx += 1) {
                    // Calculate source index
                    var src_idx: usize = 0;
                    var dst_coord_idx: usize = 0;

                    for (src_coords, 0..) |*coord, d| {
                        if (d != dim) {
                            coord.* = dst_coords[dst_coord_idx];
                            dst_coord_idx += 1;
                        }
                        src_idx += coord.* * src_strides[d];
                    }

                    // Copy data
                    result.data[result_idx] = self.data[src_idx];

                    // Update destination coordinates
                    var j = dst_coords.len;
                    while (j > 0) {
                        j -= 1;
                        dst_coords[j] += 1;
                        if (dst_coords[j] < result.shape[j]) break;
                        dst_coords[j] = 0;
                    }
                }
            }

            return result;
        }

        // Extension to the Tensor type
        pub fn getSliceRange(self: Self, slices: []const Slice) !Self {
            if (slices.len > self.shape.len) {
                return error.TooManySlices;
            }

            // Calculate new shape
            var new_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(new_shape);

            // Calculate actual start and end indices for each dimension
            var actual_starts = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(actual_starts);
            var actual_ends = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(actual_ends);

            // Initialize with full ranges for dimensions not specified
            for (self.shape, 0..) |dim_size, i| {
                if (i < slices.len) {
                    const slice = slices[i];
                    actual_starts[i] = slice.start orelse 0;
                    actual_ends[i] = slice.end orelse dim_size;

                    // Bounds checking
                    if (actual_starts[i] > dim_size or actual_ends[i] > dim_size) {
                        return error.SliceOutOfBounds;
                    }
                    if (actual_starts[i] > actual_ends[i]) {
                        return error.InvalidSlice;
                    }
                } else {
                    actual_starts[i] = 0;
                    actual_ends[i] = dim_size;
                }
                new_shape[i] = actual_ends[i] - actual_starts[i];
            }

            // Create new tensor with calculated shape
            var result = try Self.init(self.allocator, new_shape);
            errdefer result.deinit();

            // Calculate strides for the original tensor
            var strides = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(strides);

            strides[self.shape.len - 1] = 1;
            var i = self.shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * self.shape[i];
            }

            // Copy data with proper indexing
            var coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(coords);
            @memset(coords, 0);

            var result_idx: usize = 0;
            while (true) {
                // Calculate source index
                var src_idx: usize = 0;
                for (coords, 0..) |coord, dim| {
                    src_idx += (coord + actual_starts[dim]) * strides[dim];
                }

                // Copy data
                result.data[result_idx] = self.data[src_idx];
                result_idx += 1;

                // Update coordinates
                var dim = self.shape.len;
                while (dim > 0) {
                    dim -= 1;
                    coords[dim] += 1;
                    if (coords[dim] < new_shape[dim]) break;
                    coords[dim] = 0;
                    if (dim == 0) return result;
                }
            }

            return result;
        }

        // Calculate index in flattened array from n-dimensional coordinates
        pub fn calculateIndex(shape: []const usize, coords: []const usize) usize {
            var index: usize = 0;
            var stride: usize = 1;
            var i: usize = shape.len;
            while (i > 0) {
                i -= 1;
                index += coords[i] * stride;
                stride *= shape[i];
            }
            return index;
        }

        // Utility functions
        pub fn fill(self: *Self, value: DataType) void {
            for (self.data, 0..) |_, i| {
                self.data[i] = value;
            }
        }

        pub fn copy(self: Self) !Self {
            const new_tensor = try Self.init(self.allocator, self.shape);
            @memcpy(new_tensor.data, self.data);
            return new_tensor;
        }

        /// Format a single value with proper precision
        fn formatValue(value: DataType, writer: anytype, options: PrintOptions) !void {
            switch (@typeInfo(DataType)) {
                .Float => {
                    if (std.math.isNan(value)) {
                        try writer.writeAll("nan");
                    } else if (std.math.isPositiveInf(value)) {
                        try writer.writeAll("inf");
                    } else if (std.math.isNegativeInf(value)) {
                        try writer.writeAll("-inf");
                    } else {
                        const precision = options.precision;
                        switch (precision) {
                            0 => try writer.print("{d:.0}", .{value}),
                            1 => try writer.print("{d:.1}", .{value}),
                            2 => try writer.print("{d:.2}", .{value}),
                            3 => try writer.print("{d:.3}", .{value}),
                            4 => try writer.print("{d:.4}", .{value}),
                            5 => try writer.print("{d:.5}", .{value}),
                            6 => try writer.print("{d:.6}", .{value}),
                            7 => try writer.print("{d:.7}", .{value}),
                            8 => try writer.print("{d:.8}", .{value}),
                            else => try writer.print("{d:.4}", .{value}),
                        }
                    }
                },
                .Int => {
                    try writer.print("{d}", .{value});
                },
                else => {
                    try writer.print("{any}", .{value});
                },
            }
        }

        /// Calculate strides for each dimension
        fn calculateStrides(shape: []const usize, allocator: Allocator) ![]usize {
            var strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);

            if (shape.len == 0) return strides;

            strides[shape.len - 1] = 1;
            var i = shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * shape[i];
            }
            return strides;
        }

        /// Format tensor recursively
        fn formatRecursive(
            self: Self,
            writer: anytype,
            options: PrintOptions,
            current_dim: usize,
            offset: usize,
            strides: []const usize,
        ) !void {
            if (current_dim == self.shape.len) {
                try formatValue(self.data[offset], writer, options);
                return;
            }

            try writer.writeAll("[");
            const dim_size = self.shape[current_dim];

            for (0..dim_size) |i| {
                const new_offset = offset + i * strides[current_dim];
                try formatRecursive(self, writer, options, current_dim + 1, new_offset, strides);
                if (i < dim_size - 1) {
                    try writer.writeAll(", ");
                }
            }

            try writer.writeAll("]");
        }

        /// Internal formatting function
        fn formatTensor(self: Self, writer: anytype) !void {
            const options = PrintOptions{};

            try writer.writeAll("tensor(");

            // Handle empty tensor
            if (self.data.len == 0) {
                try writer.writeAll("[]");
            } else {
                const strides = try calculateStrides(self.shape, self.allocator);
                defer self.allocator.free(strides);

                try formatRecursive(self, writer, options, 0, 0, strides);
            }

            try writer.print(", dtype={s})", .{@typeName(DataType)});
        }

        /// Implement std.fmt.format interface
        pub fn format(
            self: Self,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            try self.formatTensor(writer);
        }

        /// Print the tensor to stdout
        pub fn print(self: Self) void {
            std.debug.print("{}", .{self});
        }
        /// Print a 2D tensor to stdout with truncated rows if necessary
        pub fn print2D(self: Self) void {
            if (self.shape.len != 2) {
                std.debug.print("Error: Not a 2D tensor\n", .{});
                return;
            }

            const rows = self.shape[0];
            const cols = self.shape[1];
            const options = PrintOptions{};

            // Print shape information
            std.debug.print("Tensor {d}x{d}:\n", .{ rows, cols });

            for (0..rows) |i| {
                std.debug.print("[ ", .{});

                // Calculate how many items to show at start and end
                const items_per_side = max_items_per_row / 2;
                const show_ellipsis = cols > max_items_per_row;

                // Print first few items
                const start_items = @min(items_per_side, cols);
                for (0..start_items) |j| {
                    const idx = i * cols + j;
                    formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                    std.debug.print(" ", .{});
                }

                // Print ellipsis if needed
                if (show_ellipsis) {
                    std.debug.print("... ", .{});

                    // Print last few items
                    const remaining_items = @min(items_per_side, cols - start_items);
                    const start_idx = cols - remaining_items;
                    for (start_idx..cols) |j| {
                        const idx = i * cols + j;
                        formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                        std.debug.print(" ", .{});
                    }
                }

                std.debug.print("]\n", .{});
            }
        }

        /// Print a 3D tensor to stdout with truncated rows and slices if necessary
        pub fn print3D(self: Self) void {
            if (self.shape.len != 3) {
                std.debug.print("Error: Not a 3D tensor\n", .{});
                return;
            }

            const depth = self.shape[0];
            const rows = self.shape[1];
            const cols = self.shape[2];
            const options = PrintOptions{};

            // Print shape information
            std.debug.print("Tensor {d}x{d}x{d}:\n", .{ depth, rows, cols });

            // Calculate how many slices to show
            const max_slices = max_rows;
            const show_slice_ellipsis = depth > max_slices;
            const slices_to_show = @min(max_slices, depth);

            for (0..slices_to_show) |d| {
                std.debug.print("\nSlice [{d}]:\n", .{d});

                for (0..rows) |i| {
                    std.debug.print("[ ", .{});

                    // Calculate how many items to show at start and end
                    const items_per_side = max_items_per_row / 2;
                    const show_ellipsis = cols > max_items_per_row;

                    // Print first few items
                    const start_items = @min(items_per_side, cols);
                    for (0..start_items) |j| {
                        const idx = (d * rows * cols) + (i * cols) + j;
                        formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                        std.debug.print(" ", .{});
                    }

                    // Print ellipsis if needed
                    if (show_ellipsis) {
                        std.debug.print("... ", .{});

                        // Print last few items
                        const remaining_items = @min(items_per_side, cols - start_items);
                        const start_idx = cols - remaining_items;
                        for (start_idx..cols) |j| {
                            const idx = (d * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }
                    }

                    std.debug.print("]\n", .{});
                }
            }

            // Print ellipsis for slices if needed
            if (show_slice_ellipsis) {
                std.debug.print("\n...\n\n", .{});

                // Print last slice
                const last_slice = depth - 1;
                std.debug.print("Slice [{d}]:\n", .{last_slice});

                for (0..rows) |i| {
                    std.debug.print("[ ", .{});

                    // Calculate how many items to show at start and end
                    const items_per_side = max_items_per_row / 2;
                    const show_ellipsis = cols > max_items_per_row;

                    // Print first few items
                    const start_items = @min(items_per_side, cols);
                    for (0..start_items) |j| {
                        const idx = (last_slice * rows * cols) + (i * cols) + j;
                        formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                        std.debug.print(" ", .{});
                    }

                    // Print ellipsis if needed
                    if (show_ellipsis) {
                        std.debug.print("... ", .{});

                        // Print last few items
                        const remaining_items = @min(items_per_side, cols - start_items);
                        const start_idx = cols - remaining_items;
                        for (start_idx..cols) |j| {
                            const idx = (last_slice * rows * cols) + (i * cols) + j;
                            formatValue(self.data[idx], std.io.getStdOut().writer(), options) catch {};
                            std.debug.print(" ", .{});
                        }
                    }

                    std.debug.print("]\n", .{});
                }
            }
        }

        /// Convert tensor to string
        pub fn toString(self: Self, allocator: std.mem.Allocator) ![]const u8 {
            var list = std.ArrayList(u8).init(allocator);
            errdefer list.deinit();

            try list.writer().print("{}", .{self});
            return list.toOwnedSlice();
        }

        /// StabilityInfo struct to hold detailed information about tensor stability
        pub const StabilityInfo = struct {
            has_nan: bool = false,
            has_pos_inf: bool = false,
            has_neg_inf: bool = false,
            first_nan_index: ?usize = null,
            first_pos_inf_index: ?usize = null,
            first_neg_inf_index: ?usize = null,
            nan_count: usize = 0,
            pos_inf_count: usize = 0,
            neg_inf_count: usize = 0,
        };
    };
}

/// Helper struct to hold formatting options
const PrintOptions = struct {
    precision: u8 = 8,
};
pub const StabilityError = error{
    HasNaN,
    HasPositiveInfinity,
    HasNegativeInfinity,
    HasInfinity,
};

// Slice type to represent Python-style slice operations
pub const Slice = struct {
    start: ?usize,
    end: ?usize,

    pub fn full() Slice {
        return .{ .start = null, .end = null };
    }

    pub fn from(start: ?usize, end: ?usize) Slice {
        return .{ .start = start, .end = end };
    }
};
