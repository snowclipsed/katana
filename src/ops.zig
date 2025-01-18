const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const tensormod = @import("tensor.zig");
const Tensor = tensormod.Tensor;
const StabilityError = tensormod.StabilityError;
const Slice = tensormod.Slice;
const testing = std.testing;
const builtin = @import("builtin");
const atomic = std.atomic;
const expectEqual = testing.expectEqual;
const expectError = testing.expectError;

// --- Constants ---
const max_items_per_row = 6; // Number of elements to show per row
const max_rows = 8; // Maximum number of rows to show before truncating

/// SIMD Tuning parameters
/// Tile size for SIMD matrix multiplication.
/// The tile size is used to partition the input matrices into smaller submatrices
/// that can be loaded into the cache and processed efficiently.
/// The tile size should be chosen based on the cache size and the size of the
/// input matrices to maximize cache utilization and minimize cache misses.
/// Default value is 64.
pub const Tile: usize = 160; // Tile size for matrix blocking
pub const Vec: usize = 8; // Vector size for SIMD operations

const CACHE_LINE_SIZE: usize = atomic.cache_line;
const CHUNK_SIZE: usize = 1;
const AVX2_ALIGNMENT = 32;
const MICRO_KERNEL_SIZE: usize = Vec; // Match micro-kernel to vector size

const Vec8f = @Vector(8, f32);

/// Vector size for SIMD operations.
/// The vector size is used to load and process multiple elements in parallel
/// using SIMD instructions. The vector size should be chosen based on the
/// target SIMD architecture to maximize performance.
/// Default value is 32.

//--------------------------------- Transformation Operations ---------------------------------

/// Transposes a 2D tensor in-place.
///
/// This function takes a tensor of type `T` and transposes it, effectively
/// swapping its rows and columns. The tensor must be 2-dimensional; otherwise,
/// an `UnsupportedDimension` error is returned.
///
/// The function allocates new memory for the transposed data, copies the
/// transposed elements into this new memory, frees the old data, and updates
/// the tensor's data pointer and shape to reflect the transposition.
///
/// ## Parameters:
/// - `T`: The type of the elements in the tensor.
/// - `tensor`: A pointer to the tensor to be transposed.
///
/// ## Returns:
/// - `!void`: Returns `void` on success, or an error if the tensor is not
///   2-dimensional or if memory allocation fails.
///
/// ## Errors:
/// - `UnsupportedDimension`: The tensor is not 2-dimensional.
/// - Any error that can be returned by the allocator's `alignedAlloc` method.
///
/// ## Example:
/// ```zig
/// const std = @import("std");
/// const Tensor = @import("tensor.zig").Tensor;
/// const allocator = std.heap.page_allocator;
///
/// var tensor = Tensor(f32).init(allocator, .{2, 3});
/// tensor.data[0] = 1.0;
/// tensor.data[1] = 2.0;
/// tensor.data[2] = 3.0;
/// tensor.data[3] = 4.0;
/// tensor.data[4] = 5.0;
/// tensor.data[5] = 6.0;
///
/// try transpose(f32, &tensor);
///
/// // tensor.shape is now .{3, 2}
/// // tensor.data is now .{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}
/// ```
// Tensor Operations
pub fn transpose(comptime T: type, tensor: *Tensor(T)) !void {
    if (tensor.shape.len != 2) return error.UnsupportedDimension;

    const rows = tensor.shape[0];
    const cols = tensor.shape[1];
    var new_data = try tensor.allocator.alignedAlloc(@TypeOf(tensor.data[0]), 32, rows * cols);

    for (0..rows) |i| {
        for (0..cols) |j| {
            new_data[j * rows + i] = tensor.data[i * cols + j];
        }
    }

    tensor.allocator.free(tensor.data);
    tensor.data = new_data;

    // Swap dimensions
    const temp = tensor.shape[0];
    tensor.shape[0] = tensor.shape[1];
    tensor.shape[1] = temp;
}

/// Transposes a tensor by swapping specified dimensions.
///
/// This function takes a tensor and two dimensions, and swaps the specified dimensions
/// to produce a transposed tensor. The function performs the following steps:
/// 1. Validates the dimensions to ensure they are within the bounds of the tensor's shape.
/// 2. Calculates the strides for the current shape of the tensor.
/// 3. Creates a new shape with the specified dimensions swapped.
/// 4. Allocates memory for the transposed data.
/// 5. Calculates the new strides for the transposed shape.
/// 6. Creates coordinate arrays to keep track of the element positions.
/// 7. Performs the transpose operation by iterating over each element, calculating the
///    source coordinates, swapping the coordinates for the transposed dimensions, and
///    copying the data to the new transposed tensor.
/// 8. Updates the tensor with the new data and shape.
///
/// Parameters:
/// - `T`: The type of the elements in the tensor.
/// - `tensor`: A pointer to the tensor to transpose.
/// - `dim0`: The first dimension to swap.
/// - `dim1`: The second dimension to swap.
///
/// Returns:
/// - `!void`: Returns an error if the dimensions are invalid or if memory allocation fails.
///
/// Errors:
/// - `error.InvalidDimension`: If either `dim0` or `dim1` is out of bounds of the tensor's shape.
/// - `error.OutOfMemory`: If memory allocation fails.
pub fn transposeAxes(comptime T: type, tensor: *Tensor(T), dim0: usize, dim1: usize) !void {
    if (dim0 >= tensor.shape.len or dim1 >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    // Calculate strides for the current shape
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    strides[tensor.shape.len - 1] = 1;
    var i: usize = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }

    // Create new shape with swapped dimensions
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |dim, idx| {
        if (idx == dim0) {
            new_shape[idx] = tensor.shape[dim1];
        } else if (idx == dim1) {
            new_shape[idx] = tensor.shape[dim0];
        } else {
            new_shape[idx] = dim;
        }
    }

    // Allocate memory for transposed data
    var new_data = try tensor.allocator.alignedAlloc(T, 32, tensor.data.len);
    errdefer tensor.allocator.free(new_data);

    // Calculate new strides
    var new_strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(new_strides);

    new_strides[tensor.shape.len - 1] = 1;
    i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        new_strides[i - 1] = new_strides[i] * new_shape[i];
    }

    // Create coordinate arrays
    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    // Perform the transpose operation
    const total_elements = tensor.data.len;
    var idx: usize = 0;
    while (idx < total_elements) : (idx += 1) {
        // Calculate source coordinates
        var remaining = idx;
        for (0..tensor.shape.len) |dim| {
            coords[dim] = remaining / new_strides[dim];
            remaining = remaining % new_strides[dim];
        }

        // Swap coordinates for the transposed dimensions
        const temp = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp;

        // Calculate source index using original strides
        var src_idx: usize = 0;
        for (0..tensor.shape.len) |dim| {
            src_idx += coords[dim] * strides[dim];
        }

        new_data[idx] = tensor.data[src_idx];
    }

    // Update tensor with new data and shape
    tensor.allocator.free(tensor.data);
    tensor.data = new_data;
    tensor.allocator.free(tensor.shape);
    tensor.shape = new_shape;
}

/// Accumulates the values of `other` tensor into the `tensor` in-place.
///
/// This function performs an element-wise addition of the `other` tensor to the `tensor`
/// and then accumulates the result in a cumulative sum fashion.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensor`: A pointer to the tensor that will be modified in-place.
/// - `other`: The tensor whose values will be added to `tensor`.
///
/// # Returns
/// - `void`: Returns nothing on success.
///
/// # Errors
/// - `ShapeMismatch`: If the shapes of `tensor` and `other` do not match.
///
/// # Example
/// ```zig
/// var tensor = Tensor(f32, .{2, 2}, .{1.0, 2.0, 3.0, 4.0});
/// var other = Tensor(f32, .{2, 2}, .{0.5, 1.5, 2.5, 3.5});
/// try accumulate(f32, &tensor, other);
/// // tensor.data is now {1.5, 4.0, 9.0, 16.0}
/// ```
///
/// # Notes
/// - The function assumes that the `tensor` and `other` have the same shape.
/// - The function performs an in-place modification of the `tensor`.
pub fn accumulate(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        // std.log.err("tensor shape: {d}\n", .{tensor.shape});
        // std.log.err("other shape: {d}\n", .{other.shape});
        // std.log.err("Error during accumulation \n", .{});
        return error.ShapeMismatch;
    }

    var temp = try tensor.copy();
    defer temp.deinit();

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] = temp.data[i] + other.data[i];
        if (i > 0) {
            tensor.data[i] += tensor.data[i - 1];
        }
    }
}

/// Gets a chunk of a tensor along a specified dimension.
///
/// This function divides a tensor into equal-sized chunks along a specified dimension and returns the chunk at the given index.
///
/// # Parameters
/// - `T`: The type of the elements in the tensor.
/// - `tensor`: The input tensor to be chunked.
/// - `dim`: The dimension along which to chunk the tensor.
/// - `chunk_idx`: The index of the chunk to retrieve.
/// - `num_chunks`: The total number of chunks to divide the tensor into.
///
/// # Returns
/// - A tensor containing the specified chunk of the input tensor.
///
/// # Errors
/// - `error.InvalidDimension`: If the specified dimension is out of bounds.
/// - `error.InvalidNumChunks`: If the number of chunks is zero or greater than the size of the specified dimension.
/// - `error.InvalidChunkIndex`: If the chunk index is out of bounds.
/// - `error.UnevenChunkSize`: If the tensor cannot be evenly divided into the specified number of chunks.
///
/// # Example
/// ```zig
/// const tensor = Tensor(f32).initFromSlice(allocator, &[2, 6], &[_]f32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
/// const chunk = try getChunk(f32, tensor, 1, 0, 3);
/// // chunk now contains a tensor with shape [2, 2] and data [1, 2, 3, 4]
/// ```
pub fn getChunk(comptime T: type, tensor: Tensor(T), dim: usize, chunk_idx: usize, num_chunks: usize) !Tensor(T) {
    // Validate inputs
    if (dim >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    const dim_size = tensor.shape[dim];
    if (num_chunks == 0 or dim_size < num_chunks) {
        return error.InvalidNumChunks;
    }

    if (chunk_idx >= num_chunks) {
        return error.InvalidChunkIndex;
    }

    // Calculate chunk size and start/end indices
    const chunk_size = dim_size / num_chunks;
    if (chunk_size * num_chunks != dim_size) {
        return error.UnevenChunkSize;
    }

    const start_idx = chunk_idx * chunk_size;

    // Create new shape array
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |s, i| {
        new_shape[i] = if (i == dim) chunk_size else s;
    }

    // Create result tensor
    var result = try Tensor(T).init(tensor.allocator, new_shape);
    tensor.allocator.free(new_shape);
    errdefer result.deinit();

    // Calculate strides for the input tensor
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    strides[tensor.shape.len - 1] = 1;
    var i = tensor.shape.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * tensor.shape[i];
    }

    // Copy data
    const total_elements = result.data.len;
    var result_idx: usize = 0;
    var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(coords);
    @memset(coords, 0);

    while (result_idx < total_elements) : (result_idx += 1) {
        // Calculate source coordinates
        var temp = result_idx;
        var src_idx: usize = 0;

        for (0..tensor.shape.len) |j| {
            const rev_j = tensor.shape.len - 1 - j;
            if (rev_j == dim) {
                coords[rev_j] = temp % chunk_size + start_idx;
            } else {
                coords[rev_j] = temp % tensor.shape[rev_j];
            }
            src_idx += coords[rev_j] * strides[rev_j];
            temp /= if (rev_j == dim) chunk_size else tensor.shape[rev_j];
        }

        result.data[result_idx] = tensor.data[src_idx];
    }

    return result;
}

/// Concatenates two tensors along a specified dimension.
///
/// This function takes two tensors of the same type and concatenates them along the specified dimension.
/// The resulting tensor will have a shape that is the same as the input tensors, except for the specified
/// dimension, which will be the sum of the sizes of the input tensors along that dimension.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensor`: The first tensor to concatenate.
/// - `other`: The second tensor to concatenate.
/// - `dim`: The dimension along which to concatenate the tensors.
///
/// # Returns
/// A new tensor that is the result of concatenating the input tensors along the specified dimension.
///
/// # Errors
/// This function will return an error if the tensors cannot be concatenated due to incompatible shapes or
/// if there is an allocation failure.
///
/// # Example
/// ```zig
/// const std = @import("std");
/// const Tensor = @import("tensor.zig").Tensor;
/// const concat = @import("ops.zig").concat;
///
/// var gpa = std.heap.GeneralPurposeAllocator(.{}){};
/// defer _ = gpa.deinit();
/// const allocator = gpa.allocator;
///
/// var tensor1 = try Tensor(f32).init(allocator, &[_]usize{2, 3});
/// defer tensor1.deinit();
/// tensor1.data = &[_]f32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
///
/// var tensor2 = try Tensor(f32).init(allocator, &[_]usize{2, 3});
/// defer tensor2.deinit();
/// tensor2.data = &[_]f32{7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
///
/// const result = try concat(f32, tensor1, tensor2, 0);
/// defer result.deinit();
///
/// // The resulting tensor will have shape [4, 3] and data [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
/// std.debug.print("Result shape: {}\n", .{result.shape});
/// std.debug.print("Result data: {}\n", .{result.data});
/// ```
///
/// # Notes
/// - The function assumes that the input tensors have the same shape except for the specified dimension.
/// - The function allocates memory for the new tensor and its shape, so it is important to free the allocated
///   memory using the `deinit` method of the resulting tensor.
pub fn concat(comptime T: type, tensor: Tensor(T), other: Tensor(T), dim: usize) !Tensor(T) {
    // Verify tensors can be concatenated
    try verifyCompatibleForConcat(T, tensor, other, dim);

    // Calculate new shape
    var new_shape = try tensor.allocator.alloc(usize, tensor.shape.len);
    errdefer tensor.allocator.free(new_shape);

    for (tensor.shape, 0..) |s, i| {
        new_shape[i] = if (i == dim) s + other.shape[i] else s;
    }

    // Create new tensor with combined shape
    var result = try Tensor(T).init(tensor.allocator, new_shape);
    errdefer result.deinit();
    tensor.allocator.free(new_shape);

    // Early return for zero-sized tensors
    if (calculateSize(result.shape) == 0) {
        return result;
    }

    // Helper function to get strides
    var strides = try tensor.allocator.alloc(usize, tensor.shape.len);
    defer tensor.allocator.free(strides);

    // Calculate strides for the result tensor
    strides[strides.len - 1] = 1;
    var i: usize = strides.len - 1;
    while (i > 0) {
        i -= 1;
        strides[i] = strides[i + 1] * result.shape[i + 1];
    }

    // Copy data from first tensor
    const first_size = calculateSize(tensor.shape);
    if (first_size > 0) {
        var coords = try tensor.allocator.alloc(usize, tensor.shape.len);
        defer tensor.allocator.free(coords);
        @memset(coords, 0);

        var idx: usize = 0;
        while (idx < first_size) : (idx += 1) {
            // Calculate source and destination indices
            var src_idx: usize = 0;
            var dst_idx: usize = 0;

            for (coords, 0..) |c, j| {
                if (j == dim) {
                    src_idx += c * (if (j + 1 < tensor.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..tensor.shape.len) |k| {
                            prod *= tensor.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += c * strides[j];
                } else {
                    src_idx += c * (if (j + 1 < tensor.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..tensor.shape.len) |k| {
                            prod *= tensor.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += c * strides[j];
                }
            }

            result.data[dst_idx] = tensor.data[src_idx];

            // Update coordinates
            var j = coords.len;
            while (j > 0) {
                j -= 1;
                coords[j] += 1;
                if (coords[j] < tensor.shape[j]) break;
                coords[j] = 0;
            }
        }
    }

    // Copy data from second tensor
    const second_size = calculateSize(other.shape);
    if (second_size > 0) {
        var coords = try tensor.allocator.alloc(usize, other.shape.len);
        defer tensor.allocator.free(coords);
        @memset(coords, 0);

        var idx: usize = 0;
        while (idx < second_size) : (idx += 1) {
            // Calculate source and destination indices
            var src_idx: usize = 0;
            var dst_idx: usize = 0;

            for (coords, 0..) |c, j| {
                if (j == dim) {
                    src_idx += c * (if (j + 1 < other.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..other.shape.len) |k| {
                            prod *= other.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += (c + tensor.shape[dim]) * strides[j];
                } else {
                    src_idx += c * (if (j + 1 < other.shape.len) blk: {
                        var prod: usize = 1;
                        for (j + 1..other.shape.len) |k| {
                            prod *= other.shape[k];
                        }
                        break :blk prod;
                    } else 1);
                    dst_idx += c * strides[j];
                }
            }

            result.data[dst_idx] = other.data[src_idx];

            // Update coordinates
            var j = coords.len;
            while (j > 0) {
                j -= 1;
                coords[j] += 1;
                if (coords[j] < other.shape[j]) break;
                coords[j] = 0;
            }
        }
    }

    return result;
}

fn verifyCompatibleForConcat(comptime T: type, tensor: Tensor(T), other: Tensor(T), dim: usize) !void {
    // Check if dimension is valid
    if (dim >= tensor.shape.len) {
        return error.InvalidDimension;
    }

    // Check if tensors have same number of dimensions
    if (tensor.shape.len != other.shape.len) {
        return error.DimensionMismatch;
    }

    // Check if all dimensions except concat dim are equal
    for (tensor.shape, 0..) |s, i| {
        if (i != dim and s != other.shape[i]) {
            // std.log.err("tensor shape: {d}\n", .{tensor.shape});
            // std.log.err("other shape: {d}\n", .{other.shape});
            return error.IncompatibleShapes;
        }
    }
}

/// Stacks a list of tensors along a specified dimension.
///
/// This function takes a list of tensors with the same shape and stacks them
/// along a new dimension, creating a new tensor with an additional dimension.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensors`: A list of tensors to be stacked. All tensors must have the same shape.
/// - `dim`: The dimension along which to stack the tensors. Must be less than or equal to the number of dimensions in the input tensors.
///
/// # Returns
/// - `Tensor(T)`: A new tensor with an additional dimension, containing the stacked tensors.
///
/// # Errors
/// - `error.EmptyTensorList`: If the input list of tensors is empty.
/// - `error.ShapeMismatch`: If the input tensors do not all have the same shape.
/// - `error.InvalidDimension`: If the specified dimension is greater than the number of dimensions in the input tensors.
///
/// # Example
/// ```zig
/// const tensor1 = Tensor(f32).init(...);
/// const tensor2 = Tensor(f32).init(...);
/// const stacked = try stack(f32, &[tensor1, tensor2], 0);
/// ```
///
/// # Notes
/// - The function allocates memory for the new tensor shape and strides, which is freed before returning.
/// - The function calculates the strides for the result tensor to facilitate copying data from the input tensors.
pub fn stack(comptime T: type, tensors: []const Tensor(T), dim: usize) !Tensor(T) {
    if (tensors.len == 0) {
        return error.EmptyTensorList;
    }

    const ref_tensor = tensors[0];
    const ref_shape = ref_tensor.shape;

    // Validate all tensors have the same shape
    for (tensors[1..]) |tensor| {
        if (!std.mem.eql(usize, tensor.shape, ref_shape)) {
            // std.log.err("Error during stacking \n", .{});
            return error.ShapeMismatch;
        }
    }

    // Validate dimension
    if (dim > ref_shape.len) {
        return error.InvalidDimension;
    }

    // Create new shape with extra dimension
    var new_shape = try ref_tensor.allocator.alloc(usize, ref_shape.len + 1);
    errdefer ref_tensor.allocator.free(new_shape);

    // Copy shape and insert new dimension
    var src_shape_idx: usize = 0;
    var dst_shape_idx: usize = 0;
    while (dst_shape_idx < new_shape.len) : (dst_shape_idx += 1) {
        if (dst_shape_idx == dim) {
            new_shape[dst_shape_idx] = tensors.len; // Size of new dimension
        } else {
            new_shape[dst_shape_idx] = ref_shape[src_shape_idx];
            src_shape_idx += 1;
        }
    }

    // Create result tensor
    var result = try Tensor(T).init(ref_tensor.allocator, new_shape);
    errdefer result.deinit();
    ref_tensor.allocator.free(new_shape);

    // Calculate strides for the result tensor
    var strides = try ref_tensor.allocator.alloc(usize, result.shape.len);
    defer ref_tensor.allocator.free(strides);

    strides[strides.len - 1] = 1;
    var i = strides.len - 1;
    while (i > 0) : (i -= 1) {
        strides[i - 1] = strides[i] * result.shape[i];
    }

    // Copy data from each input tensor
    var coords = try ref_tensor.allocator.alloc(usize, result.shape.len);
    defer ref_tensor.allocator.free(coords);
    @memset(coords, 0);

    const elements_per_tensor = calculateSize(ref_shape);

    // For each input tensor
    for (tensors, 0..) |tensor, tensor_idx| {
        var element_idx: usize = 0;
        while (element_idx < elements_per_tensor) : (element_idx += 1) {
            // Calculate source coordinates (excluding stacked dimension)
            var temp = element_idx;
            var src_coords = try ref_tensor.allocator.alloc(usize, ref_shape.len);
            defer ref_tensor.allocator.free(src_coords);

            var j = ref_shape.len;
            while (j > 0) : (j -= 1) {
                src_coords[j - 1] = temp % ref_shape[j - 1];
                temp /= ref_shape[j - 1];
            }

            // Calculate destination coordinates (including stacked dimension)
            var final_dst_idx: usize = 0;
            var coord_idx: usize = 0;
            for (coords, 0..) |*c, idx| {
                if (idx == dim) {
                    c.* = tensor_idx;
                } else {
                    c.* = src_coords[coord_idx];
                    coord_idx += 1;
                }
                final_dst_idx += c.* * strides[idx];
            }

            // Copy the value
            result.data[final_dst_idx] = tensor.data[element_idx];

            // Update coordinates for next iteration
            var k = coords.len;
            while (k > 0) {
                k -= 1;
                if (k == dim) continue; // Skip the stacked dimension
                coords[k] += 1;
                if (coords[k] < result.shape[k]) break;
                coords[k] = 0;
            }
        }
    }

    return result;
}

/// Convert a potentially negative dimension index to a positive index.
///
/// This function takes a dimension index `dim` which can be negative, and the total number of dimensions `n_dims`.
/// If `dim` is negative, it is converted to a positive index by adding it to `n_dims`.
/// If the resulting index is out of bounds, an `InvalidDimension` error is returned.
///
/// Parameters:
/// - `dim`: The dimension index, which can be negative.
/// - `n_dims`: The total number of dimensions, which must be a positive integer.
///
/// Returns:
/// - The positive dimension index if `dim` is within bounds.
///
/// Errors:
/// - `InvalidDimension`: If the resulting dimension index is out of bounds.
pub fn normalizeDim(dim: isize, n_dims: usize) !usize {
    const n_dims_i: isize = @intCast(n_dims);
    if (dim >= 0) {
        if (dim >= n_dims_i) return error.InvalidDimension;
        return @intCast(dim);
    } else {
        const positive_dim = n_dims_i + dim; // -1 becomes n_dims-1
        if (positive_dim < 0 or positive_dim >= n_dims_i) return error.InvalidDimension;
        return @intCast(positive_dim);
    }
}

/// Flattens dimensions from start_dim to end_dim (inclusive)
/// TODO: Convert to tensor intrinsic
// pub fn flatten(comptime T: type, tensor: *Tensor(T), start_dim: isize, end_dim: isize) !void {
//     const positive_start = try normalizeDim(start_dim, tensor.shape.len);
//     const positive_end = try normalizeDim(end_dim, tensor.shape.len);

//     if (positive_start > positive_end) {
//         return error.InvalidDimRange;
//     }

//     // Calculate the size of the flattened dimension
//     var flat_size: usize = 1;
//     for (positive_start..positive_end + 1) |i| {
//         flat_size *= tensor.shape[i];
//     }

//     // Create new shape
//     const new_shape_len = tensor.shape.len - (positive_end - positive_start);
//     var new_shape = try tensor.allocator.alloc(usize, new_shape_len);
//     errdefer tensor.allocator.free(new_shape);

//     // Copy dimensions before flattened dimensions
//     @memcpy(new_shape[0..positive_start], tensor.shape[0..positive_start]);

//     // Add flattened dimension
//     new_shape[positive_start] = flat_size;

//     // Copy dimensions after flattened dimensions
//     if (positive_end + 1 < tensor.shape.len) {
//         @memcpy(
//             new_shape[positive_start + 1 ..],
//             tensor.shape[positive_end + 1 ..],
//         );
//     }

//     // Free old shape and update with new shape
//     tensor.allocator.free(tensor.shape);
//     tensor.shape = new_shape;
// }

// Usage example:
pub fn stackAndFlatten(comptime T: type, r: Tensor(T), i: Tensor(T), dim: isize) !Tensor(T) {
    // Convert negative dimension to positive
    const positive_dim = if (dim >= 0)
        @as(usize, @intCast(dim))
    else blk: {
        const n_dims: isize = @intCast(r.shape.len);
        // -1 means last dimension + 1 (where we'll insert)
        const adjusted_dim = n_dims + 1 + dim;
        if (adjusted_dim < 0) return error.InvalidDimension;
        break :blk @as(usize, @intCast(adjusted_dim));
    };

    // Stack the tensors along specified dimension
    var tensors = [_]Tensor(T){ r, i };
    var result = try stack(T, &tensors, positive_dim);
    errdefer result.deinit();

    // Flatten the last two dimensions
    try result.flatten(@intCast(result.shape.len - 2), @intCast(result.shape.len - 1));

    return result;
}

fn calculateSize(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |dim| {
        size *= dim;
    }
    return size;
}

/// Generates a tensor with random values between -1 and 1.
///
/// This function creates a tensor of the specified shape and fills it with
/// random values of type `T` between -1 and 1 using a seeded random number generator.
///
/// - Parameters:
///   - T: The type of the elements in the tensor.
///   - allocator: The allocator to use for memory allocation.
///   - shape: The shape of the tensor as an array of `usize`.
///   - seed: The seed for the random number generator.
/// - Returns: A tensor of type `T` with random values between -1 and 1.
/// - Throws: Returns an error if tensor initialization fails.
pub fn randomTensor(comptime T: type, allocator: std.mem.Allocator, shape: []const usize, seed: u64) !Tensor(T) {
    var tensor = try Tensor(T).init(allocator, shape);
    errdefer tensor.deinit();

    var rng = std.rand.DefaultPrng.init(seed);
    for (tensor.data) |*val| {
        val.* = rng.random().float(T) * 2.0 - 1.0; // Values between -1 and 1
    }
    return tensor;
}

/// Creates a tensor filled with zeros.
///
/// This function allocates memory for a tensor of the specified shape and initializes all elements to zero.
///
/// Parameters:
/// - `T`: The type of the elements in the tensor.
/// - `allocator`: The allocator to use for memory allocation.
/// - `shape`: An array specifying the shape of the tensor.
///
/// Returns:
/// - A `Tensor(T)` instance with all elements initialized to zero.
///
/// Errors:
/// - Returns an error if memory allocation fails.
///
/// Example:
/// ```zig
/// const std = @import("std");
/// const Tensor = @import("tensor.zig").Tensor;
/// const zeros = @import("ops.zig").zeros;
///
/// const allocator = std.heap.page_allocator;
/// const shape = &[_]usize{2, 3};
/// const tensor = try zeros(f32, allocator, shape);
/// defer tensor.deinit();
/// ```
pub fn zeros(comptime T: type, allocator: Allocator, shape: []const usize) !Tensor(T) {
    // Calculate total size
    var total_size: usize = 1;
    for (shape) |dim| {
        total_size *= dim;
    }

    // Allocate aligned data array
    const alignment = 32;
    const data = try allocator.alignedAlloc(T, alignment, total_size);
    // Initialize all elements to zero
    @memset(data, 0);

    // Create tensor shape
    const tensor_shape = try allocator.alloc(usize, shape.len);
    @memcpy(tensor_shape, shape);

    // Return initialized tensor
    return Tensor(T){
        .data = data,
        .shape = tensor_shape,
        .allocator = allocator,
    };
}

// ----------------------- Safety Checks ----------------------------

/// Calculate the index in a flattened array from n-dimensional coordinates.
///
/// This function takes the shape of an n-dimensional array and the coordinates
/// within that array, and calculates the corresponding index in the flattened
/// (1-dimensional) representation of the array.
///
/// # Parameters
/// - `shape`: A slice of `usize` representing the dimensions of the n-dimensional array.
/// - `coords`: A slice of `usize` representing the coordinates within the n-dimensional array.
///
/// # Returns
/// - `usize`: The index in the flattened array corresponding to the given coordinates.
///
/// # Example
/// ```zig
/// const shape = [_]usize{3, 4, 5}; // 3x4x5 array
/// const coords = [_]usize{2, 1, 3}; // Coordinates in the 3x4x5 array
/// const index = calculateIndex(shape, coords); // index will be 53
/// ```
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

/// Checks the stability of the given tensor by inspecting its elements for NaN, positive infinity, and negative infinity values.
///
/// This function retrieves stability information for the tensor and returns an appropriate error if any instability is detected.
///
/// Parameters:
/// - `T`: The type of the elements in the tensor.
/// - `tensor`: The tensor to be checked.
///
/// Returns:
/// - `StabilityError.HasNaN` if the tensor contains NaN values.
/// - `StabilityError.HasPositiveInfinity` if the tensor contains positive infinity values.
/// - `StabilityError.HasNegativeInfinity` if the tensor contains negative infinity values.
///
/// Errors:
/// - Returns an error if the stability information cannot be retrieved.
pub fn checkStability(comptime T: type, tensor: Tensor(T)) !void {
    const info = try getStabilityInfo(T, tensor);
    if (info.has_nan) {
        return StabilityError.HasNaN;
    }
    if (info.has_pos_inf) {
        return StabilityError.HasPositiveInfinity;
    }
    if (info.has_neg_inf) {
        return StabilityError.HasNegativeInfinity;
    }
}

/// Analyzes the stability of a tensor by checking for NaN, positive infinity, and negative infinity values.
///
/// This function iterates over the elements of the given tensor and collects information about the presence
/// of NaN, positive infinity, and negative infinity values. It returns a `StabilityInfo` struct containing
/// the results of this analysis.
///
/// ## Parameters
/// - `T`: The type of the elements in the tensor. This is a compile-time parameter.
/// - `tensor`: The tensor to be analyzed.
///
/// ## Returns
/// - `Tensor(T).StabilityInfo`: A struct containing information about the stability of the tensor, including
///   counts and indices of NaN, positive infinity, and negative infinity values.
///
/// ## Errors
/// This function does not return any errors.
///
/// ## Example
/// ```zig
/// const tensor = Tensor(f32){ .data = [_]f32{ 1.0, std.math.nan, std.math.inf, -std.math.inf } };
/// const info = try getStabilityInfo(f32, tensor);
pub fn getStabilityInfo(comptime T: type, tensor: Tensor(T)) !Tensor(T).StabilityInfo {
    var info = Tensor(@TypeOf(tensor.data[0])).StabilityInfo{};

    switch (@typeInfo(@TypeOf(tensor.data[0]))) {
        .Float => {
            for (tensor.data, 0..) |value, i| {
                if (std.math.isNan(value)) {
                    info.has_nan = true;
                    info.nan_count += 1;
                    if (info.first_nan_index == null) {
                        info.first_nan_index = i;
                    }
                } else if (std.math.isPositiveInf(value)) {
                    info.has_pos_inf = true;
                    info.pos_inf_count += 1;
                    if (info.first_pos_inf_index == null) {
                        info.first_pos_inf_index = i;
                    }
                } else if (std.math.isNegativeInf(value)) {
                    info.has_neg_inf = true;
                    info.neg_inf_count += 1;
                    if (info.first_neg_inf_index == null) {
                        info.first_neg_inf_index = i;
                    }
                }
            }
        },
        else => {},
    }

    return info;
}

/// Checks if the given tensor is stable, meaning it does not contain any NaN, positive infinity, or negative infinity values.
///
/// This function retrieves stability information for the tensor and verifies that it does not contain any NaN, positive infinity, or negative infinity values.
///
/// - Parameters:
///   - T: The type of the elements in the tensor.
///   - tensor: The tensor to check for stability.
/// - Returns: A boolean indicating whether the tensor is stable.
/// - Throws: An error if retrieving the stability information fails.
pub fn isStable(comptime T: type, tensor: Tensor(T)) !bool {
    const info = try getStabilityInfo(T, tensor);
    return !info.has_nan and !info.has_pos_inf and !info.has_neg_inf;
}

/// Checks if the given tensor contains any NaN (Not-a-Number) values.
///
/// This function takes a tensor of a specified type and checks if it contains
/// any NaN values. It returns a boolean indicating the presence of NaN values.
///
/// - Parameters:
///   - T: The type of the elements in the tensor.
///   - tensor: The tensor to be checked for NaN values.
/// - Returns: A boolean indicating whether the tensor contains NaN values.
/// - Throws: An error if there is an issue retrieving stability information for the tensor.
pub fn hasNaN(comptime T: type, tensor: Tensor(T)) !bool {
    const info = try getStabilityInfo(T, tensor);
    return info.has_nan;
}

/// Checks if the given tensor contains any positive or negative infinity values.
///
/// This function examines the stability information of the tensor to determine
/// if it contains any positive or negative infinity values.
///
/// - Parameters:
///   - T: The type of the elements in the tensor.
///   - tensor: The tensor to be checked for infinity values.
///
/// - Returns: A boolean indicating whether the tensor contains any positive or
///   negative infinity values.
///
/// - Throws: An error if retrieving the stability information fails.
pub fn hasInf(comptime T: type, tensor: Tensor(T)) !bool {
    const info = try getStabilityInfo(T, tensor);
    return info.has_pos_inf or info.has_neg_inf;
}

/// Replaces all NaN or Infinity values in the given tensor with a specified replacement value.
/// This function only operates on tensors with floating-point data types.
///
/// ## Parameters:
/// - `T`: The type of the elements in the tensor. This must be a floating-point type.
/// - `tensor`: A pointer to the tensor whose NaN or Infinity values are to be replaced.
/// - `replacement`: The value to replace NaN or Infinity values with.
///
/// ## Errors:
/// This function does not return any errors.
///
/// ## Example:
/// ```zig
/// const std = @import("std");
/// const Tensor = @import("tensor.zig").Tensor;
/// const ops = @import("ops.zig");
///
/// var tensor = Tensor(f32).init([3]f32{ std.math.nan, 1.0, std.math.inf });
/// try ops.replaceUnstable(f32, &tensor, 0.0);
/// assert(tensor.data[0] == 0.0);
/// assert(tensor.data[2] == 0.0);
/// ```
pub fn replaceUnstable(comptime T: type, tensor: *Tensor(T), replacement: T) !void {
    switch (@typeInfo(@TypeOf(tensor.data[0]))) {
        .Float => {
            for (tensor.data) |*value| {
                if (std.math.isNan(value.*) or std.math.isInf(value.*)) {
                    value.* = replacement;
                }
            }
        },
        else => {},
    }
}

// ------------------------ Math Operations --------------------------------------

/// Adds the elements of one tensor to another tensor element-wise.
///
/// This function performs an element-wise addition of the elements in `other` tensor
/// to the corresponding elements in the `tensor`. Both tensors must have the same shape.
///
/// If the shapes of the two tensors do not match, an error of type `ShapeMismatch` is returned.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensor`: A pointer to the tensor to which the elements of `other` will be added.
/// - `other`: The tensor whose elements will be added to `tensor`.
///
/// # Errors
/// - `ShapeMismatch`: Returned if the shapes of the two tensors do not match.
///
/// # Example
/// ```zig
/// const std = @import("std");
/// const Tensor = @import("tensor.zig").Tensor;
/// const add = @import("ops.zig").add;
///
/// var tensor1 = Tensor(f32, .{2, 2}, .{1.0, 2.0, 3.0, 4.0});
/// var tensor2 = Tensor(f32, .{2, 2}, .{5.0, 6.0, 7.0, 8.0});
///
/// try add(f32, &tensor1, tensor2);
/// // tensor1 now contains {6.0, 8.0, 10.0, 12.0}
/// ```
///
/// # Notes
/// - The function assumes that the `tensor` and `other` have the same shape and does not perform any broadcasting.
pub fn add(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        // std.log.err("tensor shape: {d}\n", .{tensor.shape});
        // std.log.err("other shape: {d}\n", .{other.shape});
        // std.log.err("Error during addition \n", .{});
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] += other.data[i];
    }
}

/// Subtracts the elements of one tensor from another tensor element-wise.
///
/// This function performs an element-wise subtraction of the `other` tensor from the `tensor`.
/// Both tensors must have the same shape for the operation to be valid.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensor`: A pointer to the tensor from which elements will be subtracted. The result will be stored in this tensor.
/// - `other`: The tensor whose elements will be subtracted from the `tensor`.
///
/// # Returns
/// - `void`: If the operation is successful.
/// - `error.ShapeMismatch`: If the shapes of the two tensors do not match.
///
/// # Errors
/// This function returns an error if the shapes of the two tensors do not match. The error returned is `error.ShapeMismatch`.
///
/// # Example
/// ```zig
/// const T = f32;
/// var tensor1 = Tensor(T){ .shape = [2]usize{2, 2}, .data = [4]T{1.0, 2.0, 3.0, 4.0} };
/// const tensor2 = Tensor(T){ .shape = [2]usize{2, 2}, .data = [4]T{0.5, 1.5, 2.5, 3.5} };
/// try subtract(T, &tensor1, tensor2);
/// // tensor1.data is now [0.5, 0.5, 0.5, 0.5]
/// ```
pub fn subtract(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        // std.log.err("tensor shape: {d}\n", .{tensor.shape});
        // std.log.err("other shape: {d}\n", .{other.shape});
        // std.log.err("Error during subtraction \n", .{});
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] -= other.data[i];
    }
}

/// Multiplies the elements of two tensors element-wise and stores the result in the first tensor.
///
/// This function performs an element-wise multiplication of the elements in `tensor` and `other`.
/// The result of the multiplication is stored in `tensor`.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensor`: A pointer to the first tensor, which will store the result of the multiplication.
/// - `other`: The second tensor to be multiplied with the first tensor.
///
/// # Returns
/// - `void`: Returns nothing on success.
/// - `error.ShapeMismatch`: If the shapes of the two tensors do not match.
///
/// # Errors
/// This function returns an error if the shapes of the two tensors do not match. The shapes must be equal
/// for the element-wise multiplication to be performed.
///
/// # Example
/// ```zig
/// const T = f32;
/// var tensor1 = Tensor(T, .{2, 2}, .{1.0, 2.0, 3.0, 4.0});
/// const tensor2 = Tensor(T, .{2, 2}, .{5.0, 6.0, 7.0, 8.0});
/// try multiply(T, &tensor1, tensor2);
/// // tensor1.data is now .{5.0, 12.0, 21.0, 32.0}
/// ```
pub fn multiply(comptime T: type, tensor: *Tensor(T), other: Tensor(T)) !void {
    if (!std.mem.eql(usize, tensor.shape, other.shape)) {
        // std.log.err("tensor shape: {d}\n", .{tensor.shape});
        // std.log.err("other shape: {d}\n", .{other.shape});
        // std.log.err("Error during multiplication \n", .{});
        return error.ShapeMismatch;
    }

    for (tensor.data, 0..) |_, i| {
        tensor.data[i] *= other.data[i];
    }
}

/// Adds a scalar value to each element in the tensor.
///
/// This function iterates over each element in the tensor and adds the given scalar value to it.
///
/// Parameters:
/// - `T`: The type of the elements in the tensor.
/// - `tensor`: A pointer to the tensor to which the scalar value will be added.
/// - `scalar`: The scalar value to add to each element in the tensor.
///
/// Example:
/// ```zig
/// const tensor = Tensor(f32, .{1.0, 2.0, 3.0});
/// scalarAdd(f32, &tensor, 1.0);
pub fn scalarAdd(comptime T: type, tensor: *Tensor(T), scalar: T) void {
    for (tensor.data, 0..) |_, i| {
        tensor.data[i] += scalar;
    }
}

/// Multiplies each element in the given tensor by a scalar value.
///
/// This function iterates over all elements in the tensor and multiplies each
/// element by the provided scalar value, modifying the tensor in place.
///
/// - Parameters:
///   - T: The type of the elements in the tensor. This is a compile-time parameter.
///   - tensor: A pointer to the tensor to be modified. The tensor's data will be
///     multiplied by the scalar value.
///   - scalar: The scalar value to multiply each element in the tensor by.
///
/// # Example
///
/// ```zig
/// const Tensor = @import("tensor.zig").Tensor;
/// const ops = @import("ops.zig");
///
/// var tensor = Tensor(f32, .{1.0, 2.0, 3.0});
/// ops.scalarMultiply(f32, &tensor, 2.0);
/// // tensor.data is now {2.0, 4.0, 6.0}
/// ```
pub fn scalarMultiply(comptime T: type, tensor: *Tensor(T), scalar: T) void {
    for (tensor.data, 0..) |_, i| {
        tensor.data[i] *= scalar;
    }
}

/// Performs broadcasted addition between two tensors.
/// The smaller tensor is broadcast to match the shape of the larger tensor along
/// matching dimensions from right to left.
/// For example: [seq_len, dim] + [dim] -> broadcasts [dim] across seq_len
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `a`: A pointer to the larger tensor which will be modified in place.
/// - `b`: The smaller tensor which will be broadcast and added to `a`.
///
/// # Returns
/// - `!void`: Returns an error if the shapes are not compatible for broadcasting.
///
/// # Errors
/// - `error.InvalidBroadcast`: If the shape of `b` is larger than the shape of `a`.
/// - `error.IncompatibleBroadcast`: If the shapes of `a` and `b` are not compatible for broadcasting.
///
/// # Example
/// ```zig
/// const T = f32;
/// var a = Tensor(T, .{2, 3}, .{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
/// const b = Tensor(T, .{3}, .{0.5, 1.5, 2.5});
/// try broadcast_add(T, &a, b);
/// // a.data is now {1.5, 3.5, 5.5, 4.5, 6.5, 8.5}
/// ```
///
/// This function first checks if the shapes of the tensors are compatible for broadcasting.
/// If they are, it performs the addition in place, modifying the larger tensor `a`.
/// It handles both the common case of adding a 1D tensor to each row of a 2D tensor,
/// as well as the general case for tensors of any shape.
pub fn broadcast_add(comptime T: type, a: *Tensor(T), b: Tensor(T)) !void {
    // Check that shapes can be broadcast
    if (b.shape.len > a.shape.len) {
        return error.InvalidBroadcast;
    }

    // Check that dimensions match from right to left
    for (0..b.shape.len) |i| {
        const a_dim = a.shape[a.shape.len - 1 - i];
        const b_dim = b.shape[b.shape.len - 1 - i];
        if (b_dim != a_dim and b_dim != 1) {
            return error.IncompatibleBroadcast;
        }
    }

    // For common case of [seq_len, dim] + [dim]
    if (a.shape.len == 2 and b.shape.len == 1 and b.shape[0] == a.shape[1]) {
        const seq_len = a.shape[0];
        const dim = a.shape[1];

        // Add bias to each row
        var i: usize = 0;
        while (i < seq_len) : (i += 1) {
            const row_start = i * dim;
            for (0..dim) |j| {
                a.data[row_start + j] += b.data[j];
            }
        }
        return;
    }

    // Handle general case
    const total_elements = blk: {
        var prod: usize = 1;
        for (a.shape) |dim| {
            prod *= dim;
        }
        break :blk prod;
    };

    // For each element in the output
    var i: usize = 0;
    while (i < total_elements) : (i += 1) {
        // Calculate indices for both tensors
        var a_coords = try a.allocator.alloc(usize, a.shape.len);
        defer a.allocator.free(a_coords);
        var temp = i;

        // Convert flat index to coordinates
        for (0..a.shape.len) |j| {
            const rev_j = a.shape.len - 1 - j;
            a_coords[rev_j] = temp % a.shape[rev_j];
            temp /= a.shape[rev_j];
        }

        // Calculate corresponding b index
        var b_idx: usize = 0;
        var b_stride: usize = 1;

        for (0..b.shape.len) |j| {
            const b_j = b.shape.len - 1 - j;
            const a_j = a.shape.len - 1 - j;
            const coord = a_coords[a_j] % b.shape[b_j];
            b_idx += coord * b_stride;
            b_stride *= b.shape[b_j];
        }

        // Add values
        a.data[i] += b.data[b_idx];
    }
}

/// Helper function for broadcasting multiplication.
///
/// This function performs element-wise multiplication of two tensors, `a` and `b`,
/// with broadcasting support. The result is stored back in tensor `a`.
///
/// - Parameters:
///   - T: The type of the elements in the tensors.
///   - a: A pointer to the tensor `a` which will be modified to store the result.
///   - b: The tensor `b` which will be broadcasted and multiplied with tensor `a`.
///
/// - Returns: This function returns an error if the copy operation for the temporary
///   result tensor fails.
///
/// - Note: The function assumes that the dimensions of tensor `b` are compatible
///   for broadcasting with tensor `a`. The broadcasting is performed by repeating
///   the elements of tensor `b` as necessary to match the size of tensor `a`.
///
/// Example:
/// ```zig
/// const T = f32;
/// var a = Tensor(T, .{1.0, 2.0, 3.0, 4.0});
/// const b = Tensor(T, .{2.0});
/// try broadcast_multiply(T, &a, b);
/// // a.data is now {2.0, 4.0, 6.0, 8.0}
/// ```
// Helper function for broadcasting multiplication
pub fn broadcast_multiply(comptime T: type, a: *Tensor(T), b: Tensor(T)) !void {
    // Create a temporary tensor for the result
    var result = try a.copy();
    defer result.deinit();

    // Perform broadcasted multiplication
    const total_elements = a.data.len;
    const b_elements = b.data.len;

    for (0..total_elements) |i| {
        // Calculate the broadcast index for b
        const b_idx = i % b_elements;
        result.data[i] = a.data[i] * b.data[b_idx];
    }

    // Copy result back to a
    @memcpy(a.data, result.data);
}

/// Helper function for broadcasting subtraction.
///
/// This function performs element-wise subtraction of two tensors, where the second tensor
/// is broadcasted to match the shape of the first tensor. The result is stored back in the
/// first tensor.
///
/// - Parameters:
///   - T: The type of the elements in the tensors.
///   - a: A pointer to the first tensor, which will be modified to store the result.
///   - b: The second tensor, which will be broadcasted and subtracted from the first tensor.
///
/// - Returns: An error if the operation fails.
///
/// - Errors:
///   - Any error that can be returned by the `copy` method of the tensor.
///
/// - Note: The function assumes that the dimensions of the tensors are compatible for broadcasting.
// Helper function for broadcasting subtraction
pub fn broadcast_subtract(comptime T: type, a: *Tensor(T), b: Tensor(T)) !void {
    var result = try a.copy();
    defer result.deinit();

    const total_elements = a.data.len;
    const b_elements = b.data.len;

    for (0..total_elements) |i| {
        const b_idx = i % b_elements;
        result.data[i] = a.data[i] - b.data[b_idx];
    }

    @memcpy(a.data, result.data);
}

const ThreadLocalData = struct {
    current_index: atomic.Value(usize) align(CACHE_LINE_SIZE),
    _padding: [CACHE_LINE_SIZE - @sizeOf(atomic.Value(usize))]u8 = undefined,
};

const ThreadContext = struct {
    a: []const f32,
    b: []const f32,
    c: []f32,
    M: usize,
    N: usize,
    K: usize,
    tiles_M: usize,
    tiles_N: usize,
    total_tiles: usize,
    shared_counter: *ThreadLocalData,
};

/// Multiplies two matrices and returns the resulting matrix.
///
/// This function takes two 2D arrays (matrices) as input and performs matrix multiplication.
/// The number of columns in the first matrix must be equal to the number of rows in the second matrix.
///
/// # Parameters
/// - `matrix1`: The first matrix (2D array) to be multiplied.
/// - `matrix2`: The second matrix (2D array) to be multiplied.
///
/// # Returns
/// A new matrix (2D array) which is the result of multiplying `matrix1` by `matrix2`.
///
/// # Example
/// ```zig
/// const matrix1 = [[1, 2, 3], [4, 5, 6]];
/// const matrix2 = [[7, 8], [9, 10], [11, 12]];
/// const result = matmul(matrix1, matrix2);
/// // result is [[58, 64], [139, 154]]
/// ```
///
/// # Notes
/// - The function assumes that the input tensors are properly initialized and deinitialized.
/// - The function uses SIMD for f32 data type to optimize matrix multiplication
/// - The function uses a simple triple-loop matrix multiplication for other data types.
/// - SIMD support for f64 and f16 will be added in future versions.
pub fn matmul(comptime T: type, a: Tensor(T), b: Tensor(T), allocator: Allocator) !Tensor(T) {
    if (a.shape.len != 2 or b.shape.len != 2) {
        return error.InvalidDimensions;
    }
    if (a.shape[1] != b.shape[0]) {
        return error.ShapeMismatch;
    }

    // Use optimized implementation for f32
    if (T == f32) {
        return optimizedMatmulF32(a, b, allocator);
    }

    // Simple implementation for other types
    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(T).init(allocator, &result_shape);
    errdefer result.deinit();

    // Initialize result to zero
    @memset(result.data, 0);

    // Simple triple-loop matrix multiplication
    for (0..M) |i| {
        for (0..N) |j| {
            var sum: T = 0;
            for (0..K) |k| {
                sum += a.data[i * K + k] * b.data[k * N + j];
            }
            result.data[i * N + j] = sum;
        }
    }

    return result;
}

fn optimizedMatmulF32(a: Tensor(f32), b: Tensor(f32), allocator: Allocator) !Tensor(f32) {
    const M = a.shape[0];
    const N = b.shape[1];
    const K = a.shape[1];

    const result_shape = [_]usize{ M, N };
    var result = try Tensor(f32).init(allocator, &result_shape);
    errdefer result.deinit();

    // Initialize result to zero
    @memset(result.data, 0);

    // Calculate tile grid dimensions
    const tiles_M = (M + Tile - 1) / Tile;
    const tiles_N = (N + Tile - 1) / Tile;
    const total_tiles = tiles_M * tiles_N;

    // Initialize shared atomic counter
    var shared_data = ThreadLocalData{ .current_index = atomic.Value(usize).init(0) };

    // Get number of CPU cores
    const num_threads = try std.Thread.getCpuCount();

    // Create thread pool
    var thread_pool = try std.ArrayList(std.Thread).initCapacity(allocator, num_threads);
    defer thread_pool.deinit();

    // Create thread context
    const context = ThreadContext{
        .a = a.data,
        .b = b.data,
        .c = result.data,
        .M = M,
        .N = N,
        .K = K,
        .tiles_M = tiles_M,
        .tiles_N = tiles_N,
        .total_tiles = total_tiles,
        .shared_counter = &shared_data,
    };

    // Spawn worker threads
    const WorkerFn = struct {
        fn worker(ctx: ThreadContext) void {
            workerThread(ctx);
        }
    };

    for (0..num_threads) |_| {
        try thread_pool.append(try std.Thread.spawn(.{}, WorkerFn.worker, .{context}));
    }

    // Wait for all threads to complete
    for (thread_pool.items) |thread| {
        thread.join();
    }

    return result;
}

fn workerThread(ctx: ThreadContext) void {
    var local_C: [Tile][Tile]f32 align(AVX2_ALIGNMENT) = undefined;

    while (true) {
        const start_idx = ctx.shared_counter.current_index.fetchAdd(CHUNK_SIZE, .seq_cst);
        if (start_idx >= ctx.total_tiles) break;

        const end_idx = @min(start_idx + CHUNK_SIZE, ctx.total_tiles);
        var idx = start_idx;

        while (idx < end_idx) : (idx += 1) {
            const i = idx / ctx.tiles_N;
            const j = idx % ctx.tiles_N;

            const i_start = i * Tile;
            const j_start = j * Tile;
            const i_end = @min(i_start + Tile, ctx.M);
            const j_end = @min(j_start + Tile, ctx.N);

            @memset(&local_C[0], 0);

            var k: usize = 0;
            while (k < ctx.K) : (k += Tile) {
                const k_end = @min(k + Tile, ctx.K);
                microKernelAVX2(ctx, &local_C, i_start, j_start, k, i_end, j_end, k_end);
            }

            for (i_start..i_end) |ii| {
                const row_offset = ii * ctx.N;
                const local_row = ii - i_start;

                var j_idx = j_start;
                while (j_idx + Vec <= j_end) : (j_idx += Vec) {
                    const vec_idx = j_idx - j_start;

                    const c_vec = Vec8f{
                        local_C[local_row][vec_idx],
                        local_C[local_row][vec_idx + 1],
                        local_C[local_row][vec_idx + 2],
                        local_C[local_row][vec_idx + 3],
                        local_C[local_row][vec_idx + 4],
                        local_C[local_row][vec_idx + 5],
                        local_C[local_row][vec_idx + 6],
                        local_C[local_row][vec_idx + 7],
                    };

                    const dest_vec = Vec8f{
                        ctx.c[row_offset + j_idx],
                        ctx.c[row_offset + j_idx + 1],
                        ctx.c[row_offset + j_idx + 2],
                        ctx.c[row_offset + j_idx + 3],
                        ctx.c[row_offset + j_idx + 4],
                        ctx.c[row_offset + j_idx + 5],
                        ctx.c[row_offset + j_idx + 6],
                        ctx.c[row_offset + j_idx + 7],
                    };

                    const result = dest_vec + c_vec;
                    for (0..8) |offset| {
                        ctx.c[row_offset + j_idx + offset] = result[offset];
                    }
                }

                while (j_idx < j_end) : (j_idx += 1) {
                    ctx.c[row_offset + j_idx] += local_C[local_row][j_idx - j_start];
                }
            }
        }
    }
}

fn microKernelAVX2(
    ctx: ThreadContext,
    local_C: *[Tile][Tile]f32,
    i_start: usize,
    j_start: usize,
    k_start: usize,
    i_end: usize,
    j_end: usize,
    k_end: usize,
) void {
    var A_local: [Tile][Tile]f32 align(32) = undefined;
    var B_local: [Tile][Tile]f32 align(32) = undefined;

    const k_size = k_end - k_start;
    const i_size = i_end - i_start;
    const j_size = j_end - j_start;

    for (0..i_size) |i| {
        const src_idx = (i_start + i) * ctx.K + k_start;
        for (0..k_size) |k| {
            A_local[i][k] = ctx.a[src_idx + k];
        }
    }

    for (0..k_size) |k| {
        const src_idx = (k_start + k) * ctx.N + j_start;
        for (0..j_size) |j| {
            B_local[k][j] = ctx.b[src_idx + j];
        }
    }

    var i: usize = 0;
    while (i + MICRO_KERNEL_SIZE <= i_size) : (i += MICRO_KERNEL_SIZE) {
        var j: usize = 0;
        while (j + MICRO_KERNEL_SIZE <= j_size) : (j += MICRO_KERNEL_SIZE) {
            var acc: [8][8]f32 align(32) = [_][8]f32{[_]f32{0} ** 8} ** 8;

            var k: usize = 0;
            while (k < k_size) : (k += 1) {
                const a_vec = Vec8f{
                    A_local[i][k],     A_local[i + 1][k],
                    A_local[i + 2][k], A_local[i + 3][k],
                    A_local[i + 4][k], A_local[i + 5][k],
                    A_local[i + 6][k], A_local[i + 7][k],
                };

                const b_vec = Vec8f{
                    B_local[k][j],     B_local[k][j + 1],
                    B_local[k][j + 2], B_local[k][j + 3],
                    B_local[k][j + 4], B_local[k][j + 5],
                    B_local[k][j + 6], B_local[k][j + 7],
                };

                inline for (0..8) |bi| {
                    const a_broadcast: Vec8f = @splat(a_vec[bi]);
                    const c_vec = Vec8f{
                        acc[bi][0], acc[bi][1], acc[bi][2], acc[bi][3],
                        acc[bi][4], acc[bi][5], acc[bi][6], acc[bi][7],
                    };
                    const prod = @mulAdd(Vec8f, a_broadcast, b_vec, c_vec);
                    inline for (0..8) |bj| {
                        acc[bi][bj] = prod[bj];
                    }
                }
            }

            for (0..8) |bi| {
                for (0..8) |bj| {
                    local_C[i + bi][j + bj] += acc[bi][bj];
                }
            }
        }

        while (j < j_size) : (j += 1) {
            for (0..8) |bi| {
                var sum: f32 = 0;
                for (0..k_size) |k| {
                    sum = @mulAdd(f32, A_local[i + bi][k], B_local[k][j], sum);
                }
                local_C[i + bi][j] += sum;
            }
        }
    }

    while (i < i_size) : (i += 1) {
        for (0..j_size) |j| {
            var sum: f32 = 0;
            for (0..k_size) |k| {
                sum = @mulAdd(f32, A_local[i][k], B_local[k][j], sum);
            }
            local_C[i][j] += sum;
        }
    }
}
/// Computes the outer product of two 1-dimensional tensors.
///
/// The outer product of two vectors `tensor` and `other` is a matrix where each element
/// `(i, j)` is the product of `tensor[i]` and `other[j]`.
///
/// # Parameters
/// - `T`: The type of the elements in the tensors.
/// - `tensor`: The first input tensor, which must be 1-dimensional.
/// - `other`: The second input tensor, which must be 1-dimensional.
///
/// # Returns
/// - A new tensor representing the outer product of `tensor` and `other`.
///
/// # Errors
/// - `error.InvalidDimensions`: If either `tensor` or `other` is not 1-dimensional.
///
/// # Example
/// ```zig
/// const T = f32;
/// const tensor1 = try Tensor(T).init(allocator, &[_]T{1.0, 2.0});
/// const tensor2 = try Tensor(T).init(allocator, &[_]T{3.0, 4.0});
/// const result = try outer(T, tensor1, tensor2);
/// defer {
///     tensor1.deinit();
///     tensor2.deinit();
///     result.deinit();
/// }
/// // result is a 2x2 tensor with values:
/// // [[3.0, 4.0],
/// //  [6.0, 8.0]]
/// ```
///
/// # Notes
/// - The function assumes that the input tensors are properly initialized and deinitialized.
pub fn outer(comptime T: type, tensor: Tensor(T), other: Tensor(T)) !Tensor(T) {
    if (tensor.shape.len != 1 or other.shape.len != 1) {
        return error.InvalidDimensions;
    }

    const m = tensor.shape[0];
    const n = other.shape[0];

    var result = try Tensor(@TypeOf(tensor.data[0])).init(tensor.allocator, &[_]usize{ m, n });
    errdefer result.deinit();

    for (0..m) |i| {
        for (0..n) |j| {
            result.data[i * n + j] = tensor.data[i] * other.data[j];
        }
    }

    return result;
}

// ------------------------ Machine Learning --------------------------------------

/// Applies Layer Normalization to the input tensor.
///
/// Layer Normalization normalizes the input tensor along the last dimension
/// and scales it using the provided weight and bias tensors. This function
/// also includes stability checks to ensure numerical stability during the
/// normalization process.
///
/// # Parameters
///
/// - `T`: The data type of the tensor elements (e.g., `f32`, `f64`).
/// - `input`: The input tensor to be normalized.
/// - `weight`: The weight tensor used for scaling the normalized values.
/// - `bias`: The bias tensor added to the scaled values.
/// - `eps`: A small value added to the variance for numerical stability.
///
/// # Returns
///
/// - `Tensor(T)`: The normalized tensor with the same shape as the input tensor.
///
/// # Errors
///
/// - `error.InvalidEpsilon`: If `eps` is less than or equal to zero.
/// - `error.InvalidShape`: If the input tensor has less than one dimension.
/// - `error.InvalidWeightShape`: If the weight tensor shape is invalid.
/// - `error.InvalidBiasShape`: If the bias tensor shape is invalid.
/// - `error.NegativeVariance`: If the computed variance is negative.
/// - `error.ZeroStandardDeviation`: If the computed standard deviation is zero.
/// - `error.ComputedNaN`: If the computed value is NaN.
/// - `error.ComputedInfinity`: If the computed value is infinity.
///
/// # Stability Checks
///
/// This function performs several stability checks:
/// - Checks the stability of the input, weight, and bias tensors.
/// - Ensures the computed variance is not negative.
/// - Ensures the computed standard deviation is not zero.
/// - Checks for NaN and infinity in the computed values.
/// - Checks the stability of the output tensor before returning it.
///
/// # Example
///
/// ```zig
/// const input = Tensor(f32, .{2, 3}, .{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
/// const weight = Tensor(f32, .{3}, .{0.1, 0.2, 0.3});
/// const bias = Tensor(f32, .{3}, .{0.0, 0.0, 0.0});
/// const eps = 1e-5;
/// const result = try layerNorm(f32, input, weight, bias, eps);
/// ```
pub fn layerNorm(comptime T: type, input: Tensor(T), weight: Tensor(T), bias: Tensor(T), eps: T) !Tensor(T) {
    // Check input stability
    try checkStability(T, input);
    try checkStability(T, weight);
    try checkStability(T, bias);

    // Validate epsilon
    if (eps <= 0) {
        return error.InvalidEpsilon;
    }

    // Input validation
    if (input.shape.len < 1) {
        return error.InvalidShape;
    }
    const last_dim = input.shape[input.shape.len - 1];

    if (weight.shape.len != 1 or weight.shape[0] != last_dim) {
        return error.InvalidWeightShape;
    }
    if (bias.shape.len != 1 or bias.shape[0] != last_dim) {
        return error.InvalidBiasShape;
    }

    // Calculate size of dimensions before the last dimension
    var leading_dims: usize = 1;
    for (input.shape[0 .. input.shape.len - 1]) |dim| {
        leading_dims *= dim;
    }

    // Create output tensor with same shape as input
    var output = try input.copy();
    errdefer output.deinit();

    // Compute mean and variance for each feature vector
    var i: usize = 0;
    while (i < leading_dims) : (i += 1) {
        const start_idx = i * last_dim;
        const end_idx = start_idx + last_dim;

        // Calculate mean
        var mean: T = 0;
        for (start_idx..end_idx) |j| {
            mean += input.data[j];
        }
        mean /= @as(T, @floatFromInt(last_dim));

        // Calculate variance
        var variance: T = 0;
        for (start_idx..end_idx) |j| {
            const diff = input.data[j] - mean;
            variance += diff * diff;
        }
        variance /= @as(T, @floatFromInt(last_dim));

        // Check for numerical stability in variance
        if (variance < -eps) {
            return error.NegativeVariance;
        }

        // Add stability checks for the normalization process
        const std_dev = @sqrt(variance + eps);
        if (std_dev == 0) {
            return error.ZeroStandardDeviation;
        }

        // Normalize and apply scale and bias
        for (start_idx..end_idx) |j| {
            const feature_idx = j - start_idx;
            const normalized = (input.data[j] - mean) / std_dev;
            const scaled = normalized * weight.data[feature_idx];
            const final_value = scaled + bias.data[feature_idx];

            // Check for stability of computed value
            if (std.math.isNan(final_value)) {
                return error.ComputedNaN;
            }
            if (std.math.isInf(final_value)) {
                return error.ComputedInfinity;
            }

            output.data[j] = final_value;
        }
    }

    // Final stability check on output
    try checkStability(T, output);
    return output;
}

const LayerNormError = error{
    InvalidShape,
    InvalidWeightShape,
    InvalidBiasShape,
    InvalidEpsilon,
    NegativeVariance,
    ZeroStandardDeviation,
    ComputedNaN,
    ComputedInfinity,
} || StabilityError;

/// All possible errors from tensor operations and freqs computation
const FreqsError = error{
    // Tensor initialization errors
    TensorTooLarge,
    IncompatibleShape,

    // Input validation errors
    DimensionTooSmall,
    DimensionNotEven,
    EndTooSmall,
    ThetaTooSmall,
    InvalidShape,

    // Computation errors
    ComputationOverflow,
    NumericalInstability,

    // Memory errors
    OutOfMemory,
};

// Softmax operation along specified dimension
fn softmax(comptime T: type, tensor: *Tensor(T), dim: usize) !void {
    const dim_size = tensor.shape[dim];

    // Calculate stride for the specified dimension
    var stride: usize = 1;
    for (dim + 1..tensor.shape.len) |i| {
        stride *= tensor.shape[i];
    }

    // Calculate number of vectors to process
    var num_vectors: usize = 1;
    for (0..dim) |i| {
        num_vectors *= tensor.shape[i];
    }

    // Process each vector
    for (0..num_vectors) |i| {
        const base_idx = i * dim_size * stride;

        // Find max for numerical stability
        var max: T = -std.math.inf(T);
        for (0..dim_size) |j| {
            const val = tensor.data[base_idx + j * stride];
            if (val > max) max = val;
        }

        // Calculate exp and sum
        var sum: T = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * stride;
            tensor.data[idx] = @exp(tensor.data[idx] - max);
            sum += tensor.data[idx];
        }

        // Normalize
        if (sum > 0) {
            for (0..dim_size) |j| {
                const idx = base_idx + j * stride;
                tensor.data[idx] /= sum;
            }
        }
    }
}

/// Applies the Gaussian Error Linear Unit (GELU) activation function to the input.
///
/// The GELU activation function is defined as:
/// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x + 0.044715 * x^3)))
///
/// This function is used in neural networks to introduce non-linearity.
/// It is known for its smooth and differentiable properties, which can help
/// with the training of deep learning models.
///
/// Parameters:
/// - `x`: The input value to the GELU function.
///
/// Returns:
/// - The output value after applying the GELU function to the input.
///
/// Note:
/// - If the array is empty, the behavior of this function is undefined.
pub fn gelu(comptime T: type, tensor: *Tensor(T)) !void {
    if (@typeInfo(T) != .Float) {
        @compileError("GELU operation requires floating-point tensor");
    }

    // Constants for GELU approximation
    const sqrt_2_div_pi: T = @sqrt(2.0 / std.math.pi);
    const alpha: T = 0.044715;

    for (tensor.data) |*x| {
        const val = x.*;
        const x_cubed = val * val * val;
        const inner = sqrt_2_div_pi * (val + alpha * x_cubed);
        x.* = 0.5 * val * (1 + std.math.tanh(inner));
    }
}

/// Returns the index of the maximum value in the given array.
///
/// This function iterates through the provided array and compares each element
/// to find the maximum value. It then returns the index of this maximum value.
///
/// Parameters:
/// - `array`: The array of values to search through.
///
/// Returns:
/// - `usize`: The index of the maximum value in the array.
///
/// Note:
/// - If the array is empty, the behavior of this function is undefined.
pub fn argmax(comptime T: type, input: Tensor(T)) !usize {
    if (input.data.len == 0 or input.shape.len == 0) {
        return error.EmptyTensor;
    }

    // Get the last dimension size for vocab
    const vocab_size = input.shape[input.shape.len - 1];

    // For logits, we only care about the last value since we're doing token generation
    const start_idx = input.data.len - vocab_size;

    var max_value: T = input.data[start_idx];
    var max_index: usize = 0;

    // Find the maximum value and its index
    for (start_idx..input.data.len) |i| {
        if (input.data[i] > max_value) {
            max_value = input.data[i];
            max_index = i - start_idx;
        }
    }

    return max_index;
}
