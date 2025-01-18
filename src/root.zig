//! Tensor-Zig is a tensor computation library providing efficient tensor manipulations
//! and matrix operations.
const tensor = @import("tensor.zig");
pub const ops = @import("ops.zig");

pub const Tensor = tensor.Tensor;
pub const StabilityError = tensor.StabilityError;
pub const Slice = tensor.Slice;
pub const PrintOptions = tensor.PrintOptions;
