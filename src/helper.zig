const std = @import("std");

pub fn normalizeDim(dim: isize, n_dims: usize) !usize {
    if (dim >= 0) {
        const udim = @as(usize, @intCast(dim));
        if (udim >= n_dims) return error.InvalidDimension;
        return udim;
    } else {
        const adjusted = @as(isize, @intCast(n_dims)) + dim;
        if (adjusted < 0) return error.InvalidDimension;
        return @as(usize, @intCast(adjusted));
    }
}

pub fn product(arr: []usize) usize {
    var result: usize = 1;
    for (arr) |value| {
        result *= value;
    }
    return result;
}
