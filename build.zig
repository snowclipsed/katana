const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseSafe,
    });

    // Create the module
    _ = b.addModule("tensor", .{
        .root_source_file = .{ .cwd_relative = "src/tensor.zig" },
        .imports = &.{},
    });

    // Create library artifact
    const lib = b.addStaticLibrary(.{
        .name = "tensor",
        .root_source_file = .{ .cwd_relative = "src/tensor.zig" },
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(lib);

    // Add tests
    const main_tests = b.addTest(.{
        .root_source_file = .{ .cwd_relative = "src/tests.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_main_tests = b.addRunArtifact(main_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);
}
