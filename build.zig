const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseSafe,
    });

    // Create library artifact with ReleaseSafe
    const lib = b.addStaticLibrary(.{
        .name = "tensor",
        .root_source_file = .{ .cwd_relative = "src/tensor.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Install the library
    b.installArtifact(lib);

    // Add tests with ReleaseSafe
    const main_tests = b.addTest(.{
        .root_source_file = .{ .cwd_relative = "src/ops_tests.zig" },
        .target = target,
        .optimize = optimize,
    });

    const run_main_tests = b.addRunArtifact(main_tests);
    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&run_main_tests.step);
}
