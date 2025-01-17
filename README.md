# Tensor-Zig

A tensor computation library for Zig, providing efficient matrix operations and tensor manipulations.

## Installation

### Prerequisites

> **Important**: This library currently only works with Zig version 0.13.0. Support for other versions is planned for future releases as Zig itself progresses as a language.

Required:
- Zig compiler version 0.13.0
- Git (optional, for cloning the repository)

If you're using a different Zig version, you may encounter compatibility issues.

### Adding the Library to Your Project

There are two ways to add Tensor-Zig to your project:

#### Method 1: Using `zig fetch`

1. Ensure you have a `build.zig.zon` file in your project root.
2. Run the following command:
   ```bash
   zig fetch --save https://github.com/snowclipsed/tensor-zig/archive/refs/tags/v0.1.0.tar.gz
   ```

#### Method 2: Using Git
1. Clone the repository
2. Run:
   ```bash
   git fetch --save <link/to/tensor-zig>
   ```

### Configuring Your Build

Add Tensor-Zig as a dependency in your `build.zig` file. Here's a complete example:

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Load the tensor dependency
    const package = b.dependency("tensor", .{
        .target = target,
        .optimize = optimize,
    });

    const module = package.module("tensor");

    const exe = b.addExecutable(.{
        .name = "test-tensor",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);
    exe.root_module.addImport("tensor", module);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
```

#### Build File Explanation

- The build file sets up a standard Zig executable project
- It loads the tensor package using `b.dependency("tensor", ...)`
- The package is added as a module that can be imported in your code
- The executable is configured with the tensor module as a dependency
- A run step is added for convenience

## Usage

Here's a basic example showing how to use Tensor-Zig:

```zig
const std = @import("std");
const tensor = @import("tensor");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create tensors
    var a = try tensor.Tensor(f32).init(allocator, &[_]usize{ 2, 3 });
    defer a.deinit();
    var b = try tensor.Tensor(f32).init(allocator, &[_]usize{ 3, 2 });
    defer b.deinit();

    // Perform matrix multiplication
    var c = try tensor.matmul(f32, a, b, allocator);
    defer c.deinit();

    std.debug.print("Result: {}\n", .{c});
}
```

This example demonstrates:
- Importing the tensor library
- Creating tensors with specific dimensions
- Performing matrix multiplication
- Proper memory management using defer statements

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2024 Tensor-Zig Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## Contributing

We welcome contributions to Tensor-Zig! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit changes, coding standards, testing requirements, and more.