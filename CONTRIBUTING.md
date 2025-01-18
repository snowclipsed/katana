# Contributing to Katana

Thank you for your interest in contributing to Katana! We value all contributions, whether they're bug fixes, new features, documentation improvements, or test additions.

## Getting Started

1. Fork the repository
2. Create a new branch for your feature or bugfix: `git checkout -b feature-name`
3. Make your changes
4. Run tests and ensure they pass: `zig build test`
5. Submit a Pull Request

## Development Setup

1. Clone your fork:
   ```bash
   git clone https://github.com/snowclipsed/katana.git
   cd katana
   ```

2. Set up development environment:
   ```bash
   zig build test
   ```

3. Create a branch for your work:
   ```bash
   git checkout -b your-feature-name
   ```

## Code Style Guidelines

- Follow the [Zig Style Guide](https://ziglang.org/documentation/master/#Style-Guide)
- Write descriptive variable and function names
- Add comments for complex algorithms or non-obvious code
- Keep functions focused and concise
- Use meaningful error messages

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Include both unit tests and integration tests where applicable
- Test edge cases and error conditions

## Pull Request Process

1. Update documentation for any modified or new functionality
2. Add or update tests as needed
3. Ensure your code follows our style conventions
4. Submit PR with clear description of changes
5. Address any review feedback
6. Once approved, your PR will be merged

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- A clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Zig version and OS information
- Any relevant code snippets or error messages

### Feature Requests

We welcome feature requests! Please provide:

- A clear description of the feature
- Use cases and benefits
- Potential implementation approaches
- Any relevant examples or references

## Documentation

When contributing, please:

- Update documentation for modified functions
- Add examples for new features
- Keep documentation clear and concise
- Include docstrings for public APIs
- Update README if needed

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on the technical merits of contributions
- Keep discussions professional and on-topic

## License

By contributing to Katana, you agree that your contributions will be licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full license text.

All file headers in the source code should include the following license header:

```
// Copyright 2024 Katana Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```