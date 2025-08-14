# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build
```bash
cargo build
```

### Run
```bash
cargo run
```

### Test
```bash
cargo test
# Run a specific test
cargo test test_name
```

### Lint and Format
```bash
cargo clippy         # Linting
cargo fmt            # Format code
cargo fmt --check    # Check formatting without applying
```

## Architecture

This is a Rust robotics library implementing core mathematical primitives for robotics applications.

### Core Modules

- **`linalg`** (src/linalg.rs): Custom linear algebra implementation using const generics
  - `Vector<N>`: N-dimensional vectors with operations like dot product, norm, cross product (3D)
  - `Matrix<R, C>`: R×C matrices with full algebra operations, determinant, inverse, and linear system solving
  - Uses f64 throughout, no external dependencies
  - Implements LU decomposition for determinants and Gauss-Jordan for matrix inversion

- **`motion`** (src/motion.rs): Robotics motion primitives
  - Currently implements rotation matrices and transformation functions
  - Built on top of the linalg module

### Design Philosophy

The codebase prioritizes clarity and safety over performance optimization. It uses Rust's const generics for compile-time dimension checking and implements operations idiomatically with operator overloading.