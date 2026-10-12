# Build-only dependency patch

Source: crates.io `mlx-sys 0.6.0`, upstream commit
`9c2fd72e172babb1e3719327a723717d98ccbe97` in oxiglade/mlx-rs.
The bundled mlx-c targets MLX 0.32.2. MIT/Apache-2.0 licenses are included;
mlx-c retains its upstream license. C/C++ binding code is unchanged.

The only compiled source change is the `MLX_PREBUILT_ROOT` branch in build.rs.
Unused upstream documentation and Python binding generators are omitted. It uses
mlx-c's existing system-MLX CMake support, links Apple's prebuilt libmlx instead
of rebuilding Metal kernels, and stages runtime files for Cargo test execution.
The repository config selects a build-local prefix populated by
`scripts/fetch_mlx_runtime.sh`. This introduces no Python runtime or model code.

Remove this patch when upstream mlx-sys supports an external native MLX prefix.
