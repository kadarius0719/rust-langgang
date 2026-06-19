# Local CI gate — mirrors the `test` job in .github/workflows/ci.yml so that
# formatting/lint/test failures are caught before they reach a PR.
#
#   make ci      run the full gate (what CI runs)
#
# To run it automatically before every push, enable the committed hook once:
#   git config core.hooksPath .githooks
export RUSTFLAGS := -D warnings

.PHONY: ci fmt clippy test

ci: fmt clippy test

fmt:
	cargo fmt --all --check

clippy:
	cargo clippy --all-targets -- -D warnings
	cargo clippy --all-targets --features full -- -D warnings

test:
	cargo test
	cargo test --features full
