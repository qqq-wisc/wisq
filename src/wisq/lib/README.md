# Dependency versions
- Synthetiq: bbe3c1299a97295f5af38eec647f6bbe9fdd9234
- GUOQ is no longer vendored here: it is the `guoq` pip dependency, built from the
  `rust-port` branch of https://github.com/qqq-wisc/guoq. `rules/` remains vendored
  and is passed to the binary via `--rules-dir`/`-sr`.
