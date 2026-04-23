## Summary

<!-- What changed and why? -->

## OpenSpec

- Change name / proposal:
- Related spec(s):

## Scope

- [ ] Bug fix
- [ ] Documentation / governance cleanup
- [ ] Build / CI / workflow change
- [ ] API or behavior change
- [ ] Performance-sensitive change

## Validation

```bash
cmake --preset cpu-smoke
cmake --build --preset cpu-smoke --parallel 2
python3 -m build --wheel
```

- [ ] CPU smoke validation completed
- [ ] Python wheel build completed
- [ ] CUDA validation completed locally when relevant

## Review Notes

- [ ] README / Pages / GitHub metadata remain aligned
- [ ] Root docs were preferred over duplicate docs
- [ ] `/review` was used for structural or workflow changes

## Additional context

<!-- Any migration notes, screenshots, benchmark data, or follow-ups -->
