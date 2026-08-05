# Scale CLI

The `silva-scale` command lists all canonical families, audits guide coverage,
and prints one family's data contract, references, benchmark route, scale
controls, extension points, and scalable constructor defaults.

## Commands

```bash
silva-scale --list
silva-scale silva_fno_deq --tier workstation
silva-scale pideq --tier full --json
silva-scale --audit
```

The command reports configurations; it does not download data or launch a
benchmark. Use its canonical name and defaults with `build_scaled_silva`, then
provide the task-specific dimensions, modules, schedules, and constraints.

::: silva_networks.scale_cli

## Where to Go Next

| Question | Page |
| --- | --- |
| What does each reported scale field mean mathematically? | [Full-Scale SILVA](../learn/full-scale-silva.md) |
| Which Python objects expose the same information? | [Scaling API](scaling.md) |
| Where is the family selection taxonomy? | [Selecting Model Families](../learn/selecting-model-families.md) |
| Can I run the scale checks in a notebook? | [Full-Scale Family Notebook](../package-notebooks/26_full_scale_silva.ipynb) |
