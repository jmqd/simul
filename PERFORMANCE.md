# Performance notes

- `codegen-units = 1` was found to regress performance
- `opt-level = 3` was found to regress performance
- Compare builds with `tools/instruction_counts.py`: retired instructions are
  stable to about 0.3% on a busy host, while cycles vary by several percent.
- Splitting the agent slice on every visit regressed large, mostly sleeping
  simulations by 11%; only simulations that scan names split it.
- An out-of-line `AgentContext::agent_id` cost simulations with more than eight
  agents about 2%; it stays `#[inline]`.
- Checking history capacity on every push regressed long simulations by 1-4%;
  small simulations reserve histories at construction instead.
- Sharing one cursor loop between dense and sparse ticks cost dense simulations
  7-10%; dense ticks keep a plain range loop and sparse ticks are outlined.
- Separate dense and scheduled outer loops moved the dense tick and delivery out
  of line (+8% for small simulations); one loop serves both.
- The sparse scheduler's per-tick checks cost two-agent simulations about 2% of
  retired instructions with no measurable change in cycles.
- Padding `Message` by 8 bytes cost message-heavy simulations 1.5-5% and by 16
  bytes 2-6%, so a boxed interrupt beside a fat `Arc<[u8]>` payload was
  rejected; a const assertion keeps `Message` at 48 bytes.
- The two 10k-tick pair workloads have bimodal minima that differ by 2-4%
  between runs of one binary; compare them over 9 or more rounds.
