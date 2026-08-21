.. container:: solver-flow-figure-block

   .. raw:: html
      :file: solver_flow_fig1.html

**Figure 1.** Timestep driver for step index :math:`n`.
Preamble → :math:`\Delta t^\ast`-retry loop (phases (A)–(F)) → post-accept update.
Badge key: ``PASS`` / ``SKIP`` / ``RETRY`` (green) — attempt continues.
``WARN`` / ``SOFT`` (amber) — continues with a diagnostic flag.
``REJECT`` (red) — :math:`\mathbf{u}^n` restored, :math:`\Delta t^\ast` adjusted, retry.
Phase (D) never rejects. See Figure 8 for full per-gate detail.
