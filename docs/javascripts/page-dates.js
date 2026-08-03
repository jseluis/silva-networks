(() => {
  const defaultDate = "August 3, 2026";
  const pageNotes = {
    "": "Blue brand refresh, cinematic SILVA method visual, responsive equation band, navigation grids, derivation map, citation status, and package overview.",
    "index.html": "Blue brand refresh, cinematic SILVA method visual, responsive equation band, navigation grids, derivation map, citation status, and package overview.",
    "start-here/": "Learning path and package orientation reviewed.",
    "run-everything/": "Run path added for install, examples, notebooks, data, tests, release audit, and local docs.",
    "equation-and-pdf-audit/": "Equation inventory, companion-asset policy, and source-to-doc traceability added.",
    "release-readiness/": "Release checklist, citation checks, notebook smoke, and packaging validation path added.",
    "book/": "Book and solutions manual roadmap added.",
    "learn/mathematical-foundations/": "Equilibrium, damping, adjoint, and solver derivations expanded.",
    "learn/derivation-workbook/": "Step-by-step derivation workbook added from scalar fixed points to SILVA cases.",
    "learn/implementation-derivations/": "Equation-to-source traceability manual added.",
    "learn/case-atlas/": "Case families, tensor contracts, and diagnostics expanded.",
    "learn/solver-derivation-lab/": "Picard, Anderson, Broyden, GMRES, and adjoint derivations added.",
    "learn/implicit-bridge/": "Package-native implicit-layer, DEQ, ODE, optimization, MDEQ, and Jacobian-regularization bridge added.",
    "learn/implicit-backward-guide/": "Current autograd behavior and implicit-adjoint diagnostics documented.",
    "learn/interactive-diagnostics-lab/": "Interactive diagnostic controls for damping, solvers, Jacobians, and graph density added.",
    "learn/method-adaptation-atlas/": "Source-to-SILVA adaptation atlas added with derivations, citations, scope notes, and smoke notebook links.",
    "learn/cortex-hierarchy/": "Cortex hierarchy derivation, flexible architecture controls, alphas, and image preset documented.",
    "get-started/derivation-to-code/": "Linked to the implementation derivations manual.",
    "research-citation-audit/": "Method-to-paper citation audit added across package solvers, layers, presets, datasets, and bridge modules.",
    "experiments/benchmark-cards/": "Public smoke metrics summarized from checked JSON outputs.",
    "api/reference/": "API map aligned with current package modules.",
    "api/public-api/": "Top-level package export reference added from `silva_networks.__all__`.",
    "api/implicit/": "Implicit bridge public API documented from `silva_networks.implicit`.",
    "api/deq-engine/": "DEQ engine API documented with multi-state packing equations and TorchDEQ lineage.",
    "api/flow/": "Optical-flow API documented with RAFT/DEQ-Flow equations and citations.",
    "api/cases/": "Full generalized cases module reference added for outputs, helper blocks, and case classes.",
    "api/families/": "Family factory API documented and wired into the full module reference audit.",
    "api/optimization/": "Optimization API linked to the method adaptation atlas and constrained/CVXPYlayers citation scope.",
    "api/coverage/": "Coverage registry added for implementation docs, notebooks, examples, and smoke tests.",
    "api/presets/": "Preset API aligned with `silva_networks.presets`.",
    "examples/deq-engine-bridge/": "DEQ engine bridge example documented with equations and diagnostics.",
    "examples/optical-flow-silva/": "Optical-flow SILVA example documented with RAFT/DEQ-Flow equations.",
    "examples/cortex-hierarchy/": "Cortex hierarchy example documented with linked equilibrium equations.",
    "examples/citation-aware-reporting/": "Citation-aware methods-reporting example added.",
    "notebooks/": "Notebook index reviewed against package and implicit bridge tracks.",
    "package-notebooks/07_research_citation_audit/": "Citation checklist notebook added for solver/operator reporting.",
    "package-notebooks/08_equation_to_code_walkthrough/": "Executable equation-to-code walkthrough added.",
    "package-notebooks/09_family_selector_and_projected_qp/": "Family selector and projected-QP smoke notebook added.",
    "package-notebooks/10_training_helpers_smoke/": "Training-helper smoke notebook added.",
    "package-notebooks/11_cortex_hierarchy/": "Cortex hierarchy notebook added with linked points, alphas, residual plots, and image preset smoke.",
    "package-notebooks/14_point_architecture_catalog/": "Ten point architectures added with tensor contracts, residual checks, gradients, and composition examples.",
    "implicit-bridge-notebooks/01_introduction_fixed_points/": "Fixed-point tutorial converted to package-native solvers.",
    "implicit-bridge-notebooks/02_implicit_autodiff/": "Implicit autodiff tutorial connected to package Jacobian helpers.",
    "implicit-bridge-notebooks/03_neural_odes_as_implicit_layers/": "Neural ODE bridge documented with the package Euler block.",
    "implicit-bridge-notebooks/04_deq_and_silva/": "DEQ baseline and SILVA graph model track added.",
    "implicit-bridge-notebooks/05_differentiable_optimization/": "Quadratic optimization layer tutorial added.",
    "implicit-bridge-notebooks/06_mdeq_jacobian_regularization/": "MDEQ and Jacobian-regularization notebook added.",
    "implicit-bridge-notebooks/07_silva_deq_engine_torchdeq_bridge/": "TorchDEQ-style SILVA DEQ engine notebook added.",
    "implicit-bridge-notebooks/08_silva_optical_flow_deq_raft_bridge/": "RAFT/DEQ-Flow-style SILVA optical-flow notebook added.",
    "implicit-bridge-notebooks/09_method_adaptation_atlas/": "Method adaptation atlas notebook added for source-to-package smoke checks.",
    "documentation-log/": "Documentation modification log added.",
    "paper/references/": "Research references, arXiv metadata, BibTeX, and package assets expanded.",
  };

  const normalizePath = () => {
    let path = window.location.pathname.replace(/\/silva-networks\/?/, "");
    path = path.replace(/^\/+/, "");
    if (path === "") return "";
    if (!path.endsWith("/") && !path.endsWith(".html")) path += "/";
    return path;
  };

  const insertDate = () => {
    const content = document.querySelector(".md-content__inner");
    if (!content || content.querySelector(".silva-page-date")) return;

    const path = normalizePath();
    const note = pageNotes[path] || "Documentation reviewed for the current package release.";
    const date = document.createElement("div");
    date.className = "silva-page-date";
    date.innerHTML = [
      '<span class="silva-page-date__label">Updated</span>',
      `<time datetime="2026-08-03">${defaultDate}</time>`,
      `<span class="silva-page-date__note">${note}</span>`,
    ].join(" ");

    const hero = content.querySelector(".silva-hero");
    if (hero) {
      hero.insertAdjacentElement("afterend", date);
      return;
    }

    const heading = content.querySelector("h1");
    if (heading) {
      heading.insertAdjacentElement("afterend", date);
      return;
    }

    content.prepend(date);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", insertDate, { once: true });
  } else {
    insertDate();
  }
})();
