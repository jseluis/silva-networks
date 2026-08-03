var MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    processHtmlClass: "md-typeset|arithmatex|jp-RenderedHTMLCommon|jp-RenderedMarkdown|jp-MarkdownOutput"
  }
};

(function () {
  const loaderScript = document.currentScript;
  const loaderBase = loaderScript && loaderScript.src
    ? loaderScript.src.slice(0, loaderScript.src.lastIndexOf("/") + 1)
    : "";
  const mathJaxUrls = [
    `${loaderBase}vendor/mathjax/tex-mml-chtml.js`,
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"
  ];

  const typesetMath = () => {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
      MathJax.typesetPromise();
    }
  };

  const loadMathJax = (index) => {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
      typesetMath();
      return;
    }
    if (document.querySelector("script[data-silva-mathjax-loader]")) {
      return;
    }
    if (index >= mathJaxUrls.length) {
      console.warn("SILVA docs could not load MathJax.");
      return;
    }

    const script = document.createElement("script");
    script.src = mathJaxUrls[index];
    script.async = true;
    script.dataset.silvaMathjaxLoader = "true";
    script.addEventListener("load", typesetMath);
    script.addEventListener("error", () => {
      script.remove();
      loadMathJax(index + 1);
    });
    document.head.appendChild(script);
  };

  const scheduleTypeset = () => {
    if (typeof MathJax !== "undefined" && MathJax.typesetPromise) {
      typesetMath();
      return;
    }
    if (document.readyState === "complete") {
      loadMathJax(1);
    }
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(scheduleTypeset);
  }

  window.addEventListener("load", scheduleTypeset);
})();
