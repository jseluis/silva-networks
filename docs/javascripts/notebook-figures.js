(() => {
  const missingDescription = "No description has been provided for this image";
  const cleanText = (value) =>
    value.replace(/\u00b6/g, "").replace(/\s+/g, " ").trim();

  const nearestContext = (image) => {
    const headings = Array.from(
      document.querySelectorAll(".jupyter-wrapper h1, .jupyter-wrapper h2, .jupyter-wrapper h3, .jupyter-wrapper h4")
    );
    const preceding = headings.filter(
      (heading) => heading.compareDocumentPosition(image) & Node.DOCUMENT_POSITION_FOLLOWING
    );
    if (preceding.length) return cleanText(preceding[preceding.length - 1].textContent);

    const pageHeading = document.querySelector(".md-content__inner h1");
    return pageHeading && cleanText(pageHeading.textContent)
      ? cleanText(pageHeading.textContent)
      : "this SILVA notebook section";
  };

  const describeNotebookFigures = () => {
    const selector = `.jupyter-wrapper img[alt="${missingDescription}"]`;
    document.querySelectorAll(selector).forEach((image) => {
      image.alt = `Executed notebook figure for ${nearestContext(image)}.`;
    });
  };

  if (typeof document$ !== "undefined") {
    document$.subscribe(describeNotebookFigures);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", describeNotebookFigures, {
      once: true,
    });
  } else {
    describeNotebookFigures();
  }
})();
