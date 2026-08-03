(() => {
  const paletteKey = "__palette";
  const migrationKey = "silva:theme-default:v1";
  const darkPalette = {
    color: {
      scheme: "slate",
      primary: "black",
      accent: "light-blue"
    }
  };

  const applyDarkToPage = () => {
    if (document.body) {
      document.body.setAttribute("data-md-color-scheme", "slate");
      document.body.setAttribute("data-md-color-primary", "black");
      document.body.setAttribute("data-md-color-accent", "light-blue");
      return;
    }

    document.addEventListener("DOMContentLoaded", applyDarkToPage, { once: true });
  };

  const applyDarkDefault = () => {
    localStorage.setItem(paletteKey, JSON.stringify(darkPalette));
    localStorage.setItem(migrationKey, "slate");
    applyDarkToPage();
  };

  try {
    const savedPalette = localStorage.getItem(paletteKey);
    const migrated = localStorage.getItem(migrationKey);

    if (!savedPalette) {
      applyDarkDefault();
      return;
    }

    const palette = JSON.parse(savedPalette);
    const scheme = palette && palette.color && palette.color.scheme;

    if (!migrated && scheme === "default") {
      applyDarkDefault();
      return;
    }

    if (!migrated) {
      localStorage.setItem(migrationKey, scheme || "custom");
    }
  } catch {
    applyDarkToPage();
  }
})();
