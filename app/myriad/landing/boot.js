// Runs before the first paint (a blocking script in <head>, no inline code: strict CSP): applies the
// remembered theme and language so that the page does not flash. Storage may be unavailable.
(function () {
  "use strict";
  var d = document.documentElement, theme = null, lang = null;
  try {
    theme = window.localStorage.getItem("myriad.site.theme");
    lang = window.localStorage.getItem("myriad.site.lang");
  } catch (e) { /* private mode, blocked storage */ }
  try {
    var q = new URLSearchParams(window.location.search).get("lang");
    if (q === "fr" || q === "en") lang = q;
  } catch (e) { /* old browser */ }
  if (theme === "light" || theme === "dark") d.setAttribute("data-theme", theme);
  if (lang === "en") d.setAttribute("data-lang", "en");
  d.classList.add("js");
})();
