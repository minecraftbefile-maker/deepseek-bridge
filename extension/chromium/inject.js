// Runs in the page's main world. Captures API request headers plus
// navigator-derived browser info (UA, languages, platform) that JS
// cannot see on Request objects.

(function () {
  "use strict";
  if (window.__ds_bridge_hooked) return;
  window.__ds_bridge_hooked = true;

  function headersToObj(h) {
    const out = {};
    if (!h) return out;
    try {
      if (typeof Headers !== "undefined" && h instanceof Headers) {
        h.forEach((v, k) => { out[k] = v; });
      } else if (Array.isArray(h)) {
        for (const pair of h) out[String(pair[0])] = pair[1];
      } else if (typeof h === "object") {
        for (const k in h) out[k] = h[k];
      }
    } catch (_) {}
    return out;
  }

  // Real browser identity. These are the headers the browser adds
  // automatically to every request, which JavaScript cannot see on
  // the Request object itself — but we can read them off navigator.
  function getNavigatorHeaders() {
    const out = {};
    try {
      const ua = navigator.userAgent;
      if (ua) out["user-agent"] = ua;

      const langs = (navigator.languages && navigator.languages.length)
        ? navigator.languages.slice(0, 3)
        : (navigator.language ? [navigator.language] : []);
      if (langs.length) {
        // Basic q-value expansion matching what a real browser sends.
        const parts = [langs[0]];
        for (let i = 1; i < langs.length; i++) {
          parts.push(langs[i] + ";q=" + (1 - i * 0.1).toFixed(1));
        }
        out["accept-language"] = parts.join(",");
      }

      // Chrome/Edge expose userAgentData with structured client hints.
      if (navigator.userAgentData) {
        const uad = navigator.userAgentData;
        if (uad.platform) {
          out["sec-ch-ua-platform"] = '"' + uad.platform + '"';
        }
        if (Array.isArray(uad.brands) && uad.brands.length) {
          out["sec-ch-ua"] = uad.brands
            .map((b) => '"' + b.brand + '";v="' + b.version + '"')
            .join(", ");
        }
        if (typeof uad.mobile === "boolean") {
          out["sec-ch-ua-mobile"] = uad.mobile ? "?1" : "?0";
        }
      } else if (navigator.platform) {
        // Firefox and older Safari don't ship userAgentData.
        out["sec-ch-ua-platform"] = '"' + navigator.platform + '"';
      }
    } catch (_) {}
    return out;
  }

  function emit(url, headers) {
    try {
      const u = new URL(url, location.origin);
      if (!u.hostname.endsWith("deepseek.com")) return;
      if (!u.pathname.startsWith("/api/")) return;
    } catch (_) { return; }

    const obj = headersToObj(headers);
    // Layer the navigator values under any explicit page headers so
    // that if the page somehow set them (rare), page wins.
    const merged = Object.assign({}, getNavigatorHeaders(), obj);
    if (!merged || Object.keys(merged).length === 0) return;

    window.postMessage({
      type: "DS_CAPTURED",
      url: url,
      headers: merged,
      origin: location.origin,
      ts: Date.now()
    }, "*");
  }

  const _fetch = window.fetch;
  window.fetch = function (input, init) {
    try {
      let url = null, headers = null;
      if (typeof input === "string") {
        url = input;
        headers = init && init.headers;
      } else if (input && input.url) {
        url = input.url;
        headers = (init && init.headers) || input.headers;
      }
      emit(url, headers);
    } catch (_) {}
    return _fetch.apply(this, arguments);
  };

  const _open = XMLHttpRequest.prototype.open;
  const _setH = XMLHttpRequest.prototype.setRequestHeader;
  const _send = XMLHttpRequest.prototype.send;

  XMLHttpRequest.prototype.open = function (m, url) {
    this.__ds_url = url;
    this.__ds_h = {};
    return _open.apply(this, arguments);
  };
  XMLHttpRequest.prototype.setRequestHeader = function (name, value) {
    try { this.__ds_h[String(name)] = value; } catch (_) {}
    return _setH.apply(this, arguments);
  };
  XMLHttpRequest.prototype.send = function () {
    try { emit(this.__ds_url, this.__ds_h); } catch (_) {}
    return _send.apply(this, arguments);
  };

  // Also emit once on load, so the bridge gets a UA even before any
  // API call is made.
  try {
    window.postMessage({
      type: "DS_CAPTURED",
      url: location.origin + "/api/",
      headers: getNavigatorHeaders(),
      origin: location.origin,
      ts: Date.now()
    }, "*");
  } catch (_) {}

  console.log("[DS/inject] hooks installed");
})();