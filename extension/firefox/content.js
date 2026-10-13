(function () {
  "use strict";

  const BRIDGE = "http://127.0.0.1:9876/auth";
  const TOKEN_KEY = "userToken";
  const INSTANCE_KEY = "__ds_bridge_instance_id";

  let deviceId = null;
  let capturedHeaders = null;
  let lastSig = null;
  let instanceId = null;

  function getInstanceId() {
    return new Promise((resolve) => {
      browser.storage.local.get([INSTANCE_KEY]).then((res) => {
        if (res && res[INSTANCE_KEY]) {
          resolve(res[INSTANCE_KEY]);
        } else {
          const id = "firefox-" + Math.random().toString(36).slice(2, 10)
                   + "-" + Date.now().toString(36);
          browser.storage.local.set({ [INSTANCE_KEY]: id })
            .then(() => resolve(id));
        }
      }).catch(() => resolve("firefox-unknown"));
    });
  }

  const s = document.createElement("script");
  s.src = browser.runtime.getURL("inject.js");
  s.onload = () => s.remove();
  (document.head || document.documentElement).appendChild(s);

  function readToken() {
    const raw = localStorage.getItem(TOKEN_KEY);
    if (!raw) return null;
    try {
      const p = JSON.parse(raw);
      if (p && typeof p.value === "string") return p.value;
    } catch (_) {}
    return raw.length > 20 ? raw : null;
  }

  window.addEventListener("message", (e) => {
    if (e.source !== window) return;
    const d = e.data;
    if (!d || d.type !== "DS_CAPTURED") return;

    capturedHeaders = Object.assign({}, capturedHeaders || {}, d.headers);

    for (const k of Object.keys(d.headers)) {
      if (k.toLowerCase() === "x-device-id") {
        deviceId = d.headers[k];
        break;
      }
    }
    if (!deviceId) {
      const stored = localStorage.getItem("x-device-id")
                  || localStorage.getItem("device_id");
      if (stored) deviceId = stored;
    }

    console.log("[DS] captured headers, device=" + (deviceId || "?"));
    send();
  });

  function send() {
    const token = readToken();
    if (!token && !deviceId && !capturedHeaders) return;

    const payload = {
      token: token,
      deviceId: deviceId,
      headers: capturedHeaders,
      source: "firefox",
      instanceId: instanceId,
      origin: location.origin,
      ts: Date.now()
    };

    const sig = JSON.stringify({
      t: token ? token.slice(0, 30) : null,
      d: deviceId,
      h: capturedHeaders ? Object.keys(capturedHeaders).sort().join(",") : null,
      ua: capturedHeaders ? capturedHeaders["user-agent"] : null
    });
    if (sig === lastSig) return;

    fetch(BRIDGE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then((r) => r.json())
      .then((data) => {
        if (data && data.ok) {
          lastSig = sig;
          console.log("[DS] pushed: instance=" + instanceId
                    + " token=" + (token ? "yes" : "no")
                    + " device=" + (deviceId ? deviceId.slice(0, 12) : "?")
                    + " headers=" + (capturedHeaders
                        ? Object.keys(capturedHeaders).length : 0)
                    + " ua=" + (capturedHeaders && capturedHeaders["user-agent"]
                        ? capturedHeaders["user-agent"].slice(0, 40) : "?"));
        }
      })
      .catch((err) => console.debug("[DS] bridge unreachable:", err.message));
  }

  let lastHref = location.href;
  setInterval(() => {
    if (location.href !== lastHref) {
      lastHref = location.href;
      console.log("[DS] SPA navigation -> " + location.href);
      lastSig = null;
      send();
    }
  }, 1000);

  getInstanceId().then((id) => {
    instanceId = id;
    console.log("[DS] instance=" + instanceId + " ready on " + location.origin);
    send();
  });

  setInterval(send, 5000);

  window.addEventListener("storage", (e) => {
    if (e.key === TOKEN_KEY) send();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      lastSig = null;
      send();
    }
  });

  browser.runtime.onMessage.addListener((msg) => {
    if (msg && msg.type === "force-check") {
      lastSig = null;
      send();
    }
  });
})();