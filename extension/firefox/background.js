// Toolbar icon: focus an existing chat.deepseek.com tab, or open a new one.

browser.browserAction.onClicked.addListener(async () => {
  try {
    const tabs = await browser.tabs.query({
      url: ["https://chat.deepseek.com/*"]
    });

    if (tabs.length > 0) {
      const t = tabs[0];
      await browser.tabs.update(t.id, { active: true });
      if (t.windowId != null) {
        await browser.windows.update(t.windowId, { focused: true });
      }
      try {
        await browser.tabs.sendMessage(t.id, { type: "force-check" });
      } catch (_) {}
    } else {
      await browser.tabs.create({ url: "https://chat.deepseek.com/" });
    }
  } catch (err) {
    console.error("[DS] action click failed:", err);
  }
});

console.log("[DS] background ready");