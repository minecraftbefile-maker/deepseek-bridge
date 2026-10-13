// Toolbar icon: focus an existing chat.deepseek.com tab, or open a new one.

chrome.action.onClicked.addListener(async () => {
  try {
    const tabs = await chrome.tabs.query({
      url: ["https://chat.deepseek.com/*"]
    });

    if (tabs.length > 0) {
      const t = tabs[0];
      await chrome.tabs.update(t.id, { active: true });
      if (t.windowId != null) {
        await chrome.windows.update(t.windowId, { focused: true });
      }
      try {
        await chrome.tabs.sendMessage(t.id, { type: "force-check" });
      } catch (_) {}
    } else {
      await chrome.tabs.create({ url: "https://chat.deepseek.com/" });
    }
  } catch (err) {
    console.error("[DS] action click failed:", err);
  }
});

console.log("[DS] background ready");