/* =========================================================================
   AI Maintenance Cockpit - shell behaviour
   Live clock, sidebar toggle, slider readouts, flash dismissal, assistant.
   ========================================================================= */
(function () {
    "use strict";

    const csrfToken = document
        .querySelector('meta[name="csrf-token"]')
        ?.getAttribute("content") || "";

    /* ---------------------------------------------------------------- clock */
    function startClock() {
        const timeEl = document.getElementById("live-time");
        const dateEl = document.getElementById("live-date");
        if (!timeEl && !dateEl) return;

        const pad = (value) => String(value).padStart(2, "0");

        function tick() {
            const now = new Date();
            if (timeEl) {
                timeEl.textContent =
                    pad(now.getHours()) + ":" +
                    pad(now.getMinutes()) + ":" +
                    pad(now.getSeconds());
            }
            if (dateEl) {
                dateEl.textContent =
                    pad(now.getDate()) + "/" +
                    pad(now.getMonth() + 1) + "/" +
                    now.getFullYear();
            }
        }
        tick();
        window.setInterval(tick, 1000);
    }

    /* -------------------------------------------------------------- sidebar */
    function initSidebar() {
        const toggle = document.getElementById("nav-toggle");
        if (!toggle) return;

        const wide = () => window.innerWidth > 992;
        const stored = window.localStorage.getItem("cockpit.nav");
        if (stored === "collapsed" && wide()) {
            document.body.classList.add("nav-collapsed");
        }

        toggle.addEventListener("click", function () {
            if (wide()) {
                const collapsed = document.body.classList.toggle("nav-collapsed");
                window.localStorage.setItem(
                    "cockpit.nav", collapsed ? "collapsed" : "open"
                );
                toggle.setAttribute("aria-expanded", String(!collapsed));
            } else {
                const open = document.body.classList.toggle("nav-open");
                toggle.setAttribute("aria-expanded", String(open));
            }
        });

        document.addEventListener("click", function (event) {
            if (wide() || !document.body.classList.contains("nav-open")) return;
            const sidebar = document.getElementById("sidebar");
            if (!sidebar) return;
            if (!sidebar.contains(event.target) && !toggle.contains(event.target)) {
                document.body.classList.remove("nav-open");
                toggle.setAttribute("aria-expanded", "false");
            }
        });
    }

    /* --------------------------------------------------------------- flashes */
    function initFlashes() {
        document.querySelectorAll(".flash").forEach(function (node) {
            const close = node.querySelector("button");
            if (close) {
                close.addEventListener("click", () => node.remove());
            }
            window.setTimeout(function () {
                node.style.transition = "opacity .35s ease, transform .35s ease";
                node.style.opacity = "0";
                node.style.transform = "translateX(18px)";
                window.setTimeout(() => node.remove(), 380);
            }, 9000);
        });
    }

    /* --------------------------------------------------------------- sliders */
    function paintRange(input) {
        const min = parseFloat(input.min || "0");
        const max = parseFloat(input.max || "100");
        const value = parseFloat(input.value || "0");
        const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
        input.style.setProperty("--fill", pct.toFixed(2) + "%");
    }

    function initSliders() {
        document.querySelectorAll('input[type="range"]').forEach(function (input) {
            const target = input.dataset.output
                ? document.getElementById(input.dataset.output)
                : null;
            const mirror = input.dataset.mirror
                ? document.getElementById(input.dataset.mirror)
                : null;
            const decimals = parseInt(input.dataset.decimals || "0", 10);

            function sync() {
                paintRange(input);
                const value = parseFloat(input.value);
                if (target) target.textContent = value.toFixed(decimals);
                if (mirror) mirror.value = input.value;
            }

            input.addEventListener("input", sync);
            if (mirror) {
                mirror.addEventListener("input", function () {
                    input.value = mirror.value;
                    paintRange(input);
                    if (target) {
                        target.textContent =
                            parseFloat(input.value).toFixed(decimals);
                    }
                });
            }
            sync();
        });
    }

    /* -------------------------------------------------------------- presets */
    function initPresets() {
        document.querySelectorAll("[data-preset]").forEach(function (button) {
            button.addEventListener("click", function () {
                let values;
                try {
                    values = JSON.parse(button.dataset.preset);
                } catch (error) {
                    return;
                }
                Object.keys(values).forEach(function (key) {
                    const field = document.querySelector('[name="' + key + '"]');
                    if (field) {
                        field.value = values[key];
                        field.dispatchEvent(new Event("input", { bubbles: true }));
                        field.dispatchEvent(new Event("change", { bubbles: true }));
                    }
                    const slider = document.getElementById("range-" + key);
                    if (slider) {
                        slider.value = values[key];
                        slider.dispatchEvent(new Event("input", { bubbles: true }));
                    }
                });
            });
        });
    }

    /* ------------------------------------------------------- auto refreshers */
    function initAutoRefresh() {
        const holder = document.querySelector("[data-refresh-seconds]");
        if (!holder) return;
        const seconds = parseInt(holder.dataset.refreshSeconds || "0", 10);
        if (!seconds) return;
        window.setTimeout(() => window.location.reload(), seconds * 1000);
    }

    /* ------------------------------------------------------------- assistant */
    function initChat() {
        const fab = document.getElementById("chat-fab");
        const win = document.getElementById("chat-window");
        if (!fab || !win) return;

        const body = document.getElementById("chat-body");
        const form = document.getElementById("chat-form");
        const input = document.getElementById("chat-input");
        const closeBtn = document.getElementById("chat-close");
        const badge = fab.querySelector(".dot");

        function open() {
            win.classList.add("open");
            fab.setAttribute("aria-expanded", "true");
            if (badge) badge.remove();
            if (input) input.focus();
        }
        function close() {
            win.classList.remove("open");
            fab.setAttribute("aria-expanded", "false");
        }

        fab.addEventListener("click", function () {
            win.classList.contains("open") ? close() : open();
        });
        if (closeBtn) closeBtn.addEventListener("click", close);
        document.addEventListener("keydown", function (event) {
            if (event.key === "Escape" && win.classList.contains("open")) close();
        });

        function bubble(text, who) {
            const node = document.createElement("div");
            node.className = "bubble " + who;
            node.textContent = text;
            body.appendChild(node);
            body.scrollTop = body.scrollHeight;
            return node;
        }

        function ask(message) {
            if (!message.trim()) return;
            bubble(message, "me");
            const pending = bubble("Checking plant data ...", "bot");

            fetch("/api/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "X-CSRF-Token": csrfToken
                },
                body: JSON.stringify({ message: message })
            })
                .then((response) => {
                    if (!response.ok) throw new Error("HTTP " + response.status);
                    return response.json();
                })
                .then((data) => {
                    pending.textContent = data.reply;
                    body.scrollTop = body.scrollHeight;
                })
                .catch(() => {
                    pending.textContent =
                        "I could not reach the platform API. Check that your " +
                        "session is still signed in.";
                });
        }

        if (form) {
            form.addEventListener("submit", function (event) {
                event.preventDefault();
                const message = input.value;
                input.value = "";
                ask(message);
            });
        }

        document.querySelectorAll(".chat-chip").forEach(function (chip) {
            chip.addEventListener("click", function () {
                open();
                ask(chip.textContent.trim());
            });
        });
    }

    /* -------------------------------------------------------- history filter */
    function initTableFilter() {
        const input = document.getElementById("table-filter");
        if (!input) return;
        const rows = Array.from(
            document.querySelectorAll("[data-filter-row]")
        );
        const counter = document.getElementById("table-filter-count");

        input.addEventListener("input", function () {
            const needle = input.value.trim().toLowerCase();
            let shown = 0;
            rows.forEach(function (row) {
                const hit = !needle ||
                    row.textContent.toLowerCase().indexOf(needle) !== -1;
                row.style.display = hit ? "" : "none";
                if (hit) shown += 1;
            });
            if (counter) counter.textContent = shown;
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        startClock();
        initSidebar();
        initFlashes();
        initSliders();
        initPresets();
        initAutoRefresh();
        initChat();
        initTableFilter();
    });
})();
