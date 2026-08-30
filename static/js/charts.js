/* =========================================================================
   Chart.js helpers - shared theme and factory functions.
   Every chart on the platform is created through this module so the styling
   stays consistent and there is one place to change it.
   ========================================================================= */
(function (global) {
    "use strict";

    const PALETTE = {
        success: "#2ee68a",
        info: "#38bdf8",
        warning: "#f5a524",
        amber: "#fbbf24",
        danger: "#ff4257",
        primary: "#4f8cff",
        purple: "#a855f7",
        cyan: "#22d3ee",
        muted: "#8496b1"
    };

    const GRID = "rgba(255,255,255,.07)";
    const TICK = "#8496b1";

    function ready() {
        return typeof global.Chart !== "undefined";
    }

    function applyDefaults() {
        if (!ready()) return;
        const Chart = global.Chart;
        Chart.defaults.color = TICK;
        Chart.defaults.font.family =
            "Inter, 'Segoe UI', system-ui, sans-serif";
        Chart.defaults.font.size = 11;
        Chart.defaults.plugins.legend.labels.usePointStyle = true;
        Chart.defaults.plugins.legend.labels.boxWidth = 8;
        Chart.defaults.plugins.legend.labels.padding = 14;
        Chart.defaults.plugins.tooltip.backgroundColor = "rgba(8,12,21,.96)";
        Chart.defaults.plugins.tooltip.borderColor = "rgba(255,255,255,.14)";
        Chart.defaults.plugins.tooltip.borderWidth = 1;
        Chart.defaults.plugins.tooltip.padding = 11;
        Chart.defaults.plugins.tooltip.titleColor = "#e7eefc";
        Chart.defaults.plugins.tooltip.bodyColor = "#c7d4ea";
        Chart.defaults.maintainAspectRatio = false;
    }

    function colour(token) {
        return PALETTE[token] || PALETTE.primary;
    }

    function alpha(hex, amount) {
        const value = hex.replace("#", "");
        const r = parseInt(value.substring(0, 2), 16);
        const g = parseInt(value.substring(2, 4), 16);
        const b = parseInt(value.substring(4, 6), 16);
        return "rgba(" + r + "," + g + "," + b + "," + amount + ")";
    }

    function axes(options) {
        const settings = options || {};
        return {
            x: {
                grid: { color: GRID, drawBorder: false },
                ticks: { color: TICK, maxRotation: 0, autoSkipPadding: 12 }
            },
            y: {
                beginAtZero: settings.beginAtZero !== false,
                suggestedMax: settings.max,
                grid: { color: GRID, drawBorder: false },
                ticks: { color: TICK, precision: settings.precision }
            }
        };
    }

    function doughnut(canvasId, labels, values, tokens) {
        if (!ready()) return null;
        const node = document.getElementById(canvasId);
        if (!node) return null;

        return new global.Chart(node, {
            type: "doughnut",
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: tokens.map((token) => alpha(colour(token), 0.82)),
                    borderColor: tokens.map((token) => colour(token)),
                    borderWidth: 1.5,
                    hoverOffset: 8
                }]
            },
            options: {
                cutout: "62%",
                plugins: { legend: { position: "bottom" } }
            }
        });
    }

    function bars(canvasId, labels, values, tokens, options) {
        if (!ready()) return null;
        const node = document.getElementById(canvasId);
        if (!node) return null;
        const settings = options || {};
        const list = Array.isArray(tokens) ? tokens : [tokens || "primary"];

        return new global.Chart(node, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: settings.label || "Value",
                    data: values,
                    backgroundColor: labels.map(function (_, index) {
                        return alpha(colour(list[index % list.length]), 0.72);
                    }),
                    borderColor: labels.map(function (_, index) {
                        return colour(list[index % list.length]);
                    }),
                    borderWidth: 1.4,
                    borderRadius: 6,
                    maxBarThickness: settings.thickness || 42
                }]
            },
            options: {
                indexAxis: settings.horizontal ? "y" : "x",
                plugins: { legend: { display: false } },
                scales: axes({ max: settings.max, precision: settings.precision })
            }
        });
    }

    function line(canvasId, labels, series, options) {
        if (!ready()) return null;
        const node = document.getElementById(canvasId);
        if (!node) return null;
        const settings = options || {};

        return new global.Chart(node, {
            type: "line",
            data: {
                labels: labels,
                datasets: series.map(function (entry) {
                    const base = colour(entry.token || "primary");
                    return {
                        label: entry.label,
                        data: entry.values,
                        borderColor: base,
                        backgroundColor: alpha(base, 0.16),
                        borderWidth: 2,
                        pointRadius: entry.values.length > 20 ? 0 : 3,
                        pointBackgroundColor: base,
                        tension: 0.34,
                        fill: entry.fill !== false,
                        yAxisID: entry.axis || "y"
                    };
                })
            },
            options: {
                interaction: { mode: "index", intersect: false },
                plugins: {
                    legend: { display: series.length > 1, position: "bottom" }
                },
                scales: axes({
                    max: settings.max,
                    beginAtZero: settings.beginAtZero,
                    precision: settings.precision
                })
            }
        });
    }

    function mixed(canvasId, labels, barSeries, lineSeries, options) {
        if (!ready()) return null;
        const node = document.getElementById(canvasId);
        if (!node) return null;
        const settings = options || {};
        const barColour = colour(barSeries.token || "primary");
        const lineColour = colour(lineSeries.token || "warning");

        return new global.Chart(node, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [
                    {
                        type: "bar",
                        label: barSeries.label,
                        data: barSeries.values,
                        backgroundColor: alpha(barColour, 0.66),
                        borderColor: barColour,
                        borderWidth: 1.3,
                        borderRadius: 5,
                        maxBarThickness: 34,
                        order: 2
                    },
                    {
                        type: "line",
                        label: lineSeries.label,
                        data: lineSeries.values,
                        borderColor: lineColour,
                        backgroundColor: alpha(lineColour, 0.14),
                        borderWidth: 2,
                        pointRadius: 3,
                        pointBackgroundColor: lineColour,
                        tension: 0.32,
                        fill: false,
                        yAxisID: "y1",
                        order: 1
                    }
                ]
            },
            options: {
                interaction: { mode: "index", intersect: false },
                plugins: { legend: { position: "bottom" } },
                scales: {
                    x: {
                        grid: { color: GRID, drawBorder: false },
                        ticks: { color: TICK, maxRotation: 0, autoSkipPadding: 10 }
                    },
                    y: {
                        beginAtZero: true,
                        position: "left",
                        grid: { color: GRID, drawBorder: false },
                        ticks: { color: TICK },
                        title: {
                            display: !!settings.leftTitle,
                            text: settings.leftTitle,
                            color: TICK
                        }
                    },
                    y1: {
                        beginAtZero: true,
                        position: "right",
                        grid: { drawOnChartArea: false },
                        ticks: { color: TICK, precision: 0 },
                        title: {
                            display: !!settings.rightTitle,
                            text: settings.rightTitle,
                            color: TICK
                        }
                    }
                }
            }
        });
    }

    function radar(canvasId, labels, values, token) {
        if (!ready()) return null;
        const node = document.getElementById(canvasId);
        if (!node) return null;
        const base = colour(token || "cyan");

        return new global.Chart(node, {
            type: "radar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Importance %",
                    data: values,
                    borderColor: base,
                    backgroundColor: alpha(base, 0.22),
                    borderWidth: 2,
                    pointBackgroundColor: base
                }]
            },
            options: {
                plugins: { legend: { display: false } },
                scales: {
                    r: {
                        beginAtZero: true,
                        grid: { color: GRID },
                        angleLines: { color: GRID },
                        pointLabels: { color: TICK, font: { size: 10 } },
                        ticks: {
                            color: TICK,
                            backdropColor: "transparent",
                            showLabelBackdrop: false
                        }
                    }
                }
            }
        });
    }

    applyDefaults();

    global.Cockpit = {
        palette: PALETTE,
        colour: colour,
        alpha: alpha,
        doughnut: doughnut,
        bars: bars,
        line: line,
        mixed: mixed,
        radar: radar
    };
})(window);
