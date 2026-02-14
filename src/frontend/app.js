/**
 * Movie Recommender — Frontend Application Logic
 *
 * Handles auth, fetching recommendations, search, and rendering.
 */

const API_URL = window.location.hostname === "localhost"
    ? "http://localhost:5000"
    : window.API_URL || "";

let currentUser = null;
let searchTimeout = null;

// ============================================================
// Initialization
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    checkAuth();
    loadMetrics();
});

// ============================================================
// Authentication
// ============================================================

async function checkAuth() {
    try {
        const res = await fetch(`${API_URL}/auth/me`, { credentials: "include" });
        const data = await res.json();
        if (data.authenticated) {
            setUser(data.user);
        }
    } catch (e) {
        console.log("Auth check skipped — API may not be running");
    }
}

function setUser(user) {
    currentUser = user;
    document.getElementById("authArea").classList.add("hidden");
    document.getElementById("userArea").classList.remove("hidden");
    document.getElementById("loginPrompt").classList.add("hidden");
    document.getElementById("recommendations").classList.remove("hidden");

    document.getElementById("userName").textContent = user.login;
    const avatar = document.getElementById("userAvatar");
    if (user.avatar_url) {
        avatar.src = user.avatar_url;
    } else {
        avatar.src = `https://ui-avatars.com/api/?name=${encodeURIComponent(user.login)}&background=388bfd&color=fff&size=64`;
    }

    loadRecommendations(user.id);
    loadUserRatings(user.id);
}

function handleLogin() {
    window.location.href = `${API_URL}/auth/github`;
}

async function handleDemoLogin() {
    try {
        const res = await fetch(`${API_URL}/auth/demo`, { credentials: "include" });
        const data = await res.json();
        if (data.user) {
            setUser(data.user);
        }
    } catch (e) {
        // Fallback: use client-side demo mode
        setUser({
            id: 1,
            login: "Demo Reviewer",
            avatar_url: null,
            is_demo: true,
        });
    }
}

async function handleLogout() {
    try {
        await fetch(`${API_URL}/auth/logout`, { credentials: "include" });
    } catch (e) { /* ignore */ }

    currentUser = null;
    document.getElementById("authArea").classList.remove("hidden");
    document.getElementById("userArea").classList.add("hidden");
    document.getElementById("loginPrompt").classList.remove("hidden");
    document.getElementById("recommendations").classList.add("hidden");
}

// ============================================================
// Metrics
// ============================================================

async function loadMetrics() {
    try {
        const res = await fetch(`${API_URL}/api/metrics`);
        const data = await res.json();

        if (data.mae_reduction_pct) {
            document.getElementById("metricMAE").textContent = `~${data.mae_reduction_pct}%`;
        }
        if (data.algorithm) {
            document.getElementById("metricAlgorithm").textContent = data.algorithm.split(" ")[0];
        }
    } catch (e) {
        console.log("Metrics unavailable — will show defaults");
    }
}

// ============================================================
// Recommendations
// ============================================================

async function loadRecommendations(userId) {
    const grid = document.getElementById("moviesGrid");
    const loading = document.getElementById("loading");

    loading.classList.remove("hidden");
    grid.innerHTML = "";

    try {
        const res = await fetch(`${API_URL}/api/recommendations/${userId}?n=12`);
        const data = await res.json();

        loading.classList.add("hidden");

        if (data.recommendations && data.recommendations.length > 0) {
            // Update latency metric
            if (data.latency_ms) {
                document.getElementById("metricLatency").textContent = `${data.latency_ms.toFixed(0)}ms`;
            }

            data.recommendations.forEach(movie => {
                grid.appendChild(createMovieCard(movie));
            });
        } else {
            grid.innerHTML = '<p class="loading">No recommendations available for this user.</p>';
        }
    } catch (e) {
        loading.classList.add("hidden");
        grid.innerHTML = `<p class="loading">Could not load recommendations. Is the API running at ${API_URL}?</p>`;
    }
}

async function loadUserRatings(userId) {
    const grid = document.getElementById("ratingsGrid");
    try {
        const res = await fetch(`${API_URL}/api/user/${userId}/ratings`);
        const data = await res.json();

        if (data.ratings && data.ratings.length > 0) {
            grid.innerHTML = "";
            data.ratings.slice(0, 15).forEach(r => {
                const chip = document.createElement("div");
                chip.className = "rating-chip";
                chip.innerHTML = `
                    <span class="rating-chip-star">★ ${r.rating}</span>
                    <span>${r.title}${r.year ? ` (${r.year})` : ""}</span>
                `;
                grid.appendChild(chip);
            });
        }
    } catch (e) {
        console.log("User ratings unavailable");
    }
}

// ============================================================
// Search
// ============================================================

function handleSearch(query) {
    clearTimeout(searchTimeout);

    const searchResults = document.getElementById("searchResults");
    const recsContent = document.getElementById("recsContent");
    const userRatings = document.getElementById("userRatings");

    if (!query || query.length < 2) {
        searchResults.classList.add("hidden");
        recsContent.classList.remove("hidden");
        userRatings.classList.remove("hidden");
        return;
    }

    searchTimeout = setTimeout(async () => {
        try {
            const res = await fetch(`${API_URL}/api/movies/search?q=${encodeURIComponent(query)}`);
            const data = await res.json();

            recsContent.classList.add("hidden");
            userRatings.classList.add("hidden");
            searchResults.classList.remove("hidden");

            const grid = document.getElementById("searchGrid");
            grid.innerHTML = "";

            if (data.movies && data.movies.length > 0) {
                data.movies.forEach(movie => {
                    grid.appendChild(createMovieCard({
                        ...movie,
                        score: null,
                        rationale: `${movie.genres ? movie.genres.join(", ") : ""}`,
                    }));
                });
            } else {
                grid.innerHTML = '<p class="loading">No movies found.</p>';
            }
        } catch (e) {
            console.log("Search failed");
        }
    }, 300);
}

// ============================================================
// Movie Card Component
// ============================================================

function createMovieCard(movie) {
    const card = document.createElement("div");

    // Determine background class based on first genre
    const firstGenre = (movie.genres && movie.genres.length > 0) ? movie.genres[0].toLowerCase() : "default";
    const bgClass = getGenreClass(firstGenre);

    card.className = `movie-card ${bgClass}`;

    const genreTags = (movie.genres || [])
        .slice(0, 2) // Max 2 tags to keep it clean
        .map(g => `<span class="genre-tag">${g}</span>`)
        .join("");

    const scoreHtml = movie.score
        ? `<div class="movie-score">${movie.score.toFixed(1)}</div>`
        : "";

    // New structure: content overlaid on gradient background
    card.innerHTML = `
        <div class="movie-overlay"></div>
        <div class="movie-content">
            <div class="movie-year">${movie.year ? parseInt(movie.year) : ""}</div>
            <div class="movie-title">${escapeHtml(movie.title)}</div>
            <div class="movie-genres">${genreTags}</div>
            ${movie.rationale ? `<div class="movie-rationale">${escapeHtml(movie.rationale)}</div>` : ""}
        </div>
        ${scoreHtml}
    `;

    return card;
}

function getGenreClass(genre) {
    const map = {
        "action": "bg-action", "adventure": "bg-adventure",
        "animation": "bg-animation", "comedy": "bg-comedy",
        "crime": "bg-crime", "documentary": "bg-documentary",
        "drama": "bg-drama", "fantasy": "bg-fantasy",
        "horror": "bg-horror", "mystery": "bg-crime",
        "romance": "bg-romance", "sci-fi": "bg-scifi",
        "thriller": "bg-thriller", "war": "bg-action",
        "western": "bg-drama"
    };
    return map[genre] || "bg-default";
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
