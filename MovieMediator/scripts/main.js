let page = 0;
let currentQuery = "";

async function searchMovies(newSearch = true) {
  if (newSearch) {
    page = 0;
    currentQuery = document.getElementById("queryInput").value;
  }

  const response = await fetch(`/recommend?query=${encodeURIComponent(currentQuery)}&page=${page}`);
  const data = await response.json();

  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";

  if (!data.results || !data.results.length) {
    resultsDiv.innerHTML = "<p>No more results.</p>";
    return;
  }

  data.results.forEach((movie, index) => {
    const div = document.createElement("div");
    div.className = "movie";
    div.style.animationDelay = `${index * 0.2}s`; // optional stagger
  
    div.innerHTML = `
      <div class="movie-inner">
        ${movie.poster ? `<img src="${movie.poster}" alt="${movie.title} poster">` : "<p>No poster available.</p>"}
        <h2>${movie.title}</h2>
      </div>
    `;
  
    resultsDiv.appendChild(div);
  });
}

function nextPage() {
  page++;
  searchMovies(false);
}

function prevPage() {
  if (page > 0) {
    page--;
    searchMovies(false);
  }
}

//Trigger search when user presses Enter in the input
document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("queryInput").addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        searchMovies(true);
      }
    });
  });