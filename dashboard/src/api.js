const API_BASE = "http://127.0.0.1:8000";

async function request(path) {
  const response = await fetch(`${API_BASE}${path}`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `API request failed: ${response.status}`);
  }

  return response.json();
}

export function searchLoans(query, limit = 10) {
  return request(
    `/loans/search?q=${encodeURIComponent(query)}&limit=${limit}`
  );
}

export function getLoan(loanId) {
  return request(`/loans/${encodeURIComponent(loanId)}`);
}

export function getLoanHistory(loanId) {
  return request(`/loans/${encodeURIComponent(loanId)}/history`);
}

export function getLoanRisk(loanId) {
  return request(`/loans/${encodeURIComponent(loanId)}/risk`);
}

export function getLoanAnomalies(loanId) {
  return request(`/loans/${encodeURIComponent(loanId)}/anomalies`);
}

export function getLoanScenarios(loanId) {
  return request(`/loans/${encodeURIComponent(loanId)}/scenarios`);
}

export function getScenarioSummary() {
  return request("/scenarios/summary");
}