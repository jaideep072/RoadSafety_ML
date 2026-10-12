import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

export const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const getDataSummary = async (datasetType = 'original') => {
  const response = await apiClient.get(`/api/data-summary?dataset=${datasetType}`);
  return response.data.data;
};

export const getEdaAnalysis = async () => {
  const response = await apiClient.get(`/api/eda?t=${Date.now()}`);
  return response.data.data;
};

export const getPreprocessing = async () => {
  const response = await apiClient.get(`/api/preprocessing`);
  return response.data.data;
};

export const getChartUrl = (filename) => {
  return `${API_BASE}/api/charts/${filename}?t=${Date.now()}`;
};

export default apiClient;
