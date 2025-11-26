/**
 * API Client for LSTM Seismic Risk Prediction
 * Use this in your frontend to connect to the Flask API
 */

const API_BASE_URL = 'http://localhost:5000';  // Change this to your API URL

class SeismicRiskAPI {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  /**
   * Check if API is healthy
   */
  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return { status: 'error', error: error.message };
    }
  }

  /**
   * Get single prediction from 24 hours of data
   * @param {Array} data - Array of 24 data points with features
   * @returns {Promise<Object>} Prediction result
   */
  async predict(data) {
    try {
      const response = await fetch(`${this.baseURL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Prediction failed');
      }

      return await response.json();
    } catch (error) {
      console.error('Prediction error:', error);
      throw error;
    }
  }

  /**
   * Get batch predictions for multiple time periods
   * @param {Array} batchData - Array of arrays, each with 24 hours of data
   * @returns {Promise<Array>} Array of predictions
   */
  async predictBatch(batchData) {
    try {
      const response = await fetch(`${this.baseURL}/predict/batch`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ data: batchData }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Batch prediction failed');
      }

      const result = await response.json();
      return result.predictions;
    } catch (error) {
      console.error('Batch prediction error:', error);
      throw error;
    }
  }

  /**
   * Get 7-day forecast
   * @param {string} startDate - Start date (ISO format)
   * @param {Array} historicalData - Last 24 hours of historical data
   * @param {Array} futureOperationalData - Optional future operational data
   * @returns {Promise<Array>} 7-day forecast
   */
  async getForecast(startDate, historicalData, futureOperationalData = null) {
    try {
      const payload = {
        start_date: startDate,
        historical_data: historicalData,
      };

      if (futureOperationalData) {
        payload.future_operational_data = futureOperationalData;
      }

      const response = await fetch(`${this.baseURL}/predict/forecast`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Forecast failed');
      }

      const result = await response.json();
      return result.forecast;
    } catch (error) {
      console.error('Forecast error:', error);
      throw error;
    }
  }

  /**
   * Convert operational data to API format
   * Helper function to format your data for the API
   * @param {Array} operationalData - Your operational metrics
   * @param {Array} seismicData - Your seismic events (optional)
   * @returns {Array} Formatted data for API
   */
  formatDataForAPI(operationalData, seismicData = []) {
    return operationalData.map((record, index) => {
      // Find corresponding seismic data if available
      const seismic = seismicData.find(s => 
        new Date(s.occurred_at).toISOString().split('T')[0] === 
        new Date(record.recorded_at).toISOString().split('T')[0]
      ) || {};

      return {
        timestamp: record.recorded_at || record.timestamp,
        inj_flow: record.inj_flow || 0,
        inj_whp: record.inj_whp || 0,
        inj_temp: record.inj_temp || 0,
        prod_temp: record.prod_temp || 0,
        prod_whp: record.prod_whp || 0,
        event_count: seismic.count || 0,
        max_magnitude: seismic.max_magnitude || seismic.magnitude || 0,
        avg_magnitude: seismic.avg_magnitude || seismic.magnitude || 0,
        max_pgv: seismic.max_pgv || seismic.pgv_max || 0,
        avg_pgv: seismic.avg_pgv || seismic.pgv || 0,
      };
    });
  }
}

// Export for use in React/vanilla JS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SeismicRiskAPI;
}

