/**
 * Example: How to integrate the API into your React dashboard
 * Replace the generateMockData function with API calls
 */

// 1. Include the API client in your HTML (before your React code):
// <script src="api/api_client.js"></script>

// 2. Initialize the API client
const api = new SeismicRiskAPI('http://localhost:5000');

// 3. Replace generateMockData with this function:
async function fetchDataFromAPI(startDate, endDate) {
  try {
    // First, check if API is available
    const health = await api.healthCheck();
    if (health.status !== 'healthy') {
      console.warn('API not available, using mock data');
      return generateMockData(startDate, endDate, 500); // Fallback
    }

    // TODO: Load your actual operational and seismic data
    // This is a placeholder - you need to load from CSV or database
    const operationalData = await loadOperationalData(startDate, endDate);
    const seismicData = await loadSeismicData(startDate, endDate);

    // Format data for API
    const formattedData = api.formatDataForAPI(operationalData, seismicData);

    // Get predictions for each 24-hour window
    const predictions = [];
    const dates = [];
    const injFlow = [];
    const injWhp = [];
    const magnitude = [];
    const yellow = [];
    const orange = [];
    const red = [];

    // Process data in 24-hour windows
    for (let i = 24; i < formattedData.length; i += 24) {
      const windowData = formattedData.slice(i - 24, i);
      
      try {
        const prediction = await api.predict(windowData);
        
        // Extract data for charts
        const date = new Date(windowData[windowData.length - 1].timestamp);
        dates.push(date.toISOString().split('T')[0]);
        
        // Get operational metrics from last data point
        injFlow.push(windowData[windowData.length - 1].inj_flow);
        injWhp.push(windowData[windowData.length - 1].inj_whp);
        magnitude.push(windowData[windowData.length - 1].max_magnitude);
        
        // Get probabilities from prediction
        yellow.push(prediction.probabilities.yellow * 100);
        orange.push(prediction.probabilities.orange * 100);
        red.push(prediction.probabilities.red * 100);
        
        predictions.push(prediction);
      } catch (error) {
        console.error('Prediction error:', error);
        // Use fallback values
        dates.push(new Date(windowData[windowData.length - 1].timestamp).toISOString().split('T')[0]);
        injFlow.push(windowData[windowData.length - 1].inj_flow);
        injWhp.push(windowData[windowData.length - 1].inj_whp);
        magnitude.push(windowData[windowData.length - 1].max_magnitude);
        yellow.push(0);
        orange.push(0);
        red.push(0);
      }
    }

    return { dates, injFlow, injWhp, magnitude, yellow, orange, red, predictions };
  } catch (error) {
    console.error('Error fetching data from API:', error);
    // Fallback to mock data
    return generateMockData(startDate, endDate, 500);
  }
}

// 4. Update your OperationalDashboard component:
function OperationalDashboard() {
  const [startDate, setStartDate] = useState('2018-12-01');
  const [endDate, setEndDate] = useState('2025-09-22');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const chartRef = useRef(null);
  const [chartInstance, setChartInstance] = useState(null);

  // Load data when dates change
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      const result = await fetchDataFromAPI(startDate, endDate);
      setData(result);
      setLoading(false);
    };
    loadData();
  }, [startDate, endDate]);

  // Update chart when data changes
  useEffect(() => {
    if (!data || loading) return;

    const timer = setTimeout(() => {
      const ctx = document.getElementById('operationalChart');
      if (!ctx) return;

      if (chartInstance) {
        chartInstance.destroy();
      }

      const filteredLabels = data.dates.filter((_, i) => i % 50 === 0);
      const chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: filteredLabels,
          datasets: [
            {
              label: 'Injection Flow [m³/h]',
              data: data.injFlow.filter((_, i) => i % 50 === 0),
              // ... rest of your chart config
            }
            // ... other datasets
          ]
        },
        // ... rest of your chart options
      });

      setChartInstance(chart);
    }, 100);

    return () => clearTimeout(timer);
  }, [data, loading]);

  return (
    <div>
      {loading && <div>Loading predictions...</div>}
      {/* Your existing UI */}
    </div>
  );
}

// 5. For Risk Dashboard - Get 7-day forecast:
async function fetchRiskForecast() {
  try {
    // Get last 24 hours of data
    const historicalData = await getLast24HoursData();
    
    // Get forecast
    const forecast = await api.getForecast(
      new Date().toISOString(),
      historicalData
    );

    // Format for chart
    const dates = forecast.map(f => f.date);
    const yellow = forecast.map(f => f.probabilities.yellow * 100);
    const orange = forecast.map(f => f.probabilities.orange * 100);
    const red = forecast.map(f => f.probabilities.red * 100);

    return { dates, yellow, orange, red };
  } catch (error) {
    console.error('Forecast error:', error);
    return generateMockData('2023-07-01', '2025-07-31', 730);
  }
}

// Helper functions you need to implement:
async function loadOperationalData(startDate, endDate) {
  // Load from CSV, database, or API
  // Return array of operational metrics
  // Example:
  // const response = await fetch('/data/operational_metrics.csv');
  // const csv = await response.text();
  // return parseCSV(csv);
  return [];
}

async function loadSeismicData(startDate, endDate) {
  // Load seismic events
  // Return array of seismic events
  return [];
}

async function getLast24HoursData() {
  // Get the most recent 24 hours of data
  // This should match the format expected by the API
  return [];
}

// Keep your original mock data function as fallback
function generateMockData(startDate, endDate, points) {
  // Your existing mock data generation code
  // ...
}

