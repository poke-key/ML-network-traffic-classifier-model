# Network Traffic Classifier - Next.js Web Application

A modern web application for classifying network traffic using machine learning models. Built with Next.js, TypeScript, and Tailwind CSS.

## Features

- 🎯 **ML-Powered Predictions**: Uses trained SVM model for accurate traffic classification
- 📊 **Interactive Visualizations**: Beautiful charts showing prediction distributions
- 📁 **File Upload**: Drag-and-drop CSV file upload with preview
- 🎨 **Modern UI**: Dark theme with responsive design
- ⚡ **Real-time Processing**: Fast prediction results with loading states

## Traffic Categories

The model classifies network traffic into the following categories:
- **Streaming**: Video/audio streaming traffic
- **Secure**: HTTPS/SSL encrypted traffic
- **DNS**: Domain Name System queries
- **Web**: Standard web browsing traffic
- **Other**: Miscellaneous network traffic

## Prerequisites

- Node.js 18+ 
- Python 3.8+
- npm or yarn

## Installation

1. **Clone and navigate to the project:**
   ```bash
   cd network-traffic-classifier-web
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are present:**
   - `models/svm_tuned_model.pkl`
   - `models/scaler.pkl`

## Development

1. **Start the development server:**
   ```bash
   npm run dev
   ```

2. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Usage

1. **Upload CSV File**: 
   - Click "Browse files" or drag-and-drop a CSV file
   - The file should contain flow-level network features

2. **View Preview**: 
   - See the first 5 rows of your uploaded data
   - Verify the data format is correct

3. **Get Predictions**: 
   - The system automatically processes your data
   - View individual predictions and category distributions
   - Interactive bar chart shows traffic category breakdown

## API Endpoints

- `POST /api/predict` - Upload CSV file and get predictions

## File Format

Your CSV file should contain the following flow-level features:
- Source.Port
- Destination.Port  
- Protocol
- Flow.Duration
- Total.Fwd.Packets
- Total.Backward.Packets
- And other network flow features...

## Production Deployment

1. **Build the application:**
   ```bash
   npm run build
   ```

2. **Start production server:**
   ```bash
   npm start
   ```

## Technologies Used

- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Charts**: Recharts
- **File Processing**: PapaParse
- **Icons**: Lucide React
- **ML Backend**: Python, scikit-learn, pandas, joblib

## Project Structure

```
network-traffic-classifier-web/
├── src/
│   ├── app/
│   │   ├── api/predict/route.ts    # API endpoint
│   │   └── page.tsx                # Main application page
│   └── ...
├── models/                         # ML model files
├── predict.py                      # Python prediction script
├── requirements.txt                # Python dependencies
└── package.json                    # Node.js dependencies
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. 