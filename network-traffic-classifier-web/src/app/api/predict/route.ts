import { NextRequest, NextResponse } from 'next/server';

// Configuration for ML service
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:5000';

export async function POST(request: NextRequest): Promise<Response> {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file uploaded' },
        { status: 400 }
      );
    }

    // Read the file content
    const csvText = await file.text();
    
    console.log('Sending CSV data to ML service:', ML_SERVICE_URL);
    
    // Call the external ML service
    const response = await fetch(`${ML_SERVICE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        csv_data: csvText
      }),
    });
    
    console.log('ML service response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('ML service error:', errorText);
      throw new Error(`ML service failed: ${response.status}`);
    }
    
    const result = await response.json();
    
    console.log('ML service response:', result);
    
    if (result.error) {
      throw new Error(result.error);
    }
    
    return NextResponse.json(result);

  } catch (error) {
    console.error('Prediction error:', error);
    
    // Fallback to mock data if ML service is unavailable
    console.log('Using fallback mock data');
    const mockPredictions = Array.from({ length: 25 }, (_, i) => ({
      index: i,
      category: Math.floor(Math.random() * 4),
      label: ['Streaming', 'Secure', 'DNS', 'Web'][Math.floor(Math.random() * 4)]
    }));
    
    const counts: { [key: string]: number } = {};
    mockPredictions.forEach(pred => {
      counts[pred.label] = (counts[pred.label] || 0) + 1;
    });
    
    const categoryCounts = Object.entries(counts).map(([category, count]) => ({
      category,
      count
    }));
    
    return NextResponse.json({
      predictions: mockPredictions,
      categoryCounts,
      message: 'Mock predictions (ML service unavailable)'
    });
  }
} 