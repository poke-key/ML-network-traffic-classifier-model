'use client';

import { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Upload, FileText, X, CheckCircle, Network, BarChart3, FileSpreadsheet } from 'lucide-react';

import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface PredictionResult {
  index: number;
  category: number;
  label: string;
}

interface CategoryCount {
  category: string;
  count: number;
}

const labelMap: { [key: number]: string } = {
  0: "Streaming",
  1: "Secure", 
  2: "DNS",
  3: "Web",
  4: "Other"
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [data, setData] = useState<any[]>([]);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [categoryCounts, setCategoryCounts] = useState<CategoryCount[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile && uploadedFile.type === 'text/csv') {
      setFile(uploadedFile);
      setError(null);
      
      Papa.parse(uploadedFile, {
        header: true,
        complete: (results) => {
          setData(results.data as any[]);
          generatePredictions(results.data as any[]);
        },
        error: (error) => {
          setError('Error parsing CSV file');
          console.error('CSV parsing error:', error);
        }
      });
    } else {
      setError('Please upload a valid CSV file');
    }
  };

  const generatePredictions = async (inputData?: any[]) => {
    if (!file) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      console.log('Starting prediction for file:', file.name);
      
      const formData = new FormData();
      formData.append('file', file);
      
      console.log('Sending request to /api/predict');
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });
      
      console.log('Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Response error:', errorText);
        throw new Error(`Failed to get predictions: ${response.status}`);
      }
      
      const result = await response.json();
      
      console.log('API Response:', result);
      console.log('Predictions:', result.predictions);
      console.log('Category Counts:', result.categoryCounts);
      
      if (result.error) {
        throw new Error(result.error);
      }
      
      setPredictions(result.predictions || []);
      setCategoryCounts(result.categoryCounts || []);
    } catch (err) {
      setError(`Failed to generate predictions: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadSampleData = async () => {
    try {
      console.log('Loading sample data...');
      const response = await fetch('/sample_data.csv');
      const csvText = await response.text();
      
      // Create a virtual file object
      const sampleFile = new File([csvText], 'sample_data.csv', { type: 'text/csv' });
      setFile(sampleFile);
      
      // Parse the CSV data using the File object
      Papa.parse(sampleFile, {
        header: true,
        complete: (results) => {
          setData(results.data as any[]);
          // Now that we have the file set, we can generate predictions
          setTimeout(() => generatePredictions(), 100);
        },
        error: (error) => {
          setError('Error parsing sample CSV file');
          console.error('Sample CSV parsing error:', error);
        }
      });
    } catch (error) {
      console.error('Error loading sample data:', error);
      setError('Failed to load sample data');
    }
  };

  const removeFile = () => {
    setFile(null);
    setData([]);
    setPredictions([]);
    setCategoryCounts([]);
    setError(null);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const totalPredictions = predictions.length;
  const maxCount = categoryCounts.length > 0 ? Math.max(...categoryCounts.map(c => c.count)) : 0;

  // Load sample data automatically when component mounts
  useEffect(() => {
    loadSampleData();
  }, []);

  return (
    <div className="min-h-screen bg-background">
      {/* NUBROAD Header */}
      <header className="bg-white border-b border-gray-200 dark:bg-gray-900 dark:border-gray-700 relative z-50">
        {/* Drop shadow effect */}
        <div className="absolute inset-0 shadow-lg" style={{
          boxShadow: '0px 12px 48px 12px rgba(0, 0, 0, 0.1)'
        }}></div>
        
        <div className="max-w-7xl mx-auto px-4 relative">
          <div className="flex items-center justify-between h-24">
            {/* Logo and Brand */}
            <div className="flex items-center">
              <div className="flex items-center space-x-4">
                <div className="header-title-logo">
                  <a href="/" className="block">
                    <img 
                      src="/nubroad-logo.jpg" 
                      alt="NUBROAD" 
                      className="w-16 h-16 object-contain"
                      style={{ display: 'block' }}
                    />
                  </a>
                </div>
              </div>
            </div>

            {/* Theme Toggle */}
            <div className="flex items-center space-x-2">
              <ThemeToggle />
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Hero Section */}
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold tracking-tight">
            Network Traffic Classification with Sample Data
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Sample network traffic data is preloaded for immediate analysis. Our machine learning model classifies traffic into categories like Streaming, Web, DNS, and Secure connections.
          </p>
        </div>

        {/* File Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileSpreadsheet className="h-5 w-5" />
              Sample Data Loaded
            </CardTitle>
            <CardDescription>
              Sample network traffic data is preloaded for demonstration. You can also upload your own CSV file.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {!file ? (
              <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-8 text-center">
                <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg font-medium mb-2">Drag and drop file here</p>
                <p className="text-sm text-muted-foreground mb-4">Limit 200MB per file • CSV format</p>
                <Button asChild>
                  <label className="cursor-pointer">
                    Browse files
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </label>
                </Button>
              </div>
            ) : (
              <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="h-5 w-5 text-primary" />
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-muted-foreground">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  <span className="text-sm text-green-600">File successfully loaded</span>
                  <Button variant="ghost" size="sm" onClick={removeFile}>
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
            
            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Data Preview */}
        {data.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Input Preview
              </CardTitle>
              <CardDescription>
                First 5 rows of your uploaded data
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      {Object.keys(data[0] || {}).map((header) => (
                        <TableHead key={header} className="text-xs">
                          {header}
                        </TableHead>
                      ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {data.slice(0, 5).map((row, index) => (
                      <TableRow key={index}>
                        {Object.values(row).map((value: any, colIndex) => (
                          <TableCell key={colIndex} className="text-xs">
                            {value}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Predictions */}
        {(predictions.length > 0 || categoryCounts.length > 0) && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Predicted Traffic Categories
              </CardTitle>
              <CardDescription>
                Analysis results and category distribution
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {predictions.length > 0 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">Individual Predictions</h4>
                    <Badge variant="secondary">{predictions.length} total</Badge>
                  </div>
                  <div className="grid grid-cols-5 gap-2">
                    {predictions.slice(0, 20).map((pred) => (
                      <div key={pred.index} className="flex items-center justify-between p-2 bg-muted rounded text-xs">
                        <span className="text-muted-foreground">{pred.index}:</span>
                        <Badge variant="outline" className="text-xs">
                          {pred.category}
                        </Badge>
                      </div>
                    ))}
                    {predictions.length > 20 && (
                      <div className="col-span-5 text-center text-sm text-muted-foreground">
                        ... and {predictions.length - 20} more predictions
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Category Distribution */}
              {categoryCounts.length > 0 && (
                <div className="space-y-4">
                  <h4 className="font-medium">Traffic Category Distribution</h4>
                  
                  {/* Category Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {categoryCounts.map((category) => (
                      <div key={category.category} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">{category.category}</span>
                          <Badge variant="secondary">{category.count}</Badge>
                        </div>
                        <Progress 
                          value={(category.count / maxCount) * 100} 
                          className="h-2"
                        />
                        <p className="text-xs text-muted-foreground">
                          {((category.count / totalPredictions) * 100).toFixed(1)}% of total
                        </p>
                      </div>
                    ))}
                  </div>

                  {/* Bar Chart */}
                  <div className="h-80 bg-muted/50 rounded-lg p-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={categoryCounts} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--muted-foreground))" />
                        <XAxis 
                          dataKey="category" 
                          stroke="hsl(var(--muted-foreground))"
                          angle={-45}
                          textAnchor="end"
                          height={80}
                          fontSize={12}
                        />
                        <YAxis 
                          stroke="hsl(var(--muted-foreground))" 
                          fontSize={12}
                          label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: 'hsl(var(--muted-foreground))' } }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            color: 'hsl(var(--foreground))'
                          }}
                        />
                        <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

        {/* Loading State */}
        {isLoading && (
          <Card>
            <CardContent className="flex items-center justify-center py-12">
              <div className="text-center space-y-4">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
                <p className="text-muted-foreground">Analyzing network traffic patterns...</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Manual Test Button */}
        {data.length > 0 && predictions.length === 0 && !isLoading && (
          <Card>
            <CardContent className="flex items-center justify-center py-8">
              <div className="text-center space-y-4">
                <p className="text-muted-foreground">Data loaded but predictions not generated yet.</p>
                <Button onClick={() => generatePredictions(data)}>
                  Generate Predictions
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
} 