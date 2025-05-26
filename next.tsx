import React, { useState } from 'react';
import { AlertCircle, Heart, Activity } from 'lucide-react';

interface DiabetesFeatures {
  pregnancies: number;
  age: number;
  bloodpressure: number;
  skinthickness: number;
  glucose: number;
  insulin: number;
  bmi: number;
  diabetespedigreefunction: number;
}

interface PredictionResult {
  label: string;
  confidence: number | null;
}

const DiabetesPredictorApp: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<'home' | 'diabetes'>('home');
  const [formData, setFormData] = useState<DiabetesFeatures>({
    pregnancies: 0,
    age: 0,
    bloodpressure: 0,
    skinthickness: 0,
    glucose: 0,
    insulin: 0,
    bmi: 0,
    diabetespedigreefunction: 0
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Simple diabetes prediction model (mock implementation)
  const predictDiabetes = (features: DiabetesFeatures): PredictionResult => {
    // This is a simplified rule-based model for demonstration
    // In a real implementation, you'd use a trained ML model
    const { glucose, bmi, age, pregnancies, insulin, bloodpressure } = features;
    
    let riskScore = 0;
    
    // Glucose risk factors
    if (glucose > 140) riskScore += 3;
    else if (glucose > 110) riskScore += 2;
    else if (glucose > 100) riskScore += 1;
    
    // BMI risk factors
    if (bmi > 30) riskScore += 2;
    else if (bmi > 25) riskScore += 1;
    
    // Age risk factors
    if (age > 45) riskScore += 2;
    else if (age > 35) riskScore += 1;
    
    // Other factors
    if (pregnancies > 3) riskScore += 1;
    if (insulin > 200) riskScore += 1;
    if (bloodpressure > 80) riskScore += 1;
    
    const confidence = Math.min(95, Math.max(55, riskScore * 12 + Math.random() * 10));
    const isPositive = riskScore >= 4;
    
    return {
      label: isPositive ? 'Positive: Diabetes Likely' : 'Negative: Diabetes Unlikely',
      confidence: Math.round(confidence)
    };
  };

  const handleInputChange = (field: keyof DiabetesFeatures, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
    setError(null);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Validate inputs
      if (formData.age <= 0 || formData.glucose <= 0) {
        throw new Error('Please enter valid values for age and glucose');
      }
      
      // Simulate API delay
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const result = predictDiabetes(formData);
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction error occurred');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({
      pregnancies: 0,
      age: 0,
      bloodpressure: 0,
      skinthickness: 0,
      glucose: 0,
      insulin: 0,
      bmi: 0,
      diabetespedigreefunction: 0
    });
    setPrediction(null);
    setError(null);
  };

  if (currentPage === 'home') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <div className="container mx-auto px-4 py-16">
          <div className="text-center mb-12">
            <div className="flex justify-center mb-6">
              <Heart className="w-16 h-16 text-red-500" />
            </div>
            <h1 className="text-5xl font-bold text-gray-800 mb-4">
              Health Prediction Platform
            </h1>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Advanced machine learning models to help assess your health risks and make informed decisions about your wellbeing.
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-8 mb-12">
            <div className="bg-white rounded-xl shadow-lg p-8 hover:shadow-xl transition-shadow">
              <div className="flex items-center mb-4">
                <Activity className="w-8 h-8 text-blue-500 mr-3" />
                <h3 className="text-2xl font-semibold text-gray-800">Diabetes Risk Assessment</h3>
              </div>
              <p className="text-gray-600 mb-6">
                Get an assessment of your diabetes risk based on key health indicators including glucose levels, BMI, age, and family history.
              </p>
              <button
                onClick={() => setCurrentPage('diabetes')}
                className="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
              >
                Start Diabetes Assessment
              </button>
            </div>
            
            <div className="bg-white rounded-xl shadow-lg p-8 opacity-75">
              <div className="flex items-center mb-4">
                <Heart className="w-8 h-8 text-gray-400 mr-3" />
                <h3 className="text-2xl font-semibold text-gray-400">More Assessments Coming Soon</h3>
              </div>
              <p className="text-gray-500 mb-6">
                Additional health assessment tools will be available in future updates.
              </p>
              <button
                disabled
                className="w-full bg-gray-300 text-gray-500 font-semibold py-3 px-6 rounded-lg cursor-not-allowed"
              >
                Coming Soon
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-100">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center">
                <Activity className="w-8 h-8 text-green-500 mr-3" />
                <h1 className="text-3xl font-bold text-gray-800">Diabetes Risk Prediction</h1>
              </div>
              <button
                onClick={() => setCurrentPage('home')}
                className="text-blue-500 hover:text-blue-700 font-medium"
              >
                ← Back to Home
              </button>
            </div>
            
            <div className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Pregnancies
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="1"
                    value={formData.pregnancies}
                    onChange={(e) => handleInputChange('pregnancies', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Age (years) *
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="120"
                    required
                    value={formData.age}
                    onChange={(e) => handleInputChange('age', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Blood Pressure (mmHg)
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={formData.bloodpressure}
                    onChange={(e) => handleInputChange('bloodpressure', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Skin Thickness (mm)
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={formData.skinthickness}
                    onChange={(e) => handleInputChange('skinthickness', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Glucose Level (mg/dL) *
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    required
                    value={formData.glucose}
                    onChange={(e) => handleInputChange('glucose', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Insulin Level (μU/mL)
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={formData.insulin}
                    onChange={(e) => handleInputChange('insulin', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    BMI (Body Mass Index)
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    value={formData.bmi}
                    onChange={(e) => handleInputChange('bmi', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Diabetes Pedigree Function
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.001"
                    value={formData.diabetespedigreefunction}
                    onChange={(e) => handleInputChange('diabetespedigreefunction', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
              </div>
              
              {error && (
                <div className="flex items-center p-4 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <span className="text-red-700">{error}</span>
                </div>
              )}
              
              <div className="flex gap-4">
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={loading}
                  className="flex-1 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white font-semibold py-3 px-6 rounded-lg transition-colors"
                >
                  {loading ? 'Analyzing...' : 'Predict Diabetes Risk'}
                </button>
                
                <button
                  type="button"
                  onClick={resetForm}
                  className="px-6 py-3 border border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors"
                >
                  Reset
                </button>
              </div>
            </div>
            
            {prediction && (
              <div className="mt-8 p-6 bg-gradient-to-r from-blue-50 to-green-50 border border-blue-200 rounded-lg">
                <h3 className="text-xl font-semibold text-gray-800 mb-3">Prediction Result</h3>
                <div className="space-y-2">
                  <p className="text-lg">
                    <span className="font-medium">Result:</span>{' '}
                    <span className={`font-semibold ${
                      prediction.label.includes('Positive') ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {prediction.label}
                    </span>
                  </p>
                  {prediction.confidence && (
                    <p className="text-lg">
                      <span className="font-medium">Confidence:</span>{' '}
                      <span className="font-semibold text-blue-600">{prediction.confidence}%</span>
                    </p>
                  )}
                </div>
                <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded">
                  <p className="text-sm text-yellow-800">
                    <strong>Disclaimer:</strong> This is a simplified prediction model for demonstration purposes. 
                    Always consult with healthcare professionals for accurate medical diagnosis and advice.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DiabetesPredictorApp;
