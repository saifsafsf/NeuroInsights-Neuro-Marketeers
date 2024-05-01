import React, { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import '../styles/ResultsPage.css';
import Navbar from '../components/Navbar.js';

const ResultsPage = () => {
  const { state } = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    // Check if state is null
    if (!state) {
      // Navigate to a different page or do any other action
      navigate('/dashboard');
    }
  }, [state, navigate]);

  // If state is not provided, redirect to the Home page
  if (!state) {
    return null; // This will prevent the rest of the component from rendering while the navigation takes effect
  }

  const { predicted_para } = state;

  return (
    <>
    <Navbar />
    <div className='outer'>
      <div className="results-container">
        <div className="result">
          <h2>Predicted Paragraph</h2>
          <p className="result-paragraph">{predicted_para}</p>
        </div>
        <button className="home-button" onClick={() => navigate('/dashboard')}>Home</button>
        <button className="home-button" onClick={() => navigate('/home')}>Upload EEG</button>
      </div>
    </div>
    </>
  );
};

export default ResultsPage;
