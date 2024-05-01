import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../styles/Dashboard.css';
import Navbar from '../components/Navbar.js';

const Dashboard = () => {
  const [results, setResults] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const { data } = await axios.get('http://localhost:5000/get_user_results', {
          headers: { Authorization: `Bearer ${localStorage.getItem('eeg_token')}` }
        });
        setResults(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  const handleView = (resultId) => {
    navigate(`/results/${resultId}`);
  };

  const handleDelete = async (resultId) => {
    const the_id = resultId.$oid
    if (window.confirm('Are you sure you want to delete this result?')) {
      try {
        await axios.delete(`http://localhost:5000/delete_result/${the_id}`, {
          headers: { Authorization: `Bearer ${localStorage.getItem('eeg_token')}` }
        });
        setResults(prevResults => prevResults.filter(result => result._id !== resultId)); // Update state to remove the deleted item
      } catch (error) {
        console.error('Failed to delete the result:', error);
        alert('Error deleting the result.');
      }
    }
  };

  return (
    <>
      <Navbar />
      <div className='outer'>
      <div className="dashboard-container">
        <h1>Results Dashboard</h1>
        <table>
          <thead>
            <tr>
              <th>Predicted Paragraph</th>
              <th>Upload Time</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {results.map(result => (
              <tr key={result._id}>
                <td className='example-cell'>{result.predicted_para}</td>
                <td className='example-cell'>{(new Date(result.upload_time.$date)).toLocaleString("en-US", {hour12: false})}</td>
                <td className='example-cell'>
                  <button onClick={() => handleDelete(result._id)} className="delete-button">Delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </div>
    </>
  );
};

export default Dashboard;
