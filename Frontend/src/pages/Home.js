import React, { useCallback, useState, useEffect, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import '../styles/DragDrop.css';
import Navbar from '../components/Navbar.js';
import ClockLoader from "react-spinners/ClockLoader";

const Home = () => {
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [fileName, setFileName] = useState('');
  const [authToken, setAuthToken] = useState('');
  const navigate = useNavigate();
  const cancelSource = useRef(null);

  useEffect(() => {
    const token = localStorage.getItem('eeg_token');
    
    if (token) {
      setAuthToken(token);
    } else {
      navigate('/');
    }
  }, []); // Add navigate as a dependency here

  const onDrop = useCallback(acceptedFiles => {
    const file = acceptedFiles[0];
    if (file && file.name.endsWith('.mat')) {
      setFile(file);
      setFileName(file.name);
      setUploadSuccess(false);
      uploadFile(file); // Call upload function directly after file is set
    } else {
      alert('Only .mat files are allowed.');
      setFileName('');
      setFile(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: '.mat'
  });

  const uploadFile = async (file) => {
      setIsLoading(true);
      const formData = new FormData();
      formData.append("file", file);
      const aToken=localStorage.getItem('eeg_token')
      cancelSource.current = axios.CancelToken.source();

      try {
          const response = await axios.post('http://localhost:5000/upload_eeg', formData, {
              headers: {
                  'Authorization': `Bearer ${aToken}`,
                  'Content-Type': 'multipart/form-data'
              },
              cancelToken: cancelSource.current.token
          });
          setUploadSuccess(true);
          navigate('/results', { state: { ...response.data } });
      } catch (error) {
          if (axios.isCancel(error)) {
            console.log('Request canceled', error.message);
          } else {
            console.error('Error uploading file:', error);
            alert('Failed to upload file.');
            setUploadSuccess(false);
          }
      } finally {
          setIsLoading(false);
      }
  };

  const cancelUpload = () => {
    if (cancelSource.current) {
      cancelSource.current.cancel('Operation canceled by the user.');
    }
    setIsLoading(false);
    setFile(null);
    setFileName('');
    setUploadSuccess(false);
  };

  const override = {
    display: "block",
    margin: "20px",
    borderColor: "red",
  };

  return (
    <>
    <Navbar />
    <div className="drag-drop-container">
      <div {...getRootProps()} className="dropzone">
        <input {...getInputProps()} />
        {isDragActive ?
          <p>Drop the file here...</p> :
          <p>Drag 'n' drop a .mat file here, or click to select file</p>
        }
      </div>
      {fileName && uploadSuccess && (
        <p className="upload-info">Uploaded: {fileName}</p>
      )}
      {isLoading && (
        <ClockLoader
          color="#ffffff"
          loading={isLoading}
          cssOverride={override}
          size={35}
          aria-label="Loading Spinner"
          data-testid="loader"
        />
      )}
      {isLoading && (
        <button onClick={cancelUpload} className="cancel-button">
          Cancel Upload
        </button>
      )}
    </div>
    </>
  );
};

export default Home;
