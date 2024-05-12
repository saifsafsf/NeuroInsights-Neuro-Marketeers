import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, Link } from 'react-router-dom';
import { notification } from 'antd'; 
import '../styles/Signup.css';
import title_picture from '../assets/NeuroInsights.png'

export default function Signup() {
    const [form, setForm] = useState({ username: '', password: '', confirmPassword: '' });
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

    const handleChange = (e) => {
        const { id, value } = e.target;
        setForm(prevForm => ({
            ...prevForm,
            [id]: value
        }));
    };

    const verifyToken = async (token) => {
      try {
          const config = {
              headers: {
                  Authorization: token,
              }
          };
          let response = await axios.post('http://172.208.115.234:8080/verify_token', null, config);
          if (response.status === 200) {
              navigate('/Home'); // Adjust the route as per your app's requirement
          }
      } catch (error) {
          console.log(error); // You might want to handle this error differently
      }
  };

  useEffect(() => {
      let token = localStorage.getItem('eeg_token');
      if (token) {
          verifyToken(token);
      }
  }, []);


    const handleSignup = async (e) => {
        e.preventDefault(); // Prevent the default form submission behavior
        const { username, password, confirmPassword } = form;

        if (password !== confirmPassword) {
            notification.error({
                message: "Password Mismatch",
                description: "The password and confirm password fields do not match.",
                duration: 3,
            });
            return;
        }

        if (!username || !password || !confirmPassword) {
            notification.error({
                message: 'Missing Fields',
                description: 'All fields are required.',
                duration: 3,
            });
            return;
        }

        setLoading(true);
        try {
            let response = await axios.post('http://172.208.115.234:8080/signup', { username, password });

            if (response.status === 201) {
                let token = response.data.token;
                localStorage.setItem('eeg_token', token);
                notification.success({
                    message: 'Signup Successful',
                    description: `Welcome ${username}!`,
                    duration: 3,
                });
                navigate('/Home'); // Adjust the navigation route as needed
            }
        } catch (error) {
            setLoading(false);
            notification.error({
                message: 'Signup Failed',
                description: (error.response && error.response.data && error.response.data.message) || 'Unable to sign up at this time.',
                duration: 3,
            });
        }
    };

    return (
        <div className='signUpPage'>
        <div className="wrapper signUp">
        <div className="illustration">
                <img src={title_picture} alt="Illustration" className="login-logo" />
            </div>
            <div className="form">
                <div className="heading">CREATE AN ACCOUNT</div>
                <form onSubmit={handleSignup}>
                    <div>
                        <label htmlFor="username">Username</label>
                        <input type="text" id="username" placeholder="Enter your username" value={form.username} onChange={handleChange} />
                    </div>
                    <div>
                        <label htmlFor="password">Password</label>
                        <input type="password" id="password" placeholder="Enter your password" value={form.password} onChange={handleChange} />
                    </div>
                    <div>
                        <label htmlFor="confirmPassword">Confirm Password</label>
                        <input type="password" id="confirmPassword" placeholder="Confirm your password" value={form.confirmPassword} onChange={handleChange} />
                    </div>
                    <button type="submit" disabled={loading}>
                        {loading ? 'Signing Up...' : 'Submit'}
                    </button>
                </form>
                <p>
                    Have an account? <Link to="/">Login</Link>
                </p>
            </div>
        </div>
        </div>
    );
}
