import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate, Link } from 'react-router-dom';
import { notification } from 'antd'; // Make sure to have `antd` installed for notifications
import logo from '../assets/logo.svg';
import title_picture from '../assets/NeuroInsights.png'
import '../styles/Login.css';

export default function Login() {
    const [form, setForm] = useState({ username: '', password: '' });
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();

	const verifyToken = async (token) => {
        try {
            const config = {
                headers: {
                    Authorization: token,
                }
            };
            let response = await axios.post('http://localhost:5000/verify_token', null, config);
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

    const handleChange = (e) => {
        const { id, value } = e.target;
        setForm(prevForm => ({
            ...prevForm,
            [id]: value
        }));
    };

    const handleLogin = async (e) => {
        e.preventDefault(); // Prevent default form submission
        if (!form.username || !form.password) {
            notification.error({
                message: 'Missing Fields',
                description: 'Please fill all the fields',
                duration: 3,
            });
            return;
        }

        setLoading(true);
        try {
            let response = await axios.post('http://localhost:5000/login', form);

            if (response.status === 200) {
                let token = response.data.token;
                localStorage.setItem('eeg_token', token);
                navigate('/dashboard', { state: { token } });
            }
        } catch (error) {
            setLoading(false);
            if (error.response && error.response.status === 401) {
                notification.error({
                    message: 'Login Failed',
                    description: 'Invalid credentials. Please check and try again!',
                    duration: 3,
                });
            } else {
                notification.error({
                    message: 'Login Failed',
                    description: 'Unable to login. Please try again later!',
                    duration: 3,
                });
            }
        }
    };

    return (
        <div className="loginPage">
        <div className="wrapper signIn">
            <div className="illustration">
                <img src={title_picture} alt="Illustration" className="login-logo" />
            </div>
            <div className="form">
                <div className="heading">LOGIN</div>
                <form onSubmit={handleLogin}>
                    <div>
                        <label htmlFor="username">Username</label>
                        <input type="text" id="username" placeholder="Enter your username" value={form.username} onChange={handleChange} />
                    </div>
                    <div>
                        <label htmlFor="password">Password</label>
                        <input type="password" id="password" placeholder="Enter your password" value={form.password} onChange={handleChange} />
                    </div>
                    <button type="submit" disabled={loading}>
                        {loading ? 'Logging in...' : 'Submit'}
                    </button>
                </form>
                <p>
                    Don't have an account ? <Link to="/signup">Sign Up</Link>
                </p>
            </div>
        </div>
        </div>
    );
}
