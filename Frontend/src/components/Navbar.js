import React from 'react';
import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';
import { useNavigate } from 'react-router-dom';

function Navbar_two() {
  const navigate = useNavigate(); // Hook to navigate

  const handleLogout = () => {
    localStorage.removeItem("eeg_token"); // Remove the token from local storage
    navigate("/"); // Navigate to the homepage or login page
  }

  return (
    <>
      <Navbar bg="primary" variant="dark">
        <Container>
          <Navbar.Brand href="/dashboard">NeuroInsights</Navbar.Brand>
          <Nav className="mr-auto">
            <Nav.Link href="/home">Upload EEG</Nav.Link>
            <Nav.Link onClick={handleLogout}>Logout</Nav.Link> {/* Use onClick to handle logout */}
          </Nav>
        </Container>
      </Navbar>
    </>
  );
}

export default Navbar_two;
