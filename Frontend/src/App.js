// import { DndProvider } from 'react-dnd';
// import { HTML5Backend } from 'react-dnd-html5-backend';
import Login from './pages/Login';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import SignUp from './pages/Signup';
import Home from './pages/Home';
import ResultsPage from './pages/ResultsPage';
import Dashboard from './pages/Dashboard';


function App() {
  return (
    <div className="App">
      <div>
        <Router>
          <Routes>
            <Route exact path="/" element = {<Login />} />
            <Route exact path='/signup' element = {<SignUp />}/>
            <Route exact path='/home' element = {<Home />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
          </Routes>
        </Router>
      </div>
    </div>
  );
}

export default App;
