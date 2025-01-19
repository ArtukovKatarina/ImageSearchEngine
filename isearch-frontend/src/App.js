import logo from './logo.svg';
import './App.css';
import SearchForImage from "./components/searchImages/SearchForImage";
import Navbar from "./components/navbar/Navbar";
import React from "react";
import {Route, Routes} from "react-router-dom";
import {UploadImage} from "./components/uploadImages/UploadImage";

function App() {
  return (
      <div>
          <Navbar/>
          <div className="App">
              <img
                  src="/isearch_logo.png"
                  alt="ISEarch logo"
                  className="logo-icon"
              />
              <Routes>
                  <Route path="/" element={<SearchForImage />} />
                  <Route path="/upload" element={<UploadImage /> } />
              </Routes>
          </div>
      </div>
  );
}

export default App;
