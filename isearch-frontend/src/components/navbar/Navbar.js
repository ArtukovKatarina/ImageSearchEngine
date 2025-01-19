import React from "react";
import './Navbar.css';
import {Link} from "react-router-dom";

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="navbar-logo">
            </div>
            <div className="navbar-buttons">
                <Link className="navbar-button" to="/">Search images</Link>
                <Link className="navbar-button" to="/upload">Upload new image</Link>
            </div>
        </nav>
    );
};

export default Navbar;
