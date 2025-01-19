import React, { useState } from "react";
import axios from "axios";
import './SearchForImage.css';
import {Loader} from "../loader/Loader";

function SearchForImage() {
    const [query, setQuery] = useState("");       // State for the search query
    const [images, setImages] = useState([]);     // State for storing the returned image data
    const [error, setError] = useState(null);     // Error handling
    const [selectedImage, setSelectedImage] = useState(null);
    const [loading, setLoading] = useState(false); // State to track loading

    const handleImageUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => {
                setSelectedImage(reader.result);
            }
            reader.readAsDataURL(file);
        }
    }

    const handleSearch = async () => {
        setLoading(true);
        try {
            let requestData = {};

            if (selectedImage) {
                requestData = {
                    query: selectedImage.split(",")[1], // Remove data:image/jpeg;base64 prefix
                    n: 10
                }
                const response = await axios.post("http://127.0.0.1:8000/search_by_image", requestData);
                setImages(response.data.images);
            } else  {
                requestData = {
                    query: query,
                    n: 10
                };
                const response = await axios.post("http://127.0.0.1:8000/search_by_text", requestData);
                setImages(response.data.images);
            }
        } catch (err) {
            setError("Error fetching images.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleQueryChange = (event) => {
        setQuery(event.target.value);
        if (event.target.value) {
            setSelectedImage(null);
        }
    }

    function downloadImage(base64Data, fileName) {
        const linkSource = `data:image/jpeg;base64` +base64Data;
        const downloadLink = document.createElement("a");
        downloadLink.href = linkSource;
        downloadLink.download = fileName;
        downloadLink.click();
    }

    return (
        <div action="" className="search-container">
            <div className="wrap">
                <div className="search">
                    <input type="text" className="searchTerm" placeholder="What are you looking for?" value={query} onChange={handleQueryChange} />
                    <button type="submit" className="searchButton" onClick={handleSearch}>
                        <img
                            src="/search.svg"
                            alt="Search Icon"
                            className="add-photo-icon"
                        />
                    </button>
                    {!query && (
                        <label className="image-upload">
                            <input
                                type="file"
                                accept="image/*"
                                onChange={handleImageUpload}
                            />
                            <img
                                src="/add_photo_icon.svg"
                                alt="Add Photo Alternate Icon"
                                className="add-photo-icon"
                            />
                        </label>
                    )}
                </div>
            </div>

            {/*<div action="" className="search-bar">*/}
            {/*    <input*/}
            {/*        type="search"*/}
            {/*        name="search"*/}
            {/*        pattern=".*\S.*"*/}
            {/*        required*/}
            {/*        value={query}*/}
            {/*        onChange={handleQueryChange}*/}
            {/*        placeholder="Enter text or upload an image"*/}
            {/*    />*/}

            {/*<button className="search-btn" type="submit" onClick={handleSearch}/>*/}


            {/*</div>*/}

            {selectedImage && (
                <div>
                    <p className="description"> You are looking for images similar to this one:</p>
                    <img src={selectedImage} alt="Selected" className="selected-img"/>
                </div>
            )}

            {error && <p>{error}</p>}

            {loading ? (
                <Loader/>
            ) : (
                images && images.length > 0 && (
                    <div>
                        <p className="results"> Results:</p>
                        <div className="image-grid">
                            {images.map((imageData, index) => (
                                <div className="card" key={index}>
                                    <img
                                        src={imageData}
                                        alt={`Search result ${index}`}
                                        className="image-item"
                                    />
                                    <button className="download-btn"
                                            onClick={() => downloadImage(imageData, `image_${index}.jpg`)}>
                                        Download
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}
        </div>
    );
}

export default SearchForImage;
