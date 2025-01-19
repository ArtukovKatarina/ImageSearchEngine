import {useCallback, useState} from "react";
import axios from "axios";
import {useDropzone} from "react-dropzone";

export const UploadImage = () => {
    const [uploadProgress, setUploadProgress] = useState(0);
    const [isUploading, setIsUploading] = useState(false);

    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0];
        convertToBase64(file); // Convert file to Base64 string
    });

    const convertToBase64 = (file) => {
        const reader = new FileReader();
        reader.readAsDataURL(file); // Read file as Base64
        reader.onloadend = () => {
            const base64String = reader.result.split(',')[1]; // Extract the Base64 part from the result
            uploadFile(base64String); // Send the Base64 image string to the server
        };
        reader.onerror = (error) => {
            console.error('Error converting file to base64:', error);
        };
    };

    const uploadFile = (base64String) => {
        setIsUploading(true);

        let requestData = {
            image: base64String,
        }

        axios.post('http://127.0.0.1:8000/insert_image', requestData, {
            onUploadProgress: (progressEvent) => {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                setUploadProgress(percentCompleted);
            },
        }).then(() => {
            setIsUploading(false);
            setUploadProgress(0);
        }).catch(() => {
            setIsUploading(false);
        });
    };

    const { getRootProps, getInputProps, open } = useDropzone({
        onDrop,
        accept: 'image/jpeg, image/png',
        noClick: true, // Prevent file dialog from opening on click
    });

    return (
        <>
            <br/>
            <div className="upload-box" style={styles.uploadBox} {...getRootProps()}>
                <input {...getInputProps()} />
                <p>Drop your image here, or
                    <span style={styles.browseText} onClick={open}> browse </span>
                </p>
                {isUploading && (
                    <div style={styles.progressContainer}>
                        <p>Uploading... {uploadProgress}%</p>
                        <progress value={uploadProgress} max="100" style={styles.progressBar}></progress>
                    </div>
                )}
            </div>
        </>
    )
}

const styles = {
    uploadBox: {
        width: '40rem',
        height: '20rem',
        border: '5px dashed #ccc',
        borderRadius: '30px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#f9f9f9',
    },
    iconContainer: {
        marginBottom: '10px',
    },
    logoIcon: {
        width: '50px',
        height: '50px',
    },
    browseText: {
        color: 'blue',
        textDecoration: 'underline',
        cursor: 'pointer',
    },
    progressContainer: {
        marginTop: '10px',
        textAlign: 'center',
    },
    progressBar: {
        width: '100%',
        marginTop: '5px',
    },
};