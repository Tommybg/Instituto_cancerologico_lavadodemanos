// Destructuring React hooks
const { useState, useEffect, useRef } = React;
 
// Hand washing step images
// Note: You'll need to place these images in your static/images folder
const stepImages = [
    '/static/images/step1.png', // Wet hands
    '/static/images/step2.png', // Apply soap
    '/static/images/step3.png', // Rub palms
    '/static/images/step4.png', // Rub backs
    '/static/images/step5.png', // Rub fingers
    '/static/images/step6.png'  // Rinse hands
];
 
// Constants for timing
const STEP_DURATION = 7; // seconds per step
const STEPS_COUNT = 6;
 
const HandWashingApp = () => {
    // State management
    const [currentStep, setCurrentStep] = useState(0);
    const [stepProgress, setStepProgress] = useState(Array(STEPS_COUNT).fill(0));
    const [completed, setCompleted] = useState(false);
    const [streamActive, setStreamActive] = useState(false);
    const [detectedStep, setDetectedStep] = useState(-1);
    const [isCorrectStep, setIsCorrectStep] = useState(false);
    const [stepTimer, setStepTimer] = useState(0); // Time spent on current step
    // New state for processed image and bounding boxes
    const [processedImage, setProcessedImage] = useState(null);
    const [boundingBoxes, setBoundingBoxes] = useState([]);
    const [showProcessedImage, setShowProcessedImage] = useState(false);
   
    // Refs
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const socketRef = useRef(null);
    const animationRef = useRef(null);
    const lastFrameTimeRef = useRef(0);
    const stepTimerRef = useRef(null);
   
    // Initialize socket connection and video stream
    useEffect(() => {
        socketRef.current = io();
       
        socketRef.current.on('connect', () => {
            console.log('Conectado al servidor');
        });
       
        socketRef.current.on('prediction_result', (data) => {
            setDetectedStep(data.step);
            setIsCorrectStep(data.step === currentStep);
           
            // Handle processed image with bounding boxes
            if (data.image) {
                setProcessedImage(data.image);
            }
           
            // Store bounding boxes data
            if (data.bounding_boxes) {
                setBoundingBoxes(data.bounding_boxes);
                console.log('Bounding boxes:', data.bounding_boxes);
            }
        });
       
        startCamera();
       
        return () => {
            stopCamera();
            if (socketRef.current) {
                socketRef.current.disconnect();
            }
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
            if (stepTimerRef.current) {
                clearInterval(stepTimerRef.current);
            }
        };
    }, []);
   
    // Handle step progression based on detected steps
    // Handle step progression based on detected steps
    useEffect(() => {
        if (completed) return;
       
        if (stepTimerRef.current) {
            clearInterval(stepTimerRef.current);
        }
       
        // Reset isCorrectStep when step changes to ensure proper detection for new step
        setIsCorrectStep(detectedStep === currentStep);
       
        // Start a timer to track progress on the current step
        if (streamActive) {
            stepTimerRef.current = setInterval(() => {
                // Only increment time if the correct step is detected
                if (isCorrectStep) {
                    setStepTimer(prevTime => {
                        const newTime = prevTime + 0.1; // 100ms interval
                       
                        // Calculate progress percentage for this step
                        const progress = (newTime / STEP_DURATION) * 100;
                       
                        // Update progress for this step
                        setStepProgress(prev => {
                            const newProgress = [...prev];
                            newProgress[currentStep] = Math.min(progress, 100);
                            return newProgress;
                        });
                       
                        // Check if we need to move to the next step (7 seconds reached)
                        if (newTime >= STEP_DURATION) {
                            // Move to next step or complete
                            if (currentStep < STEPS_COUNT - 1) {
                                // Clear the timer immediately
                                clearInterval(stepTimerRef.current);
                               
                                // Move to next step
                                setCurrentStep(prevStep => prevStep + 1);
                                setStepTimer(0); // Reset timer for next step
                            } else {
                                setCompleted(true);
                                setTimeout(() => {
                                    resetApp();
                                }, 5000); // Reset after 5 seconds
                            }
                            return 0; // Reset timer
                        }
                       
                        return newTime;
                    });
                } else {
                    // If incorrect step detected, don't increment timer
                    // and slowly decay progress to provide visual feedback
                    setStepProgress(prev => {
                        const newProgress = [...prev];
                        if (newProgress[currentStep] > 0) {
                            // Slowly decrease progress if wrong step detected
                            newProgress[currentStep] = Math.max(0, newProgress[currentStep] - 1);
                        }
                        return newProgress;
                    });
                }
            }, 100); // Update every 100ms
        }
       
        return () => {
            if (stepTimerRef.current) {
                clearInterval(stepTimerRef.current);
            }
        };
    }, [currentStep, isCorrectStep, streamActive, completed, detectedStep]);
 
   
    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            });
           
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                setStreamActive(true);
                animationRef.current = requestAnimationFrame(captureFrame);
            }
        } catch (err) {
            console.error('Error al acceder a la cámara:', err);
            alert('Error al acceder a la cámara. Por favor verifique los permisos.');
        }
    };
   
    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
            setStreamActive(false);
        }
    };
   
    const captureFrame = (timestamp) => {
        if (!videoRef.current || !canvasRef.current || !socketRef.current) {
            animationRef.current = requestAnimationFrame(captureFrame);
            return;
        }
       
        // Throttle frame rate to 3 fps to give the model more time to process
        if (timestamp - lastFrameTimeRef.current < 333) {
            animationRef.current = requestAnimationFrame(captureFrame);
            return;
        }
       
        lastFrameTimeRef.current = timestamp;
       
        try {
            const context = canvasRef.current.getContext('2d');
            const { videoWidth, videoHeight } = videoRef.current;
           
            // Make sure video is initialized
            if (!videoWidth || !videoHeight) {
                animationRef.current = requestAnimationFrame(captureFrame);
                return;
            }
           
            // Set canvas dimensions to match video
            canvasRef.current.width = videoWidth;
            canvasRef.current.height = videoHeight;
           
            // Draw video frame to canvas
            context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);
           
            // Add visual indicator for correct/incorrect step
            if (streamActive && currentStep >= 0) {
                // Add a status indicator
                const indicatorSize = 80;
                const padding = 20;
               
                context.save();
               
                // Draw a semi-transparent background
                context.fillStyle = isCorrectStep ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 165, 0, 0.3)';
                context.beginPath();
                context.arc(
                    padding + indicatorSize/2,
                    padding + indicatorSize/2,
                    indicatorSize/2,
                    0,
                    Math.PI * 2
                );
                context.fill();
               
                // Draw step number
                context.fillStyle = '#ffffff';
                context.font = 'bold 36px Arial';
                context.textAlign = 'center';
                context.textBaseline = 'middle';
                context.fillText(
                    (currentStep + 1).toString(),
                    padding + indicatorSize/2,
                    padding + indicatorSize/2
                );
               
                context.restore();
            }
           
            // Send frame to server for analysis at a lower quality to improve performance
            const imageData = canvasRef.current.toDataURL('image/jpeg', 0.5);
            socketRef.current.emit('video_frame', { image: imageData });
        } catch (err) {
            console.error('Error capturing frame:', err);
        }
       
        // Request next frame
        animationRef.current = requestAnimationFrame(captureFrame);
    };
   
    const resetApp = () => {
        setCurrentStep(0);
        setStepProgress(Array(STEPS_COUNT).fill(0));
        setCompleted(false);
        setStepTimer(0);
    };
   
    // Format time as seconds remaining
    const formatTimeRemaining = () => {
        const timeSpent = stepTimer;
        const timeRemaining = Math.max(0, STEP_DURATION - timeSpent);
        return `${Math.ceil(timeRemaining)}s`;
    };
   
    // Toggle between original and processed view
    const toggleView = () => {
        setShowProcessedImage(!showProcessedImage);
    };
   
    return (
        <div className="container mx-auto p-4 max-w-6xl">
            <h1 className="text-3xl font-bold text-center mb-6">
                Protocolo de Lavado de Manos
            </h1>
           
            <div className="flex flex-col md:flex-row gap-6">
                {/* Main content */}
                <div className="w-full md:w-2/3">
                    {/* Video feed and progress */}
                    <div className="relative bg-white rounded-lg shadow-lg overflow-hidden">
                        {/* Video display */}
                        <div className="relative">
                            {/* Show either the camera feed or the processed image based on state */}
                            {showProcessedImage && processedImage ? (
                                <img
                                    src={processedImage}
                                    className="w-full h-auto"
                                    alt="Processed feed with bounding boxes"
                                />
                            ) : (
                                <video
                                    ref={videoRef}
                                    className="w-full h-auto"
                                    muted
                                    playsInline
                                ></video>
                            )}
                           
                            <canvas
                                ref={canvasRef}
                                className="hidden" // Hidden canvas for processing
                            ></canvas>
                           
                            {/* Toggle view button */}
                            <button
                                onClick={toggleView}
                                className="absolute top-4 left-4 bg-blue-600 text-white px-3 py-1 rounded-md text-sm"
                            >
                                {showProcessedImage ? "Ver cámara" : "Ver detecciones"}
                            </button>
                           
                            {/* Overlay for incorrect step warning */}
                            {streamActive && !isCorrectStep && detectedStep !== -1 && (
                                <div className="absolute bottom-4 left-0 right-0 mx-auto w-4/5 bg-yellow-500 bg-opacity-80 text-white py-2 px-4 rounded-lg text-center">
                                    <p>Por favor realice el paso {currentStep + 1} correctamente</p>
                                </div>
                            )}
                           
                            {/* Display current detection information */}
                            {boundingBoxes.length > 0 && (
                                <div className="absolute top-16 left-4 bg-black bg-opacity-70 text-white p-2 rounded text-sm">
                                    <p>Detección: {boundingBoxes.length > 0 ? boundingBoxes[0].label : "Ninguna"}</p>
                                    <p>Confianza: {boundingBoxes.length > 0 ? `${(boundingBoxes[0].confidence * 100).toFixed(1)}%` : "0%"}</p>
                                </div>
                            )}
                           
                            {/* Completion overlay */}
                            {completed && (
                                <div className="absolute inset-0 bg-green-500 bg-opacity-50 flex items-center justify-center">
                                    <div className="bg-white p-8 rounded-lg shadow-lg">
                                        <h2 className="text-2xl font-bold text-green-600 text-center">
                                            ¡Lavado de manos exitoso!
                                        </h2>
                                    </div>
                                </div>
                            )}
                           
                            {/* Step timer indicator */}
                            <div className="absolute top-4 right-4">
                                <div className="w-16 h-16 rounded-full bg-white flex items-center justify-center shadow-lg">
                                    <svg className="w-14 h-14 circle-progress">
                                        <circle
                                            cx="28"
                                            cy="28"
                                            r="24"
                                            fill="none"
                                            stroke="#e0e0e0"
                                            strokeWidth="4"
                                        />
                                        <circle
                                            cx="28"
                                            cy="28"
                                            r="24"
                                            fill="none"
                                            stroke="#4CAF50"
                                            strokeWidth="4"
                                            strokeLinecap="round"
                                            strokeDasharray={`${stepProgress[currentStep]} 100`}
                                        />
                                    </svg>
                                    <div className="absolute text-sm font-bold">
                                        {formatTimeRemaining()}
                                    </div>
                                </div>
                            </div>
                        </div>
                       
                        {/* Steps indicators */}
                        <div className="px-4 py-6">
                            <div className="flex justify-between items-center">
                                {stepProgress.map((progress, index) => (
                                    <div
                                        key={index}
                                        className={`step-indicator flex flex-col items-center ${
                                            currentStep === index ? 'step-active' : ''
                                        } ${
                                            progress >= 100 ? 'step-completed' : ''
                                        }`}
                                    >
                                        <div className="relative mb-2">
                                            <div className={`w-16 h-16 rounded-full border-2 flex items-center justify-center overflow-hidden ${
                                                currentStep === index ? 'border-blue-500' : 'border-gray-300'
                                            }`}>
                                                <img
                                                    src={stepImages[index]}
                                                    alt={`Paso ${index + 1}`}
                                                    className="w-10 h-10 object-contain"
                                                />
                                               
                                                {/* Progress circle */}
                                                <svg className="absolute w-16 h-16 circle-progress">
                                                    <circle
                                                        cx="32"
                                                        cy="32"
                                                        r="30"
                                                        fill="none"
                                                        stroke="transparent"
                                                        strokeWidth="4"
                                                    />
                                                    <circle
                                                        cx="32"
                                                        cy="32"
                                                        r="30"
                                                        fill="none"
                                                        stroke={progress >= 100 ? '#4CAF50' : '#3B82F6'}
                                                        strokeWidth="4"
                                                        strokeLinecap="round"
                                                        strokeDasharray={`${progress} 100`}
                                                    />
                                                </svg>
                                            </div>
                                        </div>
                                        <span className="text-xs font-medium">Paso {index + 1}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
               
                {/* Tips section */}
                <div className="w-full md:w-1/3">
                    <div className="bg-white rounded-lg shadow-lg p-6">
                        <h2 className="text-xl font-bold mb-4 text-blue-600">Consejos para un lavado efectivo</h2>
                       
                        <ul className="space-y-4">
                            <li className="flex items-start">
                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center mr-3 mt-1">
                                    <span className="text-blue-600 text-sm font-bold">1</span>
                                </div>
                                <div>
                                    <h3 className="font-medium">Moje sus manos</h3>
                                    <p className="text-sm text-gray-600">Use agua corriente limpia y aplique jabón.</p>
                                </div>
                            </li>
                           
                            <li className="flex items-start">
                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center mr-3 mt-1">
                                    <span className="text-blue-600 text-sm font-bold">2</span>
                                </div>
                                <div>
                                    <h3 className="font-medium">Frote palma con palma</h3>
                                    <p className="text-sm text-gray-600">Asegúrese de cubrir todas las superficies.</p>
                                </div>
                            </li>
                           
                            <li className="flex items-start">
                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center mr-3 mt-1">
                                    <span className="text-blue-600 text-sm font-bold">3</span>
                                </div>
                                <div>
                                    <h3 className="font-medium">Palma contra dorso</h3>
                                    <p className="text-sm text-gray-600">Con los dedos entrelazados.</p>
                                </div>
                            </li>
                           
                            <li className="flex items-start">
                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center mr-3 mt-1">
                                    <span className="text-blue-600 text-sm font-bold">4</span>
                                </div>
                                <div>
                                    <h3 className="font-medium">Entre los dedos</h3>
                                    <p className="text-sm text-gray-600">Frote vigorosamente entre los dedos.</p>
                                </div>
                            </li>
                           
                            <li className="flex items-start">
                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center mr-3 mt-1">
                                    <span className="text-blue-600 text-sm font-bold">5</span>
                                </div>
                                <div>
                                    <h3 className="font-medium">Dorso de los dedos</h3>
                                    <p className="text-sm text-gray-600">Frote el dorso de los dedos.</p>
                                </div>
                            </li>
                           
                            <li className="flex items-start">
                                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center mr-3 mt-1">
                                    <span className="text-blue-600 text-sm font-bold">6</span>
                                </div>
                                <div>
                                    <h3 className="font-medium">Enjuague y seque</h3>
                                    <p className="text-sm text-gray-600">Enjuague con agua limpia y seque con una toalla limpia.</p>
                                </div>
                            </li>
                        </ul>
                       
                        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                            <h3 className="font-bold text-blue-700">Recuerde:</h3>
                            <p className="text-sm text-gray-700">Cada paso del lavado correcto de manos debe mantenerse por al menos 7 segundos para asegurar la eficacia.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
 
ReactDOM.render(<HandWashingApp />, document.getElementById('root'));