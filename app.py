import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import os
import json
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Game Constants
WIDTH, HEIGHT = 640, 480
CATCHER_WIDTH, CATCHER_HEIGHT = 100, 20
CIRCLE_RADIUS = 20
FPS = 30  # Reduced from 60 for better performance
FALL_SPEED = 5
MAX_LIVES = 3
BALL_GENERATION_INTERVAL = 30

# Game variables
circles = []
score = 0
lives = MAX_LIVES
game_active = False
frame_counter = 0
high_score = 0
catcher_x = WIDTH // 2

# Initialize hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Game state for streaming
webcam_frame = None
hand_landmarks = None
game_state = {
    "score": 0,
    "lives": MAX_LIVES,
    "game_active": False,
    "high_score": 0,
    "catcher_position": WIDTH // 2
}

# Thread lock
lock = threading.Lock()
camera_lock = threading.Lock()

# Camera instance (initialized once)
camera = None

def initialize_camera():
    """Initialize the camera once"""
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    return camera

def update_game_state():
    """Update the game state logic"""
    global circles, score, lives, game_active, frame_counter, high_score, game_state, catcher_x
    
    if not game_active:
        return
    
    # Increment frame counter for ball generation
    frame_counter += 1
    if frame_counter >= BALL_GENERATION_INTERVAL:
        circles.append([np.random.randint(CIRCLE_RADIUS, WIDTH - CIRCLE_RADIUS), 0])
        frame_counter = 0
    
    # Update catcher position from hand landmarks
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        catcher_x = int(index_finger_tip.x * WIDTH)
    
    # Update circles and check collisions
    circles_to_remove = []
    for i, circle in enumerate(circles):
        # Update circle position
        circle[1] += FALL_SPEED + min(score * 0.1, 10)  # Cap the speed increase
        
        # Check collision with catcher
        if (circle[1] + CIRCLE_RADIUS > HEIGHT - CATCHER_HEIGHT and 
            circle[1] - CIRCLE_RADIUS < HEIGHT and
            abs(circle[0] - catcher_x) < CATCHER_WIDTH / 2):
            circles_to_remove.append(i)
            score += 1
        
        # Check if circle has fallen off screen
        elif circle[1] - CIRCLE_RADIUS > HEIGHT:
            circles_to_remove.append(i)
            lives -= 1
            if lives <= 0:
                end_game()
    
    # Remove circles (in reverse order to avoid index issues)
    for i in sorted(circles_to_remove, reverse=True):
        if i < len(circles):
            circles.pop(i)
    
    # Update game state for the web interface
    game_state["score"] = score
    game_state["lives"] = lives
    game_state["game_active"] = game_active
    game_state["high_score"] = high_score
    game_state["catcher_position"] = catcher_x
    
    # Emit updated game state to clients
    socketio.emit('game_state_update', game_state)

def process_hand_tracking(frame):
    """Process hand tracking on the webcam frame"""
    global hand_landmarks
    
    # Process every other frame to reduce CPU usage
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # Store the first detected hand for game control
        hand_landmarks = results.multi_hand_landmarks[0]
    else:
        hand_landmarks = None
    
    return frame

def start_game():
    """Start or restart the game"""
    global circles, score, lives, game_active, frame_counter, high_score
    circles = []
    score = 0
    lives = MAX_LIVES
    game_active = True
    frame_counter = 0

def end_game():
    """End the game"""
    global game_active, high_score
    game_active = False
    
    # Update high score
    if score > high_score:
        high_score = score

def pause_game():
    """Pause the game"""
    global game_active
    game_active = False

def resume_game():
    """Resume the game"""
    global game_active
    game_active = True

def generate_webcam_frames():
    """Generator function for webcam feed with hand tracking"""
    frame_skip = 2  # Process every 2nd frame for better performance
    frame_count = 0
    cam = initialize_camera()
    
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Only process hand tracking on some frames
            frame_count += 1
            if frame_count % frame_skip == 0:
                frame = process_hand_tracking(frame)
                # Update game state on processed frames
                update_game_state()
            
            # Encode the frame for streaming (using JPEG quality reduction for performance)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control frame rate
            time.sleep(1/(FPS*1.5))  # Adjust for processing time
    except GeneratorExit:
        pass

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the webcam video feed with hand tracking"""
    return Response(generate_webcam_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_state')
def get_game_state():
    """Get the current game state"""
    return jsonify(game_state)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('game_state_update', game_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('start_game')
def handle_start_game():
    """Handle start game request"""
    start_game()
    emit('game_command_response', {'status': 'success', 'command': 'start'})

@socketio.on('pause_game')
def handle_pause_game():
    """Handle pause game request"""
    pause_game()
    emit('game_command_response', {'status': 'success', 'command': 'pause'})

@socketio.on('resume_game')
def handle_resume_game():
    """Handle resume game request"""
    resume_game()
    emit('game_command_response', {'status': 'success', 'command': 'resume'})

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)