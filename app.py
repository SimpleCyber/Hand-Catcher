import cv2
import mediapipe as mp
import pygame
import random
import numpy as np
import threading
import time
import os
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__, static_folder='static', template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*")

# Game Constants
WIDTH, HEIGHT = 640, 480
CATCHER_WIDTH, CATCHER_HEIGHT = 100, 20
CIRCLE_RADIUS = 20
FPS = 30  # Reduced from 60 to lower CPU usage
FALL_SPEED = 5
MAX_LIVES = 3
BALL_GENERATION_INTERVAL = 30

# Colors
WHITE = (255, 255, 255)
RED = (0, 255, 0)

# Game variables
circles = []
score = 0
lives = MAX_LIVES
game_active = False
frame_counter = 0
high_score = 0

# Initialize hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Mock webcam frame (for cloud deployment)
mock_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
cv2.putText(mock_frame, "No Camera Available", (80, HEIGHT//2), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(mock_frame, "Use Mouse/Touch Instead", (70, HEIGHT//2 + 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Initialize pygame (headless mode)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Use dummy audio driver to avoid ALSA errors
pygame.init()
pygame.display.set_mode((1, 1))

# Disable pygame mixer to avoid audio errors
pygame.mixer.quit()

# Create a surface for the game
game_surface = pygame.Surface((WIDTH, HEIGHT))

# Create dummy sound objects
class DummySound:
    def play(self, loops=0): pass
    def stop(self): pass

catch_sound = DummySound()
miss_sound = DummySound()
game_over_sound = DummySound()
background_music = DummySound()

# Game state for streaming
webcam_frame = mock_frame.copy()
game_frame = None
hand_landmarks = None
game_state = {
    "score": 0,
    "lives": MAX_LIVES,
    "game_active": False,
    "high_score": 0
}

# Mouse position for control in cloud environment
mouse_position = WIDTH // 2

# Thread lock
lock = threading.Lock()

def draw_circles(surface):
    """Draw circles on the game surface"""
    for circle in circles:
        pygame.draw.circle(surface, WHITE, (circle[0], circle[1]), CIRCLE_RADIUS)

def draw_catcher(surface, x_position):
    """Draw the catcher at the specified x position"""
    pygame.draw.rect(surface, RED, (int(x_position - CATCHER_WIDTH / 2), HEIGHT - CATCHER_HEIGHT, CATCHER_WIDTH, CATCHER_HEIGHT))

def update_game_state():
    """Update the game state"""
    global circles, score, lives, game_active, frame_counter, high_score, game_state
    
    if not game_active:
        return
    
    # Increment frame counter for ball generation
    frame_counter += 1
    if frame_counter >= BALL_GENERATION_INTERVAL:
        circles.append((random.randint(CIRCLE_RADIUS, WIDTH - CIRCLE_RADIUS), 0))
        frame_counter = 0
    
    # Get current catcher position from hand landmarks or mouse position
    catcher_x = mouse_position  # Default to mouse position
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        catcher_x = int(index_finger_tip.x * WIDTH)
    
    # Update circles and check collisions
    circles_to_remove = []
    for i, circle in enumerate(circles):
        new_y = circle[1] + FALL_SPEED + score * 0.1
        circles[i] = (circle[0], new_y)
        
        # Check collision with catcher
        if (new_y + CIRCLE_RADIUS > HEIGHT - CATCHER_HEIGHT and 
            new_y - CIRCLE_RADIUS < HEIGHT and
            abs(circle[0] - catcher_x) < CATCHER_WIDTH / 2):
            circles_to_remove.append(i)
            score += 1
            catch_sound.play()
        
        # Check if circle has fallen off screen
        elif new_y - CIRCLE_RADIUS > HEIGHT:
            circles_to_remove.append(i)
            lives -= 1
            miss_sound.play()
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
    
    # Emit updated game state to clients
    socketio.emit('game_state_update', game_state)

def render_game_frame():
    """Render the current game state to a frame"""
    global game_frame
    
    # Clear the surface
    game_surface.fill((0, 0, 0))
    
    # Draw game elements
    draw_circles(game_surface)
    
    # Get current catcher position from hand landmarks or mouse position
    catcher_x = mouse_position  # Default to mouse position
    if hand_landmarks:
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        catcher_x = int(index_finger_tip.x * WIDTH)
    
    # Draw catcher
    draw_catcher(game_surface, catcher_x)
    
    # Draw score and lives
    font = pygame.font.Font(None, 36)
    text = font.render(f"Score: {score} | Lives: {lives}", True, WHITE)
    game_surface.blit(text, (10, 10))
    
    # If game is not active, show appropriate message
    if not game_active:
        if lives <= 0:
            text = font.render(f"Game Over - Score: {score}", True, WHITE)
        else:
            text = font.render("Press Start to Play", True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        game_surface.blit(text, text_rect)
    
    # Convert Pygame surface to OpenCV image
    view = pygame.surfarray.array3d(game_surface)
    view = view.transpose([1, 0, 2])
    
    with lock:
        game_frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

def process_hand_tracking(frame):
    """Process hand tracking on the webcam frame"""
    global hand_landmarks, webcam_frame
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)
    
    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
        
        # Store the first detected hand for game control
        hand_landmarks = results.multi_hand_landmarks[0]
    else:
        hand_landmarks = None
    
    with lock:
        webcam_frame = frame.copy()

def start_game():
    """Start or restart the game"""
    global circles, score, lives, game_active, frame_counter, high_score
    circles = []
    score = 0
    lives = MAX_LIVES
    game_active = True
    frame_counter = 0
    background_music.play(-1)

def end_game():
    """End the game"""
    global game_active, high_score
    game_active = False
    game_over_sound.play()
    
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
    """Generator function for webcam feed"""
    # Try to access camera first
    camera_available = False
    cap = None
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            camera_available = True
            print("Camera available, using real webcam feed.")
        else:
            print("No camera available, using mock webcam feed.")
    except Exception as e:
        print(f"Error accessing camera: {e}")
        print("Using mock webcam feed instead.")
    
    try:
        while True:
            if camera_available:
                ret, frame = cap.read()
                if not ret:
                    # Switch to mock frame if camera fails
                    frame = mock_frame.copy()
                else:
                    # Flip frame horizontally for a mirror effect
                    frame = cv2.flip(frame, 1)
                    # Process hand tracking
                    process_hand_tracking(frame)
            else:
                # Use mock frame
                frame = mock_frame.copy()
            
            # Encode the frame for streaming
            with lock:
                if camera_available:
                    current_frame = webcam_frame
                else:
                    current_frame = frame
                
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Limit frame rate
            time.sleep(1/FPS)
    except GeneratorExit:
        if cap and cap.isOpened():
            cap.release()

def generate_game_frames():
    """Generator function for game frames"""
    while True:
        try:
            # Update game logic
            update_game_state()
            
            # Render game frame
            render_game_frame()
            
            # Encode the frame for streaming
            with lock:
                if game_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', game_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Limit frame rate
            time.sleep(1/FPS)
        except Exception as e:
            print(f"Error in game frame generation: {e}")
            # Create an error frame
            error_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            cv2.putText(error_frame, "Game Engine Error", (150, HEIGHT//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', error_frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1)

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for the webcam video feed with hand tracking"""
    return Response(generate_webcam_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_feed')
def game_feed():
    """Route for the game video feed"""
    return Response(generate_game_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/game_state')
def get_game_state():
    """Get the current game state"""
    return jsonify(game_state)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')

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

@socketio.on('mouse_move')
def handle_mouse_move(data):
    """Handle mouse move event"""
    global mouse_position
    try:
        mouse_position = data['x']
    except Exception as e:
        print(f"Error processing mouse move: {e}")

if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)