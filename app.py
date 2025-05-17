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

# Game Constants (same as your original game)
WIDTH, HEIGHT = 640, 480
CATCHER_WIDTH, CATCHER_HEIGHT = 100, 20
CIRCLE_RADIUS = 20
FPS = 60
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

# Initialize pygame for game logic (headless mode)
os.environ['SDL_VIDEODRIVER'] = 'dummy'
pygame.init()
pygame.display.set_mode((1, 1))

# Create a surface for the game
game_surface = pygame.Surface((WIDTH, HEIGHT))

# Load sound effects (paths might need adjustment)
try:
    pygame.mixer.init()
    catch_sound = pygame.mixer.Sound('./static/assets/catch.mp3')
    miss_sound = pygame.mixer.Sound('./static/assets/miss.mp3')
    game_over_sound = pygame.mixer.Sound('./static/assets/game_over.mp3')
    background_music = pygame.mixer.Sound('./static/assets/background_music.mp3')
except Exception as e:
    print(f"Error loading sounds: {e}")
    # Create dummy sound objects
    class DummySound:
        def play(self): pass
    catch_sound = DummySound()
    miss_sound = DummySound()
    game_over_sound = DummySound()
    background_music = DummySound()

# Game state for streaming
webcam_frame = None
game_frame = None
hand_landmarks = None
game_state = {
    "score": 0,
    "lives": MAX_LIVES,
    "game_active": False,
    "high_score": 0
}

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
    """Update the game state, similar to your original game logic"""
    global circles, score, lives, game_active, frame_counter, high_score, game_state
    
    if not game_active:
        return
    
    # Increment frame counter for ball generation
    frame_counter += 1
    if frame_counter >= BALL_GENERATION_INTERVAL:
        circles.append((random.randint(CIRCLE_RADIUS, WIDTH - CIRCLE_RADIUS), 0))
        frame_counter = 0
    
    # Get current catcher position from hand landmarks
    catcher_x = WIDTH // 2  # Default position
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
    
    # Get current catcher position from hand landmarks
    catcher_x = WIDTH // 2  # Default position
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

def get_available_camera(max_index=10):
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Using camera index: {index}")
            return cap
        cap.release()
    print("No camera found.")
    return None

def generate_webcam_frames():
    """Generator function for webcam feed with hand tracking"""
    cap = get_available_camera()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for a mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process hand tracking
            process_hand_tracking(frame)
            
            # Encode the frame for streaming
            with lock:
                if webcam_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', webcam_frame)
                    if ret:
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Limit frame rate
            time.sleep(1/FPS)
    except GeneratorExit:
        cap.release()

def generate_game_frames():
    """Generator function for game frames"""
    while True:
        # Update game logic
        update_game_state()
        
        
        
        # Encode the frame for streaming
        with lock:
            # Render game frame
            render_game_frame()
            if game_frame is not None:
                ret, buffer = cv2.imencode('.jpg', game_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Limit frame rate
        time.sleep(1/FPS)

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

if __name__ == '__main__':
    # Start the game in a separate thread
    game_thread = threading.Thread(target=update_game_state)
    game_thread.daemon = True
    game_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 8000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)