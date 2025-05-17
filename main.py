import cv2
import mediapipe as mp
import pygame
import random

pygame.init()

# Constants
WIDTH, HEIGHT = 640, 480
CATCHER_WIDTH, CATCHER_HEIGHT = 100, 20
CIRCLE_RADIUS = 20
FPS = 60
FALL_SPEED = 5
MAX_LIVES = 10# Maximum number of lives
BALL_GENERATION_INTERVAL = 30  # Adjust this value to control the interval between new ball generations

# Colors
WHITE = (255, 255, 255)
RED = (0, 255, 0)

# Initialize Pygame screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Catcher Game")

# Load sound effects and music
pygame.mixer.music.load('./Assets/background_music.mp3')
catch_sound = pygame.mixer.Sound('./Assets/catch.mp3')
miss_sound = pygame.mixer.Sound('./Assets/miss.mp3')
game_over  = pygame.mixer.Sound('./Assets/game_over.mp3')

# Start the background music
pygame.mixer.music.play(-1)

# Load background image
background_image = pygame.image.load('Assets/background.jpg').convert()
background_image = pygame.transform.scale(background_image, (WIDTH, HEIGHT))

# Load catcher image
catcher_image = pygame.Surface((CATCHER_WIDTH, CATCHER_HEIGHT), pygame.SRCALPHA)
pygame.draw.rect(catcher_image, RED, (0, 0, CATCHER_WIDTH, CATCHER_HEIGHT))
catcher_rect = catcher_image.get_rect(center=(WIDTH // 2, HEIGHT - CATCHER_HEIGHT))

# Initialize Pygame clock
clock = pygame.time.Clock()

# Initialize hand tracking
hands = mp.solutions.hands.Hands()

def draw_circles(circles):
    for circle in circles:
        pygame.draw.circle(screen, WHITE, (circle[0], circle[1]), CIRCLE_RADIUS)

def show_start_screen():
    font = pygame.font.Font(None, 36)
    text = font.render("Press SPACE to Start", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)

def show_game_over_screen(score):
    font = pygame.font.Font(None, 36)
    text = font.render(f"Game Over - Score: {score}", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)

def draw_hand_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

def main():
    circles = []
    score = 0
    lives = MAX_LIVES
    game_active = False
    frame_counter = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw the hand landmarks on the frame
        draw_hand_landmarks(frame, results)

        # Display the hand landmarks window
        cv2.imshow('Hand Tracking', frame)

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                cap.release()
                cv2.destroyAllWindows()
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if not game_active:
                    game_active = True
                    circles = []
                    score = 0
                    lives = MAX_LIVES

        if not game_active:
            show_start_screen()
            pygame.display.flip()
            clock.tick(FPS)
            continue  # Skip the rest of the loop if the game is not active

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            cx, cy = int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH), int(hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)
            catcher_rect.center = (cx, catcher_rect.centery)

        frame_counter += 1
        if frame_counter >= BALL_GENERATION_INTERVAL:
            circles.append((random.randint(CIRCLE_RADIUS, WIDTH - CIRCLE_RADIUS), 0))
            frame_counter = 0

        for i, circle in enumerate(circles):
            circles[i] = (circle[0], circle[1] + FALL_SPEED + score * 0.1)  # Update the tuple with a new one

            if catcher_rect.colliderect(pygame.Rect(circle[0] - CIRCLE_RADIUS, circle[1] - CIRCLE_RADIUS, CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2)):
                circles.pop(i)
                catch_sound.play()
                score += 1

            if circle[1] > HEIGHT:
                circles.pop(i)
                miss_sound.play()
                lives -= 1
                if lives == 0:
                    game_active = False

        screen.blit(background_image, (0, 0))  # Draw the background image
        draw_circles(circles)
        screen.blit(catcher_image, catcher_rect.topleft)

        # Display score and remaining lives
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score} | Lives: {lives}", True, WHITE)
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

        if not game_active:
            pygame.time.delay(3000)  # Display game over screen for 3 seconds before quitting
            break

    show_game_over_screen(score)
    game_over.play()
    pygame.display.flip()
    pygame.time.delay(3000)  # Display game over screen for 3 seconds before quitting
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
