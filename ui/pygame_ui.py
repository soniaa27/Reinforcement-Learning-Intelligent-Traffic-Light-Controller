import pygame
import torch
import torch.nn as nn
import numpy as np
import random
import math

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, lr=3e-4):
        super(PPOAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50) # For light poles
LIGHT_OFF = (40, 40, 40) 
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)      # Car
ORANGE = (255, 165, 0)  # Truck
CYAN = (0, 255, 255)    # Bike
EV_WHITE = (255, 255, 255) # Emergency Vehicle
EV_FLASH_RED = (255, 0, 0)
EV_FLASH_BLUE = (0, 0, 255)

# Road Params
LANE_WIDTH = 50 # Width of ONE lane
NUM_LANES = 2 # Two lanes per direction
ROAD_WIDTH = LANE_WIDTH * NUM_LANES # Width of one side of the road
CENTER_X = SCREEN_WIDTH // 2
CENTER_Y = SCREEN_HEIGHT // 2

# Stop line distance from the center
STOP_LINE_DIST = (ROAD_WIDTH) + 20 

# Traffic Light Phases
PHASE_NS_GREEN = 0
PHASE_NS_YELLOW = 1
PHASE_EW_GREEN = 2
PHASE_EW_YELLOW = 3

# Vehicle Types
TYPE_CAR = 0
TYPE_TRUCK = 1
TYPE_BIKE = 2
TYPE_EMERGENCY = 3

VEHICLE_SPECS = {
    TYPE_CAR: {'size': (30, 15), 'color': BLUE, 'speed': 3.0},
    TYPE_TRUCK: {'size': (50, 20), 'color': ORANGE, 'speed': 1.5},
    TYPE_BIKE: {'size': (15, 8), 'color': CYAN, 'speed': 4.5},
    TYPE_EMERGENCY: {'size': (35, 18), 'color': EV_WHITE, 'speed': 5.0} 
}

class Vehicle:
    def __init__(self, direction):
        self.type = random.choices([TYPE_CAR, TYPE_TRUCK, TYPE_BIKE], weights=[0.6, 0.2, 0.2])[0]
        self.specs = VEHICLE_SPECS[self.type]
        self.width, self.height = self.specs['size']
        self.color = self.specs['color']
        self.base_speed = self.specs['speed']
        self.speed = self.base_speed
        self.direction = direction # 'N', 'S', 'E', 'W'
        self.stopped = False
        self.waiting_time = 0
        self.is_emergency = False 
        
        # Assign a lane (0 or 1)
        self.lane_index = random.randint(0, NUM_LANES - 1)
        lane_offset = LANE_WIDTH * 0.5 + self.lane_index * LANE_WIDTH

        # Spawn positions based on lane (Right-Hand Drive)
        if direction == 'N': # Moving North (starts at bottom)
            self.x = CENTER_X + lane_offset
            self.y = SCREEN_HEIGHT + 50
            self.dx, self.dy = 0, -1
            self.angle = 90
        elif direction == 'S': # Moving South (starts at top)
            self.x = CENTER_X - lane_offset
            self.y = -50
            self.dx, self.dy = 0, 1
            self.angle = 270
        elif direction == 'E': # Moving East (starts at left)
            self.x = -50
            self.y = CENTER_Y + lane_offset
            self.dx, self.dy = 1, 0
            self.angle = 0
        elif direction == 'W': # Moving West (starts at right)
            self.x = SCREEN_WIDTH + 50
            self.y = CENTER_Y - lane_offset
            self.dx, self.dy = -1, 0
            self.angle = 180

    def update(self, lights, vehicles_ahead):
        self.speed = self.base_speed 
        stop_for_light = False
        stop_for_vehicle = False

        # 1. Check for traffic light
        light_color = None
        stop_line = 0
        
        if self.direction == 'N':
            light_color = lights['NS']
            stop_line = CENTER_Y + STOP_LINE_DIST
            # Check if in stop zone and snap to line
            if self.y < stop_line + self.height/2 and self.y > stop_line: 
                if (light_color == RED or light_color == YELLOW) and not self.is_emergency:
                    stop_for_light = True
                    self.y = stop_line + self.height/2 
                    
        elif self.direction == 'S':
            light_color = lights['NS']
            stop_line = CENTER_Y - STOP_LINE_DIST
            if self.y > stop_line - self.height/2 and self.y < stop_line:
                if (light_color == RED or light_color == YELLOW) and not self.is_emergency:
                    stop_for_light = True
                    self.y = stop_line - self.height/2
                    
        elif self.direction == 'E':
            light_color = lights['EW']
            stop_line = CENTER_X - STOP_LINE_DIST
            if self.x > stop_line - self.width/2 and self.x < stop_line:
                if (light_color == RED or light_color == YELLOW) and not self.is_emergency:
                    stop_for_light = True
                    self.x = stop_line - self.width/2
                    
        elif self.direction == 'W':
            light_color = lights['EW']
            stop_line = CENTER_X + STOP_LINE_DIST
            if self.x < stop_line + self.width/2 and self.x > stop_line:
                if (light_color == RED or light_color == YELLOW) and not self.is_emergency:
                    stop_for_light = True
                    self.x = stop_line + self.width/2

        # 2. Check for vehicle ahead (in the same lane)
        for v in vehicles_ahead:
            if v is self or v.direction != self.direction or v.lane_index != self.lane_index:
                continue

            dist = float('inf')
            if self.direction == 'N': dist = self.y - v.y
            elif self.direction == 'S': dist = v.y - self.y
            elif self.direction == 'E': dist = v.x - self.x
            elif self.direction == 'W': dist = self.x - v.x

            # Safety gap
            safety_gap = (self.height * 2.5 if self.direction in ['N','S'] else self.width * 2.5)
            if 0 < dist < safety_gap: 
                stop_for_vehicle = True
                self.speed = v.speed 
                if v.stopped: self.speed = 0
                break

        # 3. Final decision
        if (stop_for_light or stop_for_vehicle) and not self.is_emergency:
            self.stopped = True
            self.waiting_time += 1/FPS
        else:
            self.stopped = False
            self.waiting_time = 0
            # Emergency vehicles can "ghost" through stopped cars
            if self.is_emergency and stop_for_vehicle:
                pass 
            self.x += self.dx * self.speed
            self.y += self.dy * self.speed

    def draw(self, surface, font):
        draw_color = self.color
        
        surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surf.fill(draw_color)
        
        if self.is_emergency:
            surf.fill(EV_WHITE) 
            flash_color = EV_FLASH_RED if int(pygame.time.get_ticks() / 250) % 2 == 0 else EV_FLASH_BLUE
            pygame.draw.rect(surf, flash_color, (self.width * 0.3, 0, self.width * 0.4, self.height))
            text_surf = font.render("EV", True, BLACK)
            text_rect = text_surf.get_rect(center=(self.width / 2, self.height / 2))
            surf.blit(text_surf, text_rect)
        
        rotated_surf = pygame.transform.rotate(surf, self.angle)
        rect = rotated_surf.get_rect(center=(self.x, self.y))
        surface.blit(rotated_surf, rect)

class TrafficSimulation:
    def __init__(self, model_path):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("RL Intelligent Traffic Controller")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)
        self.font_tiny = pygame.font.SysFont('Arial', 12, bold=True) # For EV text
        self.font_label = pygame.font.SysFont('Arial', 14, bold=True) # For light labels

        # Load Model
        self.state_dim = 30 
        self.action_dim = 4
        self.model = PPOAgent(self.state_dim, self.action_dim)
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

        self.vehicles = []
        self.current_phase = PHASE_NS_GREEN
        self.lights = {'NS': GREEN, 'EW': RED}
        self.spawn_timer = 0
        self.spawn_rate = 50 # Lower is faster spawning

    def get_state(self):
        cars = [v for v in self.vehicles if v.type == TYPE_CAR and not v.is_emergency]
        trucks = [v for v in self.vehicles if v.type == TYPE_TRUCK]
        bikes = [v for v in self.vehicles if v.type == TYPE_BIKE]
        
        stopped_vehicles = [v for v in self.vehicles if v.stopped and not v.is_emergency]
        total_waiting_time = sum(v.waiting_time for v in stopped_vehicles)
        avg_speed = np.mean([v.speed for v in self.vehicles if not v.stopped]) if self.vehicles else 0
        
        hour = 14.0; minute = 30.0
        
        state = [
            1.0, len(self.vehicles), avg_speed, len(cars), len(trucks), len(bikes),
            0.0, 25.0, 50.0, 0.0, 0.0, total_waiting_time,
            total_waiting_time / len(self.vehicles) if self.vehicles else 0,
            hour, minute, 0.0, 0.0, len(stopped_vehicles),
            len(self.vehicles) / 100.0, (len(self.vehicles) / 100.0) * 2,
            len(trucks) / len(self.vehicles) if self.vehicles else 0,
            len(bikes) / len(self.vehicles) if self.vehicles else 0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0 if self.current_phase in [0, 1] else 0.0,
            1.0 if self.current_phase in [2, 3] else 0.0,
        ]
        return np.array(state, dtype=np.float32)

    def update_lights(self, action):
        self.current_phase = action
        
        if action == 0: # NS Green
            self.lights['NS'] = GREEN; self.lights['EW'] = RED
        elif action == 1: # NS Yellow
            self.lights['NS'] = YELLOW; self.lights['EW'] = RED
        elif action == 2: # EW Green
            self.lights['NS'] = RED; self.lights['EW'] = GREEN
        elif action == 3: # EW Yellow
            self.lights['NS'] = RED; self.lights['EW'] = YELLOW

    def draw_roads(self):
        pygame.draw.rect(self.screen, BLACK, (CENTER_X - ROAD_WIDTH, 0, ROAD_WIDTH * 2, SCREEN_HEIGHT)) # NS
        pygame.draw.rect(self.screen, BLACK, (0, CENTER_Y - ROAD_WIDTH, SCREEN_WIDTH, ROAD_WIDTH * 2)) # EW
        
        for i in range(1, NUM_LANES):
            offset = LANE_WIDTH * i
            # NS Dashed
            for y in range(0, SCREEN_HEIGHT, 40):
                pygame.draw.line(self.screen, WHITE, (CENTER_X - offset, y), (CENTER_X - offset, y+20), 2)
                pygame.draw.line(self.screen, WHITE, (CENTER_X + offset, y), (CENTER_X + offset, y+20), 2)
            # EW Dashed
            for x in range(0, SCREEN_WIDTH, 40):
                pygame.draw.line(self.screen, WHITE, (x, CENTER_Y - offset), (x+20, CENTER_Y - offset), 2)
                pygame.draw.line(self.screen, WHITE, (x, CENTER_Y + offset), (x+20, CENTER_Y + offset), 2)

        # Centerlines (solid yellow)
        pygame.draw.line(self.screen, YELLOW, (CENTER_X, 0), (CENTER_X, CENTER_Y - ROAD_WIDTH), 3)
        pygame.draw.line(self.screen, YELLOW, (CENTER_X, SCREEN_HEIGHT), (CENTER_X, CENTER_Y + ROAD_WIDTH), 3)
        pygame.draw.line(self.screen, YELLOW, (0, CENTER_Y), (CENTER_X - ROAD_WIDTH, CENTER_Y), 3)
        pygame.draw.line(self.screen, YELLOW, (SCREEN_WIDTH, CENTER_Y), (CENTER_X + ROAD_WIDTH, CENTER_Y), 3)

        # Stop lines (solid white)
        pygame.draw.line(self.screen, WHITE, (CENTER_X - ROAD_WIDTH, CENTER_Y - STOP_LINE_DIST), (CENTER_X + ROAD_WIDTH, CENTER_Y - STOP_LINE_DIST), 5) # Top (for S-bound)
        pygame.draw.line(self.screen, WHITE, (CENTER_X - ROAD_WIDTH, CENTER_Y + STOP_LINE_DIST), (CENTER_X + ROAD_WIDTH, CENTER_Y + STOP_LINE_DIST), 5) # Bottom (for N-bound)
        pygame.draw.line(self.screen, WHITE, (CENTER_X - STOP_LINE_DIST, CENTER_Y - ROAD_WIDTH), (CENTER_X - STOP_LINE_DIST, CENTER_Y + ROAD_WIDTH), 5) # Left (for E-bound)
        pygame.draw.line(self.screen, WHITE, (CENTER_X + STOP_LINE_DIST, CENTER_Y - ROAD_WIDTH), (CENTER_X + STOP_LINE_DIST, CENTER_Y + ROAD_WIDTH), 5) # Right (for W-bound)

    def draw_traffic_light_pole(self, x, y, color_state):
        """Draws a vertical traffic light pole with all 3 slots"""
        pole_rect = pygame.Rect(0, 0, 22, 64)
        pole_rect.center = (x, y)
        pygame.draw.rect(self.screen, DARK_GRAY, pole_rect, border_radius=3)
        
        # Light states
        red_on = LIGHT_OFF
        yellow_on = LIGHT_OFF
        green_on = LIGHT_OFF
        
        if color_state == RED:
            red_on = RED
        elif color_state == YELLOW:
            yellow_on = YELLOW
        elif color_state == GREEN:
            green_on = GREEN
            
        pygame.draw.circle(self.screen, red_on, (x, y - 18), 7)
        pygame.draw.circle(self.screen, yellow_on, (x, y), 7)
        pygame.draw.circle(self.screen, green_on, (x, y + 18), 7)

    def draw_all_lights(self):
        ns_color = self.lights['NS']
        ew_color = self.lights['EW']
        
        # South-facing (at top, for N-S traffic)
        x, y = CENTER_X - ROAD_WIDTH - 20, CENTER_Y - STOP_LINE_DIST
        self.draw_traffic_light_pole(x, y, ns_color)
        label_surf = self.font_label.render("N-S", True, WHITE)
        self.screen.blit(label_surf, (x - label_surf.get_width() // 2, y - 35 - label_surf.get_height())) 

        # North-facing (at bottom, for S-N traffic)
        x, y = CENTER_X + ROAD_WIDTH + 20, CENTER_Y + STOP_LINE_DIST
        self.draw_traffic_light_pole(x, y, ns_color)
        label_surf = self.font_label.render("S-N", True, WHITE)
        self.screen.blit(label_surf, (x - label_surf.get_width() // 2, y + 35)) 

        # East-facing (at left, for W-E traffic) 
        x, y = CENTER_X - STOP_LINE_DIST, CENTER_Y + ROAD_WIDTH + 20
        self.draw_traffic_light_pole(x, y, ew_color)
        label_surf = self.font_label.render("W-E", True, WHITE)
        self.screen.blit(label_surf, (x - label_surf.get_width() // 2, y + 35)) 

        # West-facing (at right, for E-W traffic)
        x, y = CENTER_X + STOP_LINE_DIST, CENTER_Y - ROAD_WIDTH - 20
        self.draw_traffic_light_pole(x, y, ew_color)
        label_surf = self.font_label.render("E-W", True, WHITE)
        self.screen.blit(label_surf, (x - label_surf.get_width() // 2, y - 35 - label_surf.get_height())) # Above


    def run(self):
        running = True
        action_interval = 60 # Frames between model decisions (1 sec)
        frame_count = 0

        while running:
            self.screen.fill(GRAY)
            self.draw_roads()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            frame_count += 1
            if frame_count % action_interval == 0:
                state = self.get_state()
                action = self.model.act(state)
                self.update_lights(action)

            # Spawn Vehicles
            self.spawn_timer += 1
            if self.spawn_timer > self.spawn_rate: 
                self.spawn_timer = 0
                direction = random.choice(['N', 'S', 'E', 'W'])
                v = Vehicle(direction) # Create vehicle

                if random.random() < 0.02: # 2% chance of emergency
                    v.type = TYPE_EMERGENCY
                    v.specs = VEHICLE_SPECS[TYPE_EMERGENCY]
                    v.width, v.height = v.specs['size']
                    v.color = v.specs['color']
                    v.base_speed = v.specs['speed']
                    v.speed = v.base_speed
                    v.is_emergency = True
                
                self.vehicles.append(v)

            self.vehicles.sort(key=lambda v: v.y) 
            
            for v in self.vehicles[:]:
                v.update(self.lights, self.vehicles)
                v.draw(self.screen, self.font_tiny)
                if not (-50 <= v.x <= SCREEN_WIDTH + 50 and -50 <= v.y <= SCREEN_HEIGHT + 50):
                    self.vehicles.remove(v)

            self.draw_all_lights()

            info_text = f"Phase: {self.current_phase} | Vehicles: {len(self.vehicles)}"
            text_surf = self.font.render(info_text, True, WHITE)
            self.screen.blit(text_surf, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    TrafficSimulation("ppo_traffic_model_4500.pt").run()