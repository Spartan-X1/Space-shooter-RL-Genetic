from random import random, randint, uniform
from sprites import *

class WaveManager:
    def __init__(self, game):
        self.game = game
        self.current_wave = 0
        self.enemies_remaining = 0      
        self.total_enemies_to_spawn = 0 
        self.enemies_spawned = 0        
        self.wave_active = False
        self.wave_cooldown = 5.0
        self.cooldown_timer = 0
        self.base_spawn_interval = ENEMY_SPAWN_INTERVAL
        self.current_spawn_interval = self.base_spawn_interval
        
        # Wave announcement properties
        self.show_wave_message = False
        self.wave_message_timer = 0
        self.wave_message_duration = 3.0

         # Add a wave timeout to prevent stuck waves
        self.wave_timeout = 60.0  # 60 seconds maximum per wave
        self.wave_timer = 0

    def start_next_wave(self):
        """Start the next wave with increased difficulty"""
        self.current_wave += 1
        self.wave_active = True
        self.show_wave_message = True
        self.wave_message_timer = 0
        
        # Calculate enemies for this wave
        base_enemies = 5
        bonus_enemies = int(self.current_wave * 1.5)
        
        # IMPORTANT: Set both variables explicitly
        self.total_enemies_to_spawn = base_enemies + bonus_enemies
        self.enemies_remaining = self.total_enemies_to_spawn
        self.enemies_spawned = 0
        
        # Adjust spawn interval
        self.current_spawn_interval = max(0.5, self.base_spawn_interval - (self.current_wave * 0.1))
        
        # Update the spawn timer
        self.game.enemy_timer.duration = self.current_spawn_interval
        self.game.enemy_timer.activate()
        self.game.check_weapon_unlocks()
        print(f"Wave {self.current_wave} started! Total enemies: {self.total_enemies_to_spawn}")

    def spawn_enemy(self):
        """Spawn an enemy if we haven't reached the total to spawn"""
        # Check if we should spawn more enemies
        if not self.wave_active:
            print("Wave not active, not spawning")
            return None
            
        if self.enemies_spawned >= self.total_enemies_to_spawn:
            print(f"Already spawned {self.enemies_spawned}/{self.total_enemies_to_spawn}, not spawning more")
            return None
            
        # Create a wave-appropriate enemy
        enemy = self.create_wave_enemy()
        
        # Increment the spawn counter
        self.enemies_spawned += 1
        
        print(f"Spawning enemy {self.enemies_spawned}/{self.total_enemies_to_spawn} (Remaining to kill: {self.enemies_remaining})")
        return enemy
    
    def create_wave_enemy(self):
        """Create an enemy with wave-appropriate attributes"""
        # Determine enemy type based on wave
        enemy_type = "normal"  # Default

        # Use random chance to select enemy type based on wave number
        r = random()

        if self.current_wave >= 5:
            # Higher waves - full variety
            if r < 0.3:
                enemy_type = "swarm"
            elif r < 0.5:
                enemy_type = "tank"
            elif r < 0.7:
                enemy_type = "sniper"
            elif r < 0.9:
                enemy_type = "bomber"
        elif self.current_wave >= 3:
            # Wave 3-4 - introduce tanks and snipers
            if r < 0.4:
                enemy_type = "swarm"
            elif r < 0.7:
                enemy_type = "tank"
            elif r < 0.9:
                enemy_type = "sniper"
        elif self.current_wave >= 2:
            # Wave 2 - introduce swarms
            if r < 0.7:
                enemy_type = "swarm"

        # Create the enemy using the factory
        enemy = self.game.enemy_factory.create_enemy(enemy_type)
        
        # Set base health values - first wave always has 1 health
        if self.current_wave == 1:
            enemy.health = 1
        else:
            # Scale health with wave number
            if enemy_type == "normal":
                # Normal enemies: Wave 1: 1, Wave 2: 1, Wave 3: 2, etc.
                enemy.health = max(1, (self.current_wave - 1) // 1)
            elif enemy_type == "tank":
                # Tanks are always tougher (wave number + 2)
                enemy.health = min(3 + (self.current_wave - 1) // 1, 7)
            elif enemy_type == "swarm":
                # Swarms always have low health
                enemy.health = 1
            elif enemy_type == "sniper":
                # Snipers scale with wave
                enemy.health = min(1 + (self.current_wave - 1) // 1, 4)
            elif enemy_type == "bomber":
                # Bombers scale with wave
                enemy.health = min(2 + (self.current_wave - 1) // 1, 5)
        
        # Set debug type for visualization
        enemy.debug_type = enemy_type.upper()

        # Apply wave-based speed scaling (gentle increase)
        speed_multiplier = 1.0 + min(0.2, (self.current_wave - 1) * 0.05)
        enemy.speed *= speed_multiplier

        return enemy
    
    def enemy_destroyed(self):
        """Called when an enemy is destroyed"""
        self.enemies_remaining -= 1
        print(f"Enemy destroyed! Remaining to kill: {self.enemies_remaining}")
        
        # Check if wave is complete
        if self.enemies_remaining <= 0:
            self.wave_active = False
            self.cooldown_timer = 0
            
            # Give player a bonus
            self.game.score += 100 * self.current_wave
            
            print(f"Wave {self.current_wave} complete! Starting cooldown for next wave.")
    
    def update(self, dt):
        """Update the wave manager state"""
        # Handle wave message display
        if self.show_wave_message:
            self.wave_message_timer += dt
            if self.wave_message_timer >= self.wave_message_duration:
                self.show_wave_message = False
        
        # If wave is not active, handle cooldown before next wave
        if not self.wave_active:
            self.cooldown_timer += dt
            if self.cooldown_timer >= self.wave_cooldown:
                self.start_next_wave()
    
    def draw(self):
        """Draw wave information"""
        if self.show_wave_message:
            # Draw wave announcement
            wave_text = f"WAVE {self.current_wave}"
            text_size = measure_text_ex(self.game.assets['font'], wave_text, FONT_SIZE, 0)
            
            # Position in center of screen
            draw_text_ex(
                self.game.assets['font'], 
                wave_text, 
                Vector2(
                    (WINDOW_WIDTH / 2) - (text_size.x / 2),
                    (WINDOW_HEIGHT / 2) - (text_size.y / 2)
                ),
                FONT_SIZE, 
                0, 
                RED
            )
        
        # Always show current wave in top right
        current_wave_text = f"WAVE: {self.current_wave}"
        enemies_text = f"ENEMIES: {self.enemies_remaining}"
        
        # Use a smaller font size for the persistent display
        small_font_size = FONT_SIZE // 3
        
        # Draw wave number
        draw_text_ex(
            self.game.assets['font'],
            current_wave_text,
            Vector2(WINDOW_WIDTH - 250, 20),
            small_font_size,
            0,
            WHITE
        )
        
        # Draw enemies remaining
        draw_text_ex(
            self.game.assets['font'],
            enemies_text,
            Vector2(WINDOW_WIDTH - 250, 20 + small_font_size + 10),
            small_font_size,
            0,
            WHITE
        )
    
    def reset(self):
        """Reset the wave manager to initial state"""
        self.current_wave = 0
        self.enemies_remaining = 0
        self.total_enemies_to_spawn = 0
        self.enemies_spawned = 0
        self.wave_active = False
        self.cooldown_timer = 0
        self.current_spawn_interval = self.base_spawn_interval
        self.show_wave_message = False
        
        print("Wave manager reset to initial state")
        
        # Start first wave immediately
        self.start_next_wave()

class EnemyFactory:
    """Creates different types of enemies based on game progression"""
    def __init__(self, game):
        self.game = game
        
    def create_enemy(self, enemy_type):
        """Create an enemy of the specified type with its specialized weapon"""
        enemy = None
        if enemy_type == "tank":
            enemy = TankEnemy(
                self.game.assets['enemy_tank'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        elif enemy_type == "swarm":
            enemy = SwarmEnemy(
                self.game.assets['enemy_swarm'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        elif enemy_type == "sniper":
            enemy = SniperEnemy(
                self.game.assets['enemy_sniper'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        elif enemy_type == "bomber":
            enemy = BomberEnemy(
                self.game.assets['enemy_bomber'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser'],
                self.game
            )
        else:
            # Default to normal enemy
            enemy = Enemy(
                self.game.assets['enemy'],
                self.game.shoot_enemy_laser, 
                self.game.assets['enemy_laser']
            )
        
        if enemy:
            # Always set target player
            enemy.target_player = self.game.player
            
            # Add debug text to display enemy type (temporary solution)
            enemy.debug_type = enemy_type.upper()
        
        return enemy